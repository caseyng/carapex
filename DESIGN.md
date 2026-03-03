# carapex — Design

This document covers the architecture, the reasoning behind design decisions, failure semantics, what the threat model covers, and what it does not. It is written for security engineers evaluating carapex and researchers working in this area.

For installation and usage, see [README.md](README.md).

---

## What carapex Is

carapex is an application-layer prompt execution boundary. It intercepts every prompt before it reaches an LLM and every response before it reaches the caller. It applies a sequence of checks at both stages and makes the result of every check visible in a structured return value.

It is not a safety guarantee. It is a set of layered controls that raises the cost of common attacks, eliminates some categories of attack entirely, and makes failures auditable. The distinction matters — see the limitations section.

---

## Architecture

### Composition

`build()` in `__init__.py` is the single assembly point. It calls provider functions, each of which constructs one component from its own config slice. The resulting `Carapex` instance holds only abstractions — it does not know which concrete classes were used. Adding or swapping a component means changing `build()` and the relevant provider, not the pipeline logic.

Backend selection requires an explicit `"type"` field in the backend config dict. The value must match the backend's registered `name` attribute (e.g. `"openai_compatible"`, `"llama_cpp"`). A missing or unrecognised `"type"` raises `ConfigurationError` at startup. Shape-based inference (selecting a backend by the presence of config fields like `model_path`) was removed — it would silently select the wrong backend as the set of backends grows.

### Pipeline

Every `run()` call executes this sequence. The order is fixed.

```
prompt
  → Normaliser
  → PatternChecker
  → EntropyChecker
  → ScriptChecker
  → TranslationLayer
  → GuardChecker
  → [LLM call — receives original prompt]
  → OutputPatternChecker
  → OutputGuardChecker
  → Response
```

Cheap, deterministic checks run first. LLM-dependent checks run last. If an early check fails, later checks do not run — including the LLM call.

---

## Component Design

### Normaliser

Runs a set of decoders in order, repeatedly, until the output stabilises or `max_passes` is exhausted.

Built-in decoders: whitespace collapse, unicode escape sequences, HTML entities, URL percent-encoding, base64, homoglyph substitution. Each decoder is idempotent and strictly reductive — it moves text toward plain canonical form and cannot re-encode its output. This means the decoder set cannot produce oscillation cycles.

The normaliser returns a `NormaliserResult` with a `stable` flag. If the output did not stabilise — either because `max_passes` was exhausted or because a decoder cycle was detected — `stable=False` and the processor sets `failure_mode="normalisation_unstable"` and rejects the request.

**Why the LLM receives the original prompt.** Normalisation is for evaluation. A French user's prompt decoded from base64 is evaluated in its decoded form. The LLM receives what the user actually sent. Rewriting the user's input would change the task.

**Decoder extension constraint.** Custom decoders must be strictly reductive. A decoder that re-encodes its output can create a cycle with an earlier decoder. The normaliser detects and terminates cycles, but the root cause is the decoder — not the pass limit. If you observe `normalisation_unstable` on legitimate input, check your custom decoders first.

### PatternChecker

Regex-based. Deterministic. Catches structural tokens that have no legitimate use in a user prompt: `[INST]`, `[SYSTEM]`, `<s>`, `<s>`, `### Instruction`, `### System`, and explicit instruction-override phrases.

These patterns were chosen because they are zero-ambiguity — there is no legitimate user prompt that contains `[INST]` as meaningful natural language. Patterns with any plausible legitimate use belong in the guard, not here.

Invalid patterns raise `ConfigurationError` at startup. A pattern that fails to compile silently weakens this layer — startup failure is the correct behaviour.

Custom patterns replace the built-in set entirely. Passing `injection_patterns: []` in config is treated the same as `null` — both restore the defaults. This normalisation happens in `config.from_dict()`, not in `SafetyConfig` itself. The intent: an empty pattern set is almost certainly a misconfiguration, and the security-first default should be restored silently at the config boundary rather than failing at checker construction.

### EntropyChecker

Shannon entropy measures information density per character. Natural language has characteristic entropy (~3.5–4.5 bits/char). Content that survived normalisation without decoding — custom ciphers, undecoded binary — has much higher entropy (~6.5–8.0 bits/char).

The check runs on normalised text. If the normaliser successfully decoded content, entropy drops to natural-language range. Persistently high entropy after normalisation indicates content the decoder set could not process.

Default threshold: 5.8 bits/char. This clears dense technical content (code, JSON, URLs sit around 4.5–5.5) while catching most encoding attacks. Disable with `entropy_threshold: null` if your application regularly handles genuinely high-entropy input (e.g. cryptographic material).

Minimum length: 50 characters. Entropy on short strings is statistically unreliable.

### ScriptChecker and TranslationLayer

These two components work together.

`ScriptChecker` uses `langdetect` to identify the input language. It never blocks — it is a detector, not a gate. If detection fails, it flags the input for translation rather than passing it through undetected. It returns a `ScriptResult` — a `SafetyResult` subclass — carrying `detected_language` and `translation_needed`.

`TranslationLayer` translates non-English input to English before the guard sees it. The guard's system prompt, its evaluation criteria, and the safety training of the models typically used as guards are heavily English-weighted. Passing non-English text directly to the guard risks misclassification — particularly on low-resource languages, which are exactly the languages an attacker might choose.

**How the handoff works.** `TranslationLayer` is a `TextTransformingChecker` — an abstract subclass of `SafetyChecker` that signals to the composite that it may transform the working text. Before calling `check()`, the composite calls `set_prior_result()` on any `TextTransformingChecker`, passing the preceding checker's `SafetyResult`. `TranslationLayer.set_prior_result()` receives the `ScriptResult` from `ScriptChecker` and reads `translation_needed` directly — no redundant language detection. After a `safe=True` result, the composite calls `get_output_text()` and switches the working text to the English translation for all subsequent checkers.

The composite knows `TextTransformingChecker` (the abstract type). It does not know `TranslationLayer` (the concrete class). DIP is preserved — a second text-transforming checker can be added without touching the composite.

Translation uses the guard backend at temperature 0.0. Transcription should be deterministic: the same input should produce the same English output on every call.

Translation fails closed. An untranslated non-English prompt does not reach the guard. If translation fails, `failure_mode="translation_failed"` and the request is rejected.

The LLM receives the original prompt, not the translated version.

**`langdetect` seeding.** `langdetect` uses randomised internals by default, which produces non-deterministic language detection. carapex seeds it with `DetectorFactory.seed = 0` for consistent results.

### GuardChecker

An LLM call that evaluates semantic intent. Temperature 0.1 — near-deterministic, with slight variation to avoid degenerate outputs on ambiguous edge cases.

The guard evaluates:
- Framing manipulation: role-play, persona assignment, hypotheticals used to extract harmful content
- Instruction injection: instructions embedded in what appears to be data or context
- Coercion and urgency framing
- Attempts to redefine the model's identity, rules, or constraints

Per-call random delimiters (64-char hex tokens, `secrets.token_hex(32)`) wrap the evaluated content. The delimiter values are not logged. This prevents an attacker from constructing input that closes the delimiter boundary early.

The guard system prompt is fixed in code. It is not configurable. This is a deliberate constraint — a configurable guard system prompt is a system prompt that can be misconfigured.

**Guard temperature.** 0.1, not 0.0. Classification tasks benefit from slight variation because a fully greedy model can get stuck on degenerate outputs for edge cases where the correct classification is genuinely ambiguous. 0.1 provides enough variation to escape these without materially affecting determinism.

**Fail closed.** If the guard backend returns nothing, `failure_mode="guard_unavailable"` and the request is rejected. If the guard returns unparseable output, `failure_mode="guard_evaluation_corrupt"` and the request is rejected. If the guard returns valid JSON that does not contain a `"safe"` key, this is also treated as `"guard_evaluation_corrupt"` — a missing key is not a pass. Only the boolean literal `true` is accepted as a safe result. Any other value for `"safe"` (absent, `null`, an integer) fails closed.

The distinction between these two failure modes matters for alerting. Unavailability is an infrastructure event. Corrupt output may indicate prompt injection caused the guard to abandon its output format — it is a potential security signal. Both fail the same way (request rejected), but they should trigger different responses from your monitoring.

**Why fail closed rather than fail open.** Failing open on guard unavailability means requests pass with only pattern and entropy checking while appearing to the caller to have passed full evaluation. `Response.safe` would be `True`. The caller has no signal that the guard did not run. This is a false guarantee — the system claims to have checked something it did not check. carapex does not provide false guarantees.

### OutputPatternChecker and OutputGuardChecker

The output pipeline applies the same principle as the input pipeline to the LLM's response.

`OutputPatternChecker` catches known jailbreak success indicators ("as an unrestricted AI"), system prompt leakage, and prompt injection in the response targeting downstream systems.

`OutputGuardChecker` evaluates the response semantically — checking whether it contains harmful content, whether it appears to be a response to an injection rather than the legitimate task, and whether it reveals system configuration. It uses a separate system prompt from the input guard. The input guard evaluates user intent. The output guard evaluates AI-generated content.

Output guard is on by default. It adds one LLM call per `run()` on the guard backend. Disable with `output_guard_enabled: false`. Disabling removes semantic evaluation of responses — the output pattern checker still runs.

---

## Failure Semantics

### The core rule

Every failure is visible in `Response.failure_mode`. The caller always knows what happened. There are no silent failures.

`safe=False` alone is ambiguous — it covers both content refusals and infrastructure failures. Always branch on `failure_mode`. A guard that is down produces the same `safe=False` as a blocked prompt — the difference is in `failure_mode`.

### Failure mode reference

| `failure_mode` | Cause | Category |
|---|---|---|
| `None` | All checks passed | — |
| `"safety_violation"` | Content refused by pattern or guard | Content |
| `"entropy_exceeded"` | Input entropy above threshold after normalisation | Content |
| `"guard_unavailable"` | Guard backend returned no response | Infrastructure |
| `"guard_evaluation_corrupt"` | Guard responded but output was unparseable | Security signal |
| `"translation_failed"` | Language detection or translation call failed | Infrastructure |
| `"normalisation_unstable"` | Input did not converge after max passes | Security signal / decoder defect |
| `"backend_unavailable"` | Main LLM returned no response | Infrastructure |

### Guard backend at startup

`build()` health-checks the guard backend before returning a `Carapex` instance. If the guard is not reachable at startup, `build()` raises `BackendUnavailableError`. Starting the application with a dead guard means every `run()` call will be rejected under fail-closed semantics. This is caught at startup rather than at first request.

The main backend is not health-checked at startup. If it is down, `run()` returns `failure_mode="backend_unavailable"` on every call — a clear operational signal. This asymmetry is intentional: a dead guard is a security failure (no evaluation can complete), a dead main LLM is an infrastructure outage (the pipeline is honest about it). Treat persistent `backend_unavailable` responses as a monitoring alert, not a security event.

---

## The Stateless Guard

The guard has no session state. It evaluates each prompt from scratch without knowledge of previous calls.

This has one concrete security consequence: an attacker cannot groom the guard. Grooming attacks build up a shared context with a model over multiple turns — establishing premises, shifting framing, normalising constraint violations gradually. This works on stateful models. It does not work on a guard that starts fresh on every call.

This does not prevent grooming the main LLM. The main LLM retains session context across turns (by design — that is what makes it useful). A patient attacker can shape the main LLM's context through legitimate-looking turns and then send a payload that exploits that accumulated state. The guard sees the payload as a cold-start evaluation and may catch it — but this is not certain, and it is one of carapex's documented limitations.

---

## Multilingual Input

The check pipeline is language-independent at every stage except the guard. The normaliser, pattern checker, entropy checker, and script checker all operate without language assumptions.

The translation layer exists because guard evaluation is not language-independent. Rather than modifying the guard system prompt to handle all languages (which creates an evaluation quality problem for low-resource languages), carapex translates to English first. The guard always evaluates English. This is a simpler and more reliable design.

---

## Audit Log

JSONL. Thread-safe. One record per event. Records are linked by `audit_id` — a short identifier generated per `run()` call.

Events and what they carry:

| Event | Key fields |
|---|---|
| `carapex_init` | backend, input_checker, output_checker, normaliser, debug |
| `input_normalised` | stable flag, original_length, normalised_length |
| `input_safety_check` | safe, reason, failure_mode |
| `llm_call` | success flag |
| `output_safety_check` | safe, reason, failure_mode, token counts |
| `safety_refused` | stage, reason, failure_mode, token counts |
| `run_complete` | failure_mode, token counts |

Token counts appear on every event following the LLM call. They are captured before output safety evaluation, so they appear on output refusals as well as completions.

---

## What carapex Does Not Solve, and Directions to Address It

### Streaming is not currently supported

carapex requires the complete LLM response before output checks run. `OutputPatternChecker` applies regex to the full response text. `OutputGuardChecker` sends the full response to the guard backend for semantic evaluation. A pattern violation can appear anywhere in a response, including the final sentence. The guard cannot evaluate intent from an incomplete generation.

This is the most significant architectural constraint on how carapex can be deployed. It makes carapex unsuitable for interactive chat interfaces where users expect tokens to arrive as they are generated. It is well-suited to agentic workflows, API backends, and any application where LLM output feeds into a downstream system action — cases where unverified content acting on infrastructure is the higher risk.

The practical workaround for interactive interfaces is to have carapex return a verified response, then stream it to the user from the verified buffer at the application layer. The user experience looks like streaming. The latency cost is the full generation time before the first token appears.

### Current Limitations

**Semantic manipulation of LLMs is probabilistic.** The guard reduces the risk of semantic attacks. It does not eliminate it. A well-crafted prompt that frames a harmful request inside innocent-looking narrative, hypotheticals, or indirect instruction may pass the guard. LLM-based classification is not deterministic and can be fooled by distribution-shifted inputs. This is a fundamental property of the current approach, not a fixable edge case.

**Static defences degrade.** The pattern checker and entropy thresholds are fixed at deployment. Attackers who know the system can probe its thresholds, fingerprint its behaviour, and iterate bypass attempts. A static system provides diminishing returns against an active adversary who is specifically targeting it. The window of effectiveness depends on how widely the system is deployed and how motivated the attacker is.

**Multi-turn grooming of the main LLM.** The stateless guard eliminates grooming of the guard. It does not prevent an attacker from gradually shaping the main LLM's context over many turns that individually appear legitimate. The guard evaluates each prompt in isolation. If the accumulated context has shifted the main LLM's effective system prompt, the guard may not detect this until the attacker sends a payload that is itself overtly adversarial.

**No cross-session visibility.** carapex operates per-call. It has no visibility into patterns across sessions or accounts. A distributed attack that probes the system from multiple accounts, fragments intent across sessions, or uses coordinated timing is invisible at the per-call level.

**No behavioural trajectory.** The current architecture scores each prompt independently. It does not track escalation patterns, topic drift, refusal frequency, or other signals that accumulate across a session and may indicate a strategic attack even when individual prompts appear benign.

**General-purpose guard models.** The guard uses a general-purpose LLM. These models are not trained specifically to recognise prompt injection patterns, jailbreak taxonomies, or obfuscation variants. A model trained specifically on curated attack corpora would produce stronger signal.

**Text-level evaluation only.** carapex operates on the text of prompts and responses. It does not inspect intermediate model states, token probability distributions, or internal reasoning traces. Some attack signals exist at the latent representation level and are not visible in surface text.

### Directions to Address Them

**Probabilistic streaming.** A streaming mode that maintains meaningful output safety coverage is possible but requires careful design. The structure would be: input checks run as now (blocking); if they pass, streaming begins and the LLM streams tokens to the caller while carapex buffers the stream; `OutputPatternChecker` runs incrementally on each chunk, terminating the stream immediately on a pattern match; `OutputGuardChecker` runs on the complete buffer after streaming ends, with retraction signalled to the caller on failure.

This accepts a weaker output guarantee than the current design. The pattern layer provides incremental coverage. The semantic layer can only retract — it cannot prevent tokens from having reached the caller. A caller who renders content as it arrives may have already displayed unverified content before retraction arrives. Whether that tradeoff is acceptable depends on the use case. For agentic flows where LLM output triggers system actions, the current blocking model is safer. For interactive chat where the caller can handle retraction by replacing displayed content with an error state, probabilistic streaming is a reasonable direction.

The API shape would be `stream=True` on `run()`, returning an iterator of verified chunks plus a final verification result. The caller contract must make the probabilistic nature explicit — the stream is not fully verified until the final result arrives.

**Specialised small model for the guard.** A small language model fine-tuned on a curated corpus of prompt injection attempts, jailbreak patterns, indirect instruction attacks, and obfuscation variants would provide stronger classification signal than a general-purpose model asked to reason about safety. The carapex backend abstraction supports this without code changes — the specialised model would be a drop-in replacement for the guard backend. The work is in dataset curation and training, not in carapex's architecture.

**Session-level trajectory tracking.** Adding a session object to carapex that accumulates signals across calls — entropy trends, language switches, topic drift, proximity to known patterns, refusal history — would enable detection of attacks that are designed to be invisible at the per-call level. This requires a design pass: what state is tracked, how it decays over time, what threshold triggers an escalation response, and where the session object lives in the architecture. This is not a configuration option on the current design — it is an architectural addition.

**Continuous adversarial testing.** A fuzzing framework that generates mutated attack variants, obfuscated encodings, multi-language attacks, and narrative-framed indirect exploits, and runs them against a carapex deployment, would provide ongoing signal about which attacks succeed. Without this, the defence has no feedback loop and degrades relative to attacker innovation. This is operational infrastructure, not a library feature.

**Risk-weighted response degradation.** Rather than binary allow/deny, a session-aware carapex could reduce the LLM's response capabilities as risk signals accumulate: removing tool access, limiting context depth, switching to constrained output templates. This requires the session tracking infrastructure above, plus a defined policy for how accumulated risk maps to capability reduction.

**Cross-account pattern detection.** Detecting distributed probing requires infrastructure above the library level — identity correlation, IP reputation, behavioural signature matching across accounts. carapex can surface per-call signals (via `failure_mode` and the audit log) that a platform-level system could aggregate. The library is not the right place for this, but it can be designed to feed into one.

---

## Extension Points

**Backend** — `backends/`, subclass `LLMBackend`. Autodiscovered. Implement `chat()`, `health_check()`, `last_usage()`, `close()`. Set `name` to a unique string — this is also the value required in the config `"type"` field.

**Audit backend** — `audit/`, subclass `AuditBackend`. Autodiscovered. Implement `log()`, `close()`, `from_config()`, `default_config()`.

**Decoder** — `normaliser/`, subclass `Decoder`. Add name to config. Must be strictly reductive — it must only move text toward canonical form.

**Safety checker** — `safety/`, subclass `SafetyChecker`. Wire into `providers.py`. Checker order is a security decision.

Implement `close()` if your checker holds its own resources (a loaded model, a connection pool, an open file). `CompositeSafetyChecker.close()` propagates to all child checkers automatically. Checkers that hold a reference to an *injected* backend do not call `close()` on it — they do not own the backend's lifecycle. Only checkers that construct their own resources should implement `close()`.

**Backend ownership.** Backends are owned by the composition root (`build()`), not by the checkers that reference them. `Carapex.close()` closes backends after checkers. When a separate guard backend is configured, `Carapex` holds it explicitly for lifecycle management — the guard backend is referenced by multiple checkers; if each called `close()` on it, it would be closed multiple times. Centralised ownership avoids this without reference counting.

Adding any of these requires no changes to existing components. The only file that changes is the relevant registry (for autodiscovered components) or `providers.py` (for safety checkers that need explicit positioning in the pipeline).

**Duplicate names raise at startup.** Every autodiscovered component must have a unique `name` attribute within its registry. If two plugins share a name, `RuntimeError` is raised during autodiscovery — before the application starts. This is intentional: silent overwrite means last-import-wins, which is nondeterministic and produces no signal at the point of failure. If you see this error, check for naming collisions in your custom plugins or between a custom plugin and a built-in.

---

## Directory Structure

```
carapex/
├── __init__.py              build(), public API
├── __main__.py              CLI
├── processor.py             Carapex, Response
├── providers.py             component assembly
├── config.py                CarapexConfig, load(), write_default()
├── exceptions.py            exception hierarchy
├── backends/
│   ├── base.py              LLMBackend ABC
│   ├── openai_compatible.py HTTP backend
│   └── llama_cpp.py         in-process backend
├── safety/
│   ├── base.py              SafetyChecker ABC, SafetyResult, SafetyConfig
│   ├── composite.py         CompositeSafetyChecker
│   ├── pattern.py           PatternSafetyChecker
│   ├── entropy.py           EntropyChecker
│   ├── script.py            ScriptChecker
│   ├── translation.py       TranslationLayer
│   ├── guard.py             GuardSafetyChecker
│   ├── output.py            OutputSafetyChecker (pattern)
│   └── output_guard.py      OutputGuardChecker (semantic)
├── normaliser/
│   ├── base.py              Decoder ABC, NormaliserConfig
│   ├── normaliser.py        Normaliser, NormaliserResult
│   ├── whitespace.py
│   ├── unicode_escape.py
│   ├── html_entity.py
│   ├── url.py
│   ├── base64.py
│   └── homoglyph.py
├── audit/
│   ├── base.py              AuditBackend ABC
│   ├── file_backend.py      JSONL file backend
│   └── null_backend.py      no-op backend for testing
├── tests/
└── scripts/
    └── run_carapex.py      integration test script
```

---

## Exception Hierarchy

```
CarapexError
├── ConfigurationError
│   Startup failure — invalid config, bad patterns, missing backend.
│   The application should not start if this is raised.
│
├── BackendUnavailableError
│   A backend is not reachable.
│
├── PipelineInternalError
│   A component failed due to a bug — carries the component name and
│   the original exception. Distinct from operational failures.
│
├── CarapexViolation          [caller-raised only]
│   Raise when failure_mode indicates a content refusal.
│   Not raised internally — run() always returns a Response.
│
├── IntegrityFailure           [caller-raised only]
│   Raise when failure_mode indicates a check component failed.
│   Semantically different from CarapexViolation — the prompt was
│   not evaluated, not refused.
│
├── NormalisationError         [caller-raised only]
│   Raise when failure_mode is "normalisation_unstable".
│
└── PluginNotFoundError
    A component name was not found in the registry.
```

Exception clarity matters in security code. A catch block that conflates `IntegrityFailure` with `CarapexViolation` may retry a request on the content when the actual problem is a downed guard backend. The exception types enforce the distinction at the language level.
