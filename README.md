# carapex

carapex is a Python library that sits between your application and any LLM. It checks what goes in and what comes out. It is not a guarantee of safety — no software can provide that. It is a structured set of checks that raises the cost of common attacks and makes failures visible.

---

## What It Does

Every call to `run()` passes a prompt through a sequence of checks before the LLM sees it. The LLM's response passes through a second sequence before it reaches your code. If any check fails, you get a `Response` that tells you exactly what failed and why. Nothing is silent.

**Input checks, in order:**

1. **Normaliser** — decodes base64, URL encoding, unicode escapes, HTML entities, and homoglyph substitutions recursively until the text stabilises. Exposes obfuscated content before any safety check runs.
2. **Pattern checker** — regex detection of structural tokens that have no legitimate place in a user prompt (`[INST]`, `<s>`, `### System`, and similar).
3. **Entropy checker** — measures Shannon entropy of the normalised text. High entropy after normalisation means content that wasn't decoded — possible custom cipher or binary injection.
4. **Script checker** — detects the input language.
5. **Translation layer** — translates non-English input to English before the guard sees it. The guard evaluates English reliably. It does not reliably evaluate all languages.
6. **Guard** — an LLM call that evaluates semantic intent. Checks for framing manipulation, instruction injection, coercion, and other patterns the rule-based layers cannot catch.

**The LLM receives the original prompt** — not the normalised or translated version. Normalisation and translation are for evaluation only.

**Output checks, in order:**

1. **Output pattern checker** — detects jailbreak success indicators, system prompt leakage, and injection targeting downstream systems.
2. **Output guard** — an LLM call that evaluates the response for harmful content, signs of compromise, or content a properly aligned model would not produce. On by default.

---

## When to Use carapex

carapex is built for use cases where the complete LLM response is verified before it reaches the caller. It does not support streaming.

**carapex fits well:**
- Agentic workflows where LLM output feeds into a system action — a tool call, a database write, a downstream API request. Unverified content acting on infrastructure is a concrete risk.
- API backends where the consumer is another system, not a human watching tokens arrive.
- Applications where a wrong or harmful response has real consequences and latency to first token is not the primary concern.

**carapex does not fit:**
- Interactive chat interfaces where users expect to see tokens as they are generated. carapex must hold the full response until output checks complete. The UX cost is real.
- Any application where streaming is a hard requirement. The output guard cannot evaluate partial content — this is not a configuration option, it is how the verification works.

For interactive chat, the practical pattern is: carapex returns a verified response, the application layer streams it to the user from the verified buffer. The user sees streaming. The content has been checked. The latency cost is the full generation time before the first token appears.

Streaming with safety verification is a direction for future work. See DESIGN.md.

---

## What carapex Cannot Do

### Streaming is not currently supported

Output checks require the complete LLM response. `OutputPatternChecker` runs regex against the full text. `OutputGuardChecker` evaluates the full response semantically. Neither can operate on a partial stream — a pattern violation may appear in the last sentence, the guard cannot evaluate intent from an incomplete generation.

This is the most significant deployment constraint. It determines where carapex fits, which is why it is covered above under "When to Use carapex."

A future design for probabilistic streaming exists — if the input pipeline passes, streaming begins, pattern checks run incrementally on each chunk, and semantic evaluation runs on the complete buffer after the stream ends with retraction signalled on failure. This accepts a weaker output guarantee in exchange for lower latency to first token. It is not built yet because the retraction model requires careful design: a caller who renders content as it arrives may have already displayed unverified tokens before retraction arrives.

- It cannot eliminate semantic manipulation of LLMs. The guard reduces the risk but it is probabilistic, not deterministic.
- It cannot detect attacks spread across multiple sessions or accounts.
- It cannot prevent a patient attacker from gradually shaping the main LLM's context over many legitimate-looking turns.
- It does not adapt automatically as attack patterns evolve. Static rules degrade over time.

These are real limits. See DESIGN.md for a full discussion.

---

## Install

```bash
pip install carapex
pip install langdetect        # required — language detection
pip install llama-cpp-python  # required only if using a local model backend
```

---

## Quick Start

```python
import carapex

cp = carapex.build("config.json")
r  = cp.run("What were the main causes of World War I?")

if r.failure_mode in ("guard_unavailable", "guard_evaluation_corrupt", "translation_failed"):
    # A component of the check pipeline failed — this is an infrastructure problem
    # The prompt was not fully evaluated — do not treat this as a content decision
    alert_oncall(r.failure_mode)
    return service_unavailable()

elif r.failure_mode == "normalisation_unstable":
    # The input did not decode to a stable form — possible adversarial construction
    log_security_event(r)
    return service_unavailable()

elif not r.safe:
    # The prompt or response failed a safety check
    return r.refusal

else:
    return r.output
```

**`r.safe` alone is not sufficient.** `safe=False` covers both content refusals and infrastructure failures. A guard that is down produces `safe=False` — the same as a blocked prompt. Always read `failure_mode` first.

---

## First Time Setup

```bash
python -m carapex --init          # writes a default config.json
python -m carapex --show-config   # prints all config keys with defaults
python -m carapex --check         # checks that backends are reachable
```

---

## Configuration

```json
{
    "system_prompt": "You are a helpful assistant.",
    "debug": false,
    "backend": {
        "type": "openai_compatible",
        "base_url": "http://localhost:11434/v1",
        "model_name": "llama3",
        "api_key": "ollama"
    },
    "guard_backend": null,
    "normaliser": {
        "max_passes": 5,
        "decoders": ["whitespace", "unicode_escape", "html_entity", "url", "base64", "homoglyph"]
    },
    "safety": {
        "injection_patterns": null,
        "entropy_threshold": 5.8,
        "entropy_min_length": 50,
        "output_guard_enabled": true
    },
    "audit": {
        "backend": "file",
        "log_path": "carapex.audit.log"
    }
}
```

### Key decisions

**`backend.type`** — required. Selects the backend implementation. Use `"openai_compatible"` for any HTTP endpoint (Ollama, LM Studio, OpenAI API, llama-server). Use `"llama_cpp"` for in-process model loading. Missing or unrecognised `type` raises `ConfigurationError` at startup.

**`guard_backend`** — in production, use a separate small model for the guard. When `null`, the guard shares the main LLM backend. A busy main LLM under load can cause guard timeouts, which fail closed and reject requests. Requires the same `type` field as `backend`.

**`safety.injection_patterns`** — `null` or `[]` uses the built-in pattern set. A non-empty list replaces the defaults entirely. Invalid patterns raise `ConfigurationError` at startup — they are never silently skipped.

**`safety.output_guard_enabled`** — `true` by default. Setting it to `false` removes semantic evaluation of LLM responses. This reduces protection. If you disable it, document why.

---

## Response Fields

```python
r.output        # str | None  — LLM response. None if any layer failed or refused.
r.safe          # bool        — False if refused OR if any check layer failed.
r.refusal       # str | None  — plain-language explanation when safe=False.
r.failure_mode  # str | None  — machine-readable failure category. Read this first.
r.audit_id      # str         — links this response to audit log entries.
r.tokens_in     # int | None
r.tokens_out    # int | None
r.tokens_total  # int | None
```

### `failure_mode` values

| Value | What happened | How to handle it |
|---|---|---|
| `None` | All checks passed | Use `r.output` |
| `"safety_violation"` | Content refused by pattern or guard | Use `r.refusal` |
| `"entropy_exceeded"` | Input entropy too high after normalisation | Log, reject |
| `"guard_unavailable"` | Guard backend did not respond | Alert, treat as infrastructure failure |
| `"guard_evaluation_corrupt"` | Guard responded but output was unusable | Alert, treat as potential security signal |
| `"translation_failed"` | Language processing failed | Alert, treat as infrastructure failure |
| `"normalisation_unstable"` | Input did not stabilise after max passes | Log as security event |
| `"backend_unavailable"` | Main LLM did not respond | Infrastructure failure |

---

## Exception-Based Handling

`run()` always returns a `Response`. If you prefer exceptions, raise them yourself after reading `failure_mode`:

```python
from carapex.exceptions import CarapexViolation, IntegrityFailure, NormalisationError

r = cp.run(prompt)

if r.failure_mode in ("guard_unavailable", "guard_evaluation_corrupt", "translation_failed"):
    raise IntegrityFailure(r.failure_mode)

if r.failure_mode == "normalisation_unstable":
    raise NormalisationError(r.failure_mode)

if not r.safe:
    raise CarapexViolation(r.refusal, r.failure_mode)

return r.output
```

**`CarapexViolation`** — the prompt or response was evaluated and refused.
**`IntegrityFailure`** — a check component could not complete evaluation.
**`NormalisationError`** — the input did not decode to a stable form.

These are never raised by carapex itself. They exist so your code can use exception-based control flow if that matches your architecture.

---

## Audit Log

JSONL. One record per event. Every `run()` call produces multiple records linked by `audit_id`.

Events: `carapex_init`, `input_normalised`, `input_safety_check`, `llm_call`, `output_safety_check`, `safety_refused`, `run_complete`.

Token counts appear on every event that follows the LLM call, including refusals. The audit trail is complete regardless of where in the pipeline a request is stopped.

---

## Extending carapex

**New LLM backend** — add a file to `backends/`, subclass `LLMBackend`, set `name`, implement `chat()`, `health_check()`, `last_usage()`, `close()`. Autodiscovered. The `name` value is also what users set as `"type"` in their config — `default_config()` includes it automatically.

**New audit backend** — add a file to `audit/`, subclass `AuditBackend`, set `name`, implement `log()`, `close()`. Autodiscovered.

**New decoder** — add a file to `normaliser/`, subclass `Decoder`, set `name`, implement `decode()`. Add the name to `normaliser.decoders` in config. Decoders must only move text toward plain canonical form — a decoder that re-encodes its output will cause normalisation instability.

**New safety checker** — add a file to `safety/`, subclass `SafetyChecker`, wire it into `providers.py`. The order of checkers is a security decision — edit `providers.py` deliberately.

---

## Exceptions

```
CarapexError
├── ConfigurationError       — bad config at startup
├── BackendUnavailableError  — backend unreachable
├── PipelineInternalError    — component bug
├── CarapexViolation        — caller raises on content refusal
├── IntegrityFailure         — caller raises on check layer failure
├── NormalisationError       — caller raises on unstable normalisation
└── PluginNotFoundError      — unknown component name in registry
```

---

## Running Tests

```bash
pip install langdetect
python -m unittest discover -s carapex/tests -v
```

Integration test (requires a running LLM server):

```bash
python carapex/scripts/run_carapex.py --config config.json
python carapex/scripts/run_carapex.py --check
```

---

For architecture, failure semantics, threat model, and known limitations, see [DESIGN.md](DESIGN.md).
