# carapex

carapex is a Python library that sits between your application and any LLM. It checks what goes in and what comes out. It is not a guarantee of safety — no software can provide that. It is a structured set of checks that raises the cost of common attacks and makes failures visible.

It can be used as a library or as a transparent HTTP proxy. In proxy mode, you point your existing OpenAI client at carapex and change nothing else.

---

## What It Does

Every call to `evaluate()` passes the last user message through a sequence of checks before the LLM sees it. The LLM's response passes through a second sequence before it reaches your code. If any check fails, you get an `EvaluationResult` that tells you exactly what failed and why. Nothing is silent.

**Input checks, in order:**

1. **Normaliser** — decodes base64, URL encoding, unicode escapes, and HTML entities recursively until the text stabilises. Exposes obfuscated content before any safety check runs.
2. **Entropy checker** — measures Shannon entropy of the normalised text. High entropy after normalisation means content that wasn't decoded — possible custom cipher or binary injection.
3. **Pattern checker** — regex detection of structural tokens that have no legitimate place in a user prompt (`[INST]`, `<|im_start|>`, `### System`, and similar).
4. **Script checker** — detects the input language.
5. **Translation layer** — translates non-English input to English before the guard sees it. The guard evaluates English reliably. It does not reliably evaluate all languages.
6. **Input guard** — an LLM call that evaluates semantic intent. Checks for framing manipulation, instruction injection, coercion, and other patterns the rule-based layers cannot catch.

**The LLM receives the original messages list, unmodified** — not the normalised or translated version. Normalisation and translation are for evaluation only.

**Output checks, in order:**

1. **Output pattern checker** — detects jailbreak success indicators, system prompt leakage, and injection targeting downstream systems.
2. **Output guard** — an LLM call that evaluates the response for harmful content, signs of compromise, or content a properly aligned model would not produce. On by default.

---

## Two Entry Points, One Pipeline

**Library interface** — call `evaluate()` directly from Python. You own transport.

**HTTP proxy** — call `serve()` and point your OpenAI client at carapex's URL. You change one value — `base_url` — and nothing else. Same model name, same request shape, same response handling.

```python
client = OpenAI(
    base_url="http://carapex-host:8000/v1",
    api_key="sk-..."   # forwarded to the underlying LLM unchanged
)
```

The caller's API key is never stored or logged by carapex. It is forwarded on every LLM call and the LLM authenticates against it directly.

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

---

## What carapex Cannot Do

**Streaming is not currently supported.** Output checks require the complete LLM response. `OutputPatternChecker` runs regex against the full text. `OutputGuardChecker` evaluates the full response semantically. Neither can operate on a partial stream.

Beyond streaming:

- It cannot eliminate semantic manipulation of LLMs. The guard reduces the risk but it is probabilistic, not deterministic.
- It cannot detect attacks spread across multiple sessions or accounts.
- It cannot prevent a patient attacker from gradually shaping the main LLM's context over many legitimate-looking turns.
- It does not adapt automatically as attack patterns evolve. Static rules degrade over time.

---

## Install

```bash
pip install carapex
pip install "lingua-language-detector>=2.2"   # required — language detection
```

For HTTP proxy mode:

```bash
pip install carapex[server]   # adds fastapi + uvicorn
```

---

## Quick Start — Library

```python
from carapex import build
from carapex.core.config import CarapexConfig

config = CarapexConfig.load("carapex.yaml")
cx = build(config)

result = cx.evaluate([{"role": "user", "content": "What were the main causes of World War I?"}])

if result.failure_mode in ("guard_unavailable", "guard_evaluation_corrupt", "translation_failed"):
    # A component of the check pipeline failed — infrastructure problem
    # The prompt was not fully evaluated — do not treat this as a content decision
    alert_oncall(result.failure_mode)
    return service_unavailable()

elif result.failure_mode == "normalisation_unstable":
    # Input did not decode to a stable form — possible adversarial construction
    log_security_event(result)
    return service_unavailable()

elif not result.safe:
    # Prompt or response failed a safety check
    return result.reason or "Request could not be processed."

else:
    return result.content
```

**`result.safe` alone is not sufficient.** `safe=False` covers both content refusals and infrastructure failures. A guard that is down produces `safe=False` — the same as a blocked prompt. Always read `failure_mode` first.

## Quick Start — HTTP Proxy

```python
from carapex import build
from carapex.core.config import CarapexConfig

config = CarapexConfig.load("carapex.yaml")
cx = build(config)
cx.serve()   # blocks — ctrl+c or SIGTERM to stop
```

Then point any OpenAI client at `http://localhost:8000/v1`. No other changes.

---

## Configuration

```yaml
main_llm:
  type: openai
  url: https://api.openai.com
  model: gpt-4o

# Optional — null means fall back to main_llm for each role
input_guard_llm: null
output_guard_llm: null
translator_llm: null

safety:
  injection_patterns: null        # null uses the built-in default set
  entropy_threshold: 5.8
  entropy_min_length: 50
  output_guard_enabled: true
  input_guard_temperature: 0.1
  output_guard_temperature: 0.1
  translation_temperature: 0.0
  input_guard_system_prompt_path: null   # null uses the built-in default
  output_guard_system_prompt_path: null

normaliser:
  max_passes: 5
  decoders: null   # null uses all built-in decoders

audit:
  type: file
  path: carapex_audit.jsonl

server:                # omit entirely for library-only mode
  type: fastapi
  host: 0.0.0.0
  port: 8000
  workers: 1

debug: false
```

To generate a default config file:

```python
from carapex.core.config import CarapexConfig
CarapexConfig.write_default("carapex.yaml")
```

### Key decisions

**`main_llm.type`** — required. `"openai"` works with any OpenAI-compatible endpoint (OpenAI, Ollama, LM Studio, llama-server). Missing or unrecognised `type` raises `ConfigurationError` at startup.

**`input_guard_llm`** — in production, use a separate model for the guard. When `null`, the guard shares the main LLM. A busy main LLM under load can cause guard timeouts, which fail closed and reject requests. A separate guard model also provides stronger defence-in-depth — the guard cannot be primed by conversation history the caller passes.

**`safety.injection_patterns`** — `null` uses the built-in pattern set. A non-empty list replaces the defaults entirely. Invalid patterns raise `ConfigurationError` at startup. An empty list `[]` also raises — accidental clearance of the pattern set is always a misconfiguration.

**`safety.output_guard_enabled`** — `true` by default. Setting it to `false` removes semantic evaluation of LLM responses. If you disable it, document why. Setting this to `false` while also supplying `output_guard_llm` raises `ConfigurationError` — the contradiction is always a mistake.

**`safety.input_guard_temperature`** — must be in `(0.0, 1.0]`, exclusive of `0.0`. A fully greedy guard can produce degenerate outputs on ambiguous inputs. `0.0` raises `ConfigurationError`.

---

## EvaluationResult Fields

```python
result.safe          # bool        — False if refused OR if any check layer failed
result.content       # str | None  — LLM response. None if any layer failed or refused
result.failure_mode  # str | None  — machine-readable failure category. Read this first
result.reason        # str | None  — human-readable explanation when safe=False
```

### `failure_mode` values

| Value | What happened | How to handle it |
|---|---|---|
| `None` | All checks passed | Use `result.content` |
| `"safety_violation"` | Content refused by pattern or guard | Use `result.reason` |
| `"entropy_exceeded"` | Input entropy too high after normalisation | Log, reject |
| `"guard_unavailable"` | Guard LLM did not respond | Alert, treat as infrastructure failure |
| `"guard_evaluation_corrupt"` | Guard responded but output was unusable | Alert, treat as potential security signal |
| `"translation_failed"` | Language processing failed | Alert, treat as infrastructure failure |
| `"normalisation_unstable"` | Input did not stabilise after max passes | Log as security event |
| `"llm_unavailable"` | Main LLM did not respond | Infrastructure failure |
| `"no_user_message"` | Messages list contained no user-role entry | Caller error |

---

## Exception-Based Handling

`evaluate()` always returns an `EvaluationResult`. If you prefer exceptions, raise them yourself after reading `failure_mode`:

```python
from carapex import CarapexViolation, IntegrityFailure, NormalisationError

result = cx.evaluate(messages)

if result.failure_mode in ("guard_unavailable", "guard_evaluation_corrupt", "translation_failed"):
    raise IntegrityFailure(result.failure_mode)

if result.failure_mode == "normalisation_unstable":
    raise NormalisationError(result.failure_mode)

if not result.safe:
    raise CarapexViolation(result.reason)

return result.content
```

**`CarapexViolation`** — the prompt or response was evaluated and refused.  
**`IntegrityFailure`** — a check component could not complete evaluation. Do not retry as a content decision.  
**`NormalisationError`** — the input did not decode to a stable form.

These are never raised by carapex itself. They exist so your code can use exception-based control flow if that matches your architecture.

---

## Audit Log

JSONL. One record per event. Every `evaluate()` call produces multiple records linked by `audit_id`.

Events: `carapex_init`, `input_normalised`, `llm_call`, `input_safety_check`, `output_safety_check`, `evaluate_complete`.

`carapex_init` is emitted once at `build()` time. All other records are per-call. Use `(instance_id, audit_id)` as the composite key when aggregating logs from multiple carapex instances.

The prompt text, response content, and API keys are **never written to the audit log**. Only metadata — lengths, flags, failure modes, token counts — is recorded. The audit log is safe to ship to external observability systems.

---

## Extending carapex

**New LLM backend** — add a file to `carapex/llm/`, subclass `LLMProvider`, set `name`, implement `complete()` and `close()`. Decorate with `@register_llm`. Autodiscovered. Override `complete_with_temperature()` if the backend supports per-call temperature — recommended for any LLM used as a guard or translator.

**New audit backend** — add a file to `carapex/audit/`, subclass `Auditor`, set `name`, implement `log()` and `close()`. Decorate with `@register_auditor`. Autodiscovered. `log()` must never raise and must never block.

**New decoder** — add a file to `carapex/normaliser/`, subclass `Decoder`, set `name`, implement `decode()`. Decorate with `@register_decoder`. Autodiscovered. Decoders must only move text toward plain canonical form — a decoder that re-encodes its output will cause normalisation instability.

**New safety checker** — add a file to `carapex/safety/`, subclass `SafetyChecker`, set `name`, implement `inspect()` and `close()`. Then update the composition root in `carapex/carapex.py` to insert the checker at the correct position. Autodiscovery cannot determine where in the pipeline a checker belongs — that is a security decision made by a human.

**New server backend** — add a file to `carapex/server/`, subclass `ServerBackend`, set `name`, implement `serve()` and `close()`. Decorate with `@register_server`. Autodiscovered.

---

## Exceptions

```
ConfigurationError      — bad config at build() time
PipelineInternalError   — unexpected exception from a component (instance must be discarded)
CarapexViolation        — caller raises on content refusal (never raised internally)
IntegrityFailure        — caller raises on check layer failure (never raised internally)
NormalisationError      — caller raises on unstable normalisation (never raised internally)
```

Precondition violations (`ValueError`, `TypeError`) are raised by `evaluate()` directly on invalid input — null messages, empty list, no user message, wrong type. These indicate a caller bug.

---

## Running Tests

```bash
pip install -e ".[dev]"
python -m pytest tests/
```

Tests use `InMemoryAuditor` and stub LLM implementations — no network calls, no filesystem side effects.
