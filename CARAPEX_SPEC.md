# carapex — Specification

Version: 0.13
Status: VERIFIED. Full A+B+C verification pass complete. Gaps found this pass: 4 (A-1, B-1, B-2, C-S5-1), all resolved by standards or contradiction removal, no decisions required. A-1: CompletionResult.usage always present, zeroed if provider omits — grounded in OpenAI non-streaming usage guarantee. B-1: Auditor log() precondition contradiction removed — "raises ValueError" incompatible with "never raises" invariant. B-2: stream:true HTTP 400 body specified — grounded in OpenAI error object shape. C-S5-1: evaluate() precondition violation via HTTP path now specified — HTTP 400, invalid_request_error, grounded in OpenAI Chat Completions API error conventions. Non-blocking: B-V7-1 (stale version string in §20, corrected). Implementation Readiness: READY. Gap List: 0 blocking, 6 non-blocking (from v12, address in next revision). Verification Currency: CURRENT.

---

## Table of Contents

- [§1 Purpose and Scope](#1-purpose-and-scope)
- [§2 Concepts and Vocabulary](#2-concepts-and-vocabulary)
- [§3 System Boundary](#3-system-boundary)
- [§4 Data Contracts](#4-data-contracts)
  - EvaluationResult
  - Audit Log
- [§5 Component Contracts](#5-component-contracts)
- [§6 Lifecycle](#6-lifecycle)
- [§6a ServerBackend Lifecycle](#6a-serverbackend-lifecycle)
- [§7 Failure Taxonomy](#7-failure-taxonomy)
- [§8 Boundary Conditions](#8-boundary-conditions)
- [§9 Sentinel Values and Encoding Conventions](#9-sentinel-values-and-encoding-conventions)
- [§10 Atomicity and State on Failure](#10-atomicity-and-state-on-failure)
- [§11 Ordering and Sequencing](#11-ordering-and-sequencing)
- [§12 Interaction Contracts](#12-interaction-contracts)
- [§13 Concurrency and Re-entrancy](#13-concurrency-and-re-entrancy)
- [§14 External Dependencies](#14-external-dependencies)
- [§15 Configuration](#15-configuration)
- [§16 Extension Contracts](#16-extension-contracts)
- [§17 Error Propagation](#17-error-propagation)
- [§18 Observability Contract](#18-observability-contract)
- [§19 Security Properties](#19-security-properties)
- [§20 Versioning and Evolution](#20-versioning-and-evolution)
- [§21 What Is Not Specified](#21-what-is-not-specified)
- [§22 Assumptions](#22-assumptions)
- [§23 Performance Contracts](#23-performance-contracts)
- [§24 Future Directions](#24-future-directions)
- [Deployment Guidance](#deployment-guidance)

---

## §1 Purpose and Scope

carapex sits between a caller and an LLM. Every prompt passes through it before reaching the LLM. Every response passes through it before reaching the caller. It applies a fixed sequence of safety checks at both stages and returns a structured result that makes the outcome of every check visible and auditable.

It exists to raise the cost of common prompt injection and jailbreak attacks, eliminate some attack categories entirely, and make failures distinguishable from content refusals. Without carapex, a caller has no structured signal about why an LLM produced a given output or whether the input was evaluated at all.

**Design Principles**

The name is literal. A carapace is a shell — when it breaks, the organism is exposed. An implementation that silently degrades, fails open, or returns `safe=true` on an unevaluated prompt has broken the shell. The organism is exposed. This is not metaphor — it is the design constraint every implementation decision must be tested against.

**Compute before tokens.** Deterministic checks run before LLM calls. Once a violation is detected, the pipeline exits. Expensive checks never run on input that cheap checks already rejected.

**Fail closed.** Every infrastructure failure is a rejection. There is no condition under which an unevaluated prompt produces `safe=true`.

**False security is worse than no security.** A system that claims to check and does not is more dangerous than a system that makes no claim. Every `safe=true` result must mean all configured checks ran and passed.

**Inversion of control throughout.** The pipeline path is fixed. The modules that fill it are not. Components are swappable. The sequence is invariant.

**Drop-in replacement.** carapex is a transparent proxy. A caller using any OpenAI-compatible client — OpenAI SDK, llama.cpp, or any other — changes one value: `base_url`. No function calls change. No parameters change. No error handling changes. A caller's existing `api_key` passes through to the underlying LLM unchanged. Backends that do not use authentication (such as llama.cpp) accept and ignore the key — the caller's code is identical in both cases. Any design decision that requires a caller to change more than `base_url` violates this principle.

**No special casing.** carapex does not branch on LLM role, caller identity, or credential type. The orchestrator applies the same call convention to every LLM — main, guard, translator. If a guard or translator falls back to the main LLM instance, it uses the main LLM's call convention, including the caller's key. Separately-configured LLMs receive the caller's key and decide internally what to do with it. The orchestrator makes no assumptions about what any LLM does with its credentials. This keeps the pipeline uniform and prevents role-specific logic from becoming a maintenance surface.

**Out of scope.**
carapex does not guarantee that harmful content cannot reach an LLM. It does not provide session-level or cross-session attack detection. It does not support streaming — it requires the complete LLM response before output checks run. It does not perform access control, authentication, or rate limiting. It does not inspect intermediate model states, token probability distributions, or internal reasoning traces. It does not prevent grooming of the main LLM across multiple turns. These are documented limitations, not implementation gaps — see §19.

---

## §2 Concepts and Vocabulary

**prompt** — the content of the last user message in the messages list. Extracted by `evaluate()` from the `messages` parameter — the last entry where `role == "user"`. This is the text that passes through the safety pipeline. The original messages list (including history) is passed to the main LLM unchanged if evaluation passes.

**working text** — the text actively being evaluated at a given stage of the pipeline. Starts as the prompt. May be replaced by a translation during input evaluation. The LLM always receives the original prompt, not the working text.

**original prompt** — the prompt as submitted by the caller, unmodified. Preserved across the pipeline for delivery to the LLM regardless of any transformations applied during evaluation.

**normalisation** — the process of decoding obfuscated or encoded input into a canonical plain-text form for evaluation. Normalisation is not applied to the prompt delivered to the LLM.

**strictly reductive** — a property of a decoder: it only moves text toward canonical plain form and cannot re-encode its output. A strictly reductive decoder cannot produce output that another decoder would expand further than the original input.

**canonical form** — the plain-text representation that normalisation converges toward. No encoding, no obfuscation, no special characters with structural meaning.

**stable** — a property of normalisation output: the output did not change on the final pass. Stable output means no further decoding is possible with the configured decoder set.

**decoder cycle** — a condition detected during normalisation when a pass produces output identical to the output of an earlier pass in the same call. Formally: a cycle exists when `output[pass N] == output[pass N-k]` for any k > 0 where k < N. A cycle means the decoder set is oscillating — it cannot converge. Detection is immediate on the pass where the repeat is found. A decoder cycle sets `stable=false` and produces `failure_mode="normalisation_unstable"`. It is distinct from `max_passes` exhaustion, which occurs when the output is still changing on the final allowed pass — both produce `normalisation_unstable`, but a cycle indicates a decoder configuration defect while exhaustion indicates adversarially deep encoding.

**safe** — a property of a check result: the evaluated content did not trigger any configured check. `safe=true` means all checks passed. It does not mean the content is safe in an absolute sense.

**failure_mode** — a string identifier attached to an `EvaluationResult` when `safe=false`. Identifies the specific cause of failure. `null` when all checks passed. The caller MUST branch on `failure_mode` to distinguish content refusals from infrastructure failures.

**content refusal** — a failure caused by the content of the prompt or response triggering a safety check. Represented by `failure_mode="safety_violation"` or `failure_mode="entropy_exceeded"`.

**infrastructure failure** — a failure caused by a component being unavailable or returning unusable output. Represented by `failure_mode` values in the infrastructure and security-signal categories. The prompt was not fully evaluated — it was not refused.

**input guard** — the LLM used for semantic safety evaluation of input prompts. Distinct from the main LLM. Stateless — evaluates each prompt independently with no knowledge of prior calls. See also: output guard (§5 OutputGuardChecker).

**main LLM** — the LLM that produces the response the caller receives. Receives the original prompt if all input checks pass.

**grooming** — an attack that builds up a shared context with a model across multiple turns, gradually shifting its framing or constraints before sending a payload. Effective against stateful models; not effective against the input guard, which is stateless.

**delimiter** — a per-call random token that wraps evaluated content in a guard's prompt. Generated fresh for every guard call — both input and output. Prevents an attacker from constructing input that closes the delimiter boundary and escapes the evaluated region.

**audit_id** — a short identifier generated once per `evaluate()` call and attached to every audit log record produced by that call. Used to correlate all events from a single execution.

**Carapex** — the top-level class and orchestrator of the pipeline. Owns all components, executes `evaluate()`, and manages the component lifecycle from `build()` through `close()`.

**composition root** — the single location in the system that resolves registered components, constructs them, and wires them together.

**PatternChecker** — the input-side checker that evaluates the normalised working text against a compiled set of regex patterns. Detects structural injection tokens. Stateless. See §5.

**EntropyChecker** — the checker that measures Shannon entropy of the normalised working text and rejects input above the configured threshold. Stateless. See §5.

**InputGuardChecker** — the checker that submits the working text (post-translation, if applicable) to the input guard LLM for semantic safety evaluation. Stateful — holds a reference to the input guard LLM. See §5.

**OutputPatternChecker** — the output-side checker that evaluates the LLM response against a compiled set of regex patterns detecting jailbreak success indicators, system prompt leakage, and downstream injection attempts. Stateless. See §5.

**output guard** — the LLM used for semantic safety evaluation of LLM responses. Evaluates each response independently. Optional — disabled via `output_guard_enabled: false`. When enabled and no separate model is configured, falls back to the main LLM. See also: input guard (above), OutputGuardChecker (§5).

**OutputGuardChecker** — the pipeline component that evaluates the main LLM's response semantically using the output guard LLM. Stateful — holds a reference to the output guard LLM. Disabled via `output_guard_enabled: false` in config, in which case it is not instantiated. Implements the `SafetyChecker` contract. See §5.

**SafetyChecker** — the extension interface for all pipeline checkers, both input and output. Every checker in the pipeline implements this contract: `inspect(text) → SafetyResult`, `close()`, and a `name` attribute. Full contract in §16.

**CheckerCoordinator** — executes an ordered sequence of `SafetyChecker` instances, stopping at the first failure. `InputCoordinator` and `OutputCoordinator` are the two instances of `CheckerCoordinator` — one runs the input checker sequence, the other runs the output checker sequence. Full contract in §5.

**LLMProvider** — the extension interface for all LLM implementations. Every LLM in the pipeline — main, input guard, output guard, translator — is an `LLMProvider`. Implements `complete(messages: list[dict], api_key: str) -> CompletionResult | None` and `close()`. Full contract in §16.

**Auditor** — the extension interface for audit log destinations. Receives `log()` calls throughout `evaluate()`. Never raises — a log failure must not fail the request. Full contract in §16.

**Decoder** — the extension interface for normaliser decoding passes. Each decoder applies one strictly reductive transformation to the working text. Full contract in §16.

**TextTransformingChecker** — a `SafetyChecker` that may replace the working text as a side effect of a successful check. Implements two additional operations beyond the base `SafetyChecker` contract: `set_prior_result()`, which receives the preceding checker's result before `inspect()` is called, and `get_output_text()`, which returns the transformed text after a `safe=true` result. `Translator` is the only built-in `TextTransformingChecker`. The `CheckerCoordinator` identifies a checker as a `TextTransformingChecker` by the presence of `get_output_text()` on the checker instance — if the operation is present, the coordinator calls `set_prior_result()` before `inspect()` and calls `get_output_text()` after a `safe=true` result to update the working text.

**Translator** — the pipeline component that translates non-English working text to English before input guard evaluation. A `TextTransformingChecker` — replaces the working text on a successful translation. Stateful — holds a reference to the translator LLM. Reads `translation_needed` from the preceding `ScriptChecker` result via `set_prior_result()`. The original prompt delivered to the main LLM is never the translated text. See §5.

**ScriptResult** — the return type of `ScriptChecker.inspect()`. MUST be a `SafetyResult` subclass — implementations MUST use inheritance, not duck-typing. Always carries `safe=true` and additionally carries `detected_language` (string or null) and `translation_needed` (boolean). Passed to `Translator.set_prior_result()` before translation is attempted. Internal to the pipeline — not part of `EvaluationResult`.

**SafetyResult** — the return type of every `SafetyChecker.inspect()` call. Carries `safe` (boolean), `failure_mode` (string or null), and `reason` (string or null). `safe=true` means the checker found no issue. `safe=false` means the checker blocked the request — `failure_mode` identifies why. Internal to the pipeline — not part of `EvaluationResult`. The full field contract is in §5 per checker.

**NormaliserResult** — the return type of `Normaliser.normalise()`. Carries `text` (the decoded output string) and `stable` (boolean — whether the output converged on the final pass). Always returned, even when the input did not change. Internal to the pipeline. Full contract in §5.


**CompletionResult** — the return type of `LLMProvider.complete()`. Wraps the OpenAI Chat Completions response shape. Carries `content` (string — the LLM's response text, always non-empty), `finish_reason` (string — the reason the LLM stopped generating), and `usage` (dict with `prompt_tokens` (integer), `completion_tokens` (integer), and `total_tokens` (integer) — matching OpenAI field names). `complete()` returns `None` if the LLM is unavailable or returns no usable content — `CompletionResult` is only present on success. "No usable content" follows the OpenAI Chat Completions spec: `content` is `null` when `finish_reason` is `"content_filter"` or `"function_call"` — this is an intentional withholding, not a string value. `content: ""` (empty string) is not a specified OpenAI response state and is treated as malformed. In both cases `complete()` returns `None`. `CompletionResult.content` is therefore always a non-empty string — the invariant is enforced by the LLM implementation before constructing the result. `CompletionResult.usage` is always present — the OpenAI Chat Completions API always returns usage for non-streaming requests. If a provider returns a successful response but omits usage data, the LLM implementation MUST zero all three fields (`prompt_tokens: 0`, `completion_tokens: 0`, `total_tokens: 0`) rather than returning `None`. Absent usage data is not a failure condition. Internal to the pipeline.

**PipelineInternalError** — the error signalled by `evaluate()` out-of-band from `EvaluationResult` when a component raises an unexpected exception not covered by its contract. Indicates a bug in a component, not an operational failure. Carries the component name and the original exception. Never returned in `EvaluationResult`. Full propagation contract in §17.

**ConfigurationError** — raised by `build()` when the supplied configuration is invalid. Covers missing required fields, unrecognised type values, contradictory field combinations, and syntactically invalid values (e.g. bad regex patterns). Never raised at runtime — all configuration validation occurs at `build()` time. The error message MUST identify the field that caused the failure.

**CarapexViolation** — a caller-raised exception, never raised internally by `evaluate()`. Callers who prefer exception-based control flow MAY raise this from an `EvaluationResult` where `failure_mode` indicates a content refusal (`"safety_violation"` or `"entropy_exceeded"`). Signals that the prompt was evaluated and refused. See §17.

**IntegrityFailure** — a caller-raised exception, never raised internally by `evaluate()`. Callers MAY raise this from an `EvaluationResult` where `failure_mode` indicates a component failure (`"guard_unavailable"`, `"guard_evaluation_corrupt"`, `"translation_failed"`). Signals that the prompt was not fully evaluated — must not be treated as a content decision. See §17.

**NormalisationError** — a caller-raised exception, never raised internally by `evaluate()`. Callers MAY raise this from an `EvaluationResult` where `failure_mode` is `"normalisation_unstable"`. Signals that the input did not converge during decoding. See §17.

**RuntimeError** — Python built-in exception. Raised during autodiscovery at import time when two registered components share the same `name` attribute. Occurs before any `build()` call is possible. Indicates a packaging or registration error in a custom extension — two files in `backends/`, `safety/`, `audit/`, `normaliser/`, or `server/` declare the same `name`. The application cannot start. Distinct from `ConfigurationError` (which is raised at `build()` time) — `RuntimeError` is raised at import time. See §8 and §16.

**programming error** — a precondition violation raised when a caller passes invalid arguments to a component operation. Raised as `ValueError` when the argument is null, None, empty (null messages list, empty messages list, null content string), or semantically invalid (null text to a checker). Raised as `TypeError` when the argument is the wrong type. These are Python built-in exceptions. Programming errors indicate a bug in the caller — they are distinct from `ConfigurationError` (configuration-time error), `PipelineInternalError` (component bug at runtime), and operational failure modes (which are returned in `EvaluationResult`, never raised). carapex does not define or raise a custom exception class for precondition violations.

**Normaliser** — the pipeline component that decodes obfuscated or encoded input into canonical plain-text form before any checker runs. Stateless — holds an immutable decoder set and configuration. Exposes `normalise(text) → NormaliserResult`. Always runs first in the input pipeline. See §5.

**ServerBackend** — the extension interface for HTTP server implementations. Exposes `/v1/chat/completions` and delegates each request to the `Carapex` orchestrator via a `ChatHandler`. Pluggable via registry — `FastAPIBackend` is the default. Optional — present only when `server` config is supplied. Full contract in §16.

**ScriptChecker** — the pipeline component that detects the language of the normalised input and signals whether translation is required before input guard evaluation. Stateless between calls. Always returns `safe=true` — it is a detector, not a gate. Its output (`ScriptResult`) is consumed by `Translator`. See §5.

**ChatHandler** — the callable provided by `Carapex` to `ServerBackend.serve()`. The backend invokes it for each incoming Chat Completions request. The backend owns transport; `Carapex` owns request handling logic. The handler signature is `(request: ChatCompletionRequest, headers: dict) -> ChatCompletionResponse`. The `headers` dict contains the HTTP request headers with lowercase keys (e.g. `"authorization"`, `"content-type"`) — normalised by the `ServerBackend` before invocation. The handler extracts the caller's `authorization` header from `headers` and passes it as `api_key` to all LLM calls via `complete()` and `complete_with_temperature()`.

**messages** — the list of message dicts passed to `evaluate()`. Each entry carries `role` (string) and `content` (string). The caller assembles and manages the messages list — carapex does not inject messages, validate history, or maintain state between calls. The last entry where `role == "user"` is extracted as the prompt for safety evaluation. The full list is passed to the main LLM if evaluation passes.

**ChatCompletionRequest** — the request shape accepted by `/v1/chat/completions`. Matches the OpenAI Chat Completions API request format. The `messages` field is required. Other fields (`model`, `temperature`, etc.) are passed through to the main LLM. Unknown fields are not validated by carapex.

**ChatCompletionResponse** — the response shape returned by `/v1/chat/completions`. HTTP 200 in all cases. When `safe=true`: standard OpenAI Chat Completions response with the LLM's reply. When `safe=false`: synthetic assistant message with `finish_reason: "content_filter"`. See §3 for the full shape.

**CarapexConfig** — the structured configuration object passed to `build()`. Produced by `CarapexConfig.load(path)` from a JSON or YAML file, or constructed directly by callers. Carries all top-level configuration fields defined in §15. `CarapexConfig.write_default(path)` writes a complete configuration file with default values for all registered components.

**FastAPIBackend** — the default built-in `ServerBackend` implementation. Uses FastAPI and uvicorn. Registered under `type: fastapi`. Not production-hardened for high-concurrency deployments — operators who require that SHOULD supply a custom `ServerBackend` implementation.

**FileAuditor** — the default and required `Auditor` implementation for production deployments. Writes JSONL audit records to a configured file path. Required when the `audit` config block is present — there is no null or discard auditor. Full contract in §16.

**InMemoryAuditor** — a test-only `Auditor` implementation that stores records in memory for direct assertion. Not selectable via the `type:` config field — it is not a valid deployment auditor. Tests MUST use this rather than a file auditor to avoid filesystem side effects.

---

## §3 System Boundary

carapex exposes two entry points over the same pipeline.

**Library interface.** `evaluate(messages: list[dict]) -> EvaluationResult` — direct Python call. The caller passes a messages list, receives a structured result. Suitable for Python-native callers who manage their own transport.

**HTTP interface.** `POST /v1/chat/completions` — an OpenAI-compatible HTTP endpoint. The caller points their OpenAI client at carapex's URL and changes nothing else. The server backend is pluggable (see §5 ServerBackend, §16). The HTTP layer maps to `evaluate()` internally — there is one pipeline, two entry points.

**Drop-in replacement contract.**
A caller using any OpenAI-compatible client changes only `base_url` to point at carapex. No other code changes are required. The caller's existing `api_key` is forwarded to the underlying LLM unchanged. Backends that do not authenticate (such as llama.cpp) accept and ignore the key — the caller's code is identical in both cases. This is a hard contract, not a goal: any behaviour that requires a caller to change more than `base_url` is a spec violation.

**Input.**
`messages` is a non-empty list of dicts, each carrying `role` (string) and `content` (string). The list MUST contain at least one entry with `role == "user"` — the last such entry is the prompt evaluated by the safety pipeline. `null`/`None` is not valid — raises a programming error. An empty list raises a programming error. No maximum length is enforced on the list or on individual message content — the LLM determines what it accepts.

**Output.**
`evaluate()` returns an `EvaluationResult`. All outcomes including infrastructure failures are represented in `EvaluationResult`. The one exception: if a component raises an unexpected exception due to a bug, `evaluate()` signals `PipelineInternalError` out-of-band. This is not an operational failure — it indicates a defect in a component. See §17.

The HTTP layer maps `EvaluationResult` to a `ChatCompletionResponse`. When `safe=true`: normal assistant response. When `safe=false`: HTTP 200 with a synthetic assistant message and `finish_reason: "content_filter"`. HTTP 200 applies to all outcomes representable in `EvaluationResult` — the caller's code is unchanged in both cases. The one exception is `PipelineInternalError` (a component bug, not an operational failure), which the HTTP layer maps to HTTP 500 (see §17). Normal deployments should never encounter this path.

When `safe=false`, carapex returns HTTP 200 with a synthetic assistant message:

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "<reason from EvaluationResult, or 'I\'m unable to process this request.' if reason is null>"
    },
    "finish_reason": "content_filter"
  }],
  "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
}
```

`finish_reason` is always `"content_filter"` — this is an existing OpenAI enum value. It is not customisable. Clients that switch on `finish_reason` see expected behaviour.

`content` carries the human-readable explanation from `EvaluationResult.reason` when one is available — for example, the guard's description of what it detected. When `reason` is null (infrastructure failures), a generic fallback string is used. `content` is a free string — it is the appropriate place for explanatory text, not `finish_reason`.

`failure_mode` is recorded in the audit log. It is not exposed in the HTTP response.

When `evaluate()` raises a precondition violation (`ValueError` or `TypeError`) — for example, a `messages` list with no user-role entry, a null or empty list, or a null content field — `chat()` catches the exception and returns HTTP 400. The response body follows the OpenAI error object shape:

```json
{
  "error": {
    "message": "<description of the violation>",
    "type": "invalid_request_error",
    "code": "invalid_messages"
  }
}
```

`type: "invalid_request_error"` is an existing OpenAI enum value for malformed caller requests. This is distinct from `PipelineInternalError` (HTTP 500 — a component bug) and from the `stream: true` rejection (HTTP 400 — rejected before `chat()` is invoked). No audit record is produced — `evaluate()` raised before the pipeline ran and no `audit_id` was generated.

Caller code is unchanged — no new exception types, no new error handling required. The caller changes `base_url` only. Their response handling code runs unchanged regardless of whether carapex blocked the request or passed it through.

**Side effects.**
Every `evaluate()` call produces audit log records. Emission is best-effort — an audit log failure does not fail the request. One `audit_id` is generated per call and links all records from that call.

**Caller types.**
One caller type. The public API does not distinguish between caller roles or trust levels. Trust-level decisions are internal to the pipeline.

**Idempotency.**
`evaluate()` is not idempotent. Guard classification operates at a temperature above zero — results may vary across calls on genuinely ambiguous prompts. See §15 for temperature values and rationale. Translation produces consistent output for identical input. Pattern and entropy checks are deterministic. A caller must not assume that a prompt which passed once will always pass.

**Delivery semantics.**
Not applicable — carapex does not support streaming. The complete LLM response is required before output checks run.


**API key pass-through.**
The caller's `Authorization` header is forwarded to all LLM calls via the `api_key` parameter on `complete()` and `complete_with_temperature()`. carapex never stores, logs, or holds credentials. The orchestrator passes the caller's key to every LLM call — main LLM, guard LLMs, and translator LLM — regardless of whether they are the main LLM instance or a separately-configured instance. Separately-configured LLM implementations receive the caller's key and decide internally whether to use it or their own configured credentials. No special casing applies: the orchestrator's behaviour is identical for every LLM call. If a key is invalid or rejected by any LLM, the call returns `None` and `evaluate()` produces the appropriate failure mode (`llm_unavailable` or `guard_unavailable`).

**HTTP path.** `api_key` is the value of the `Authorization` header from the incoming request. If the header is absent or its value is empty, `api_key` is an empty string. An LLM that requires authentication will return `None` for an empty key — no special casing required.

**Library path.** `evaluate()` has no `api_key` parameter. Library callers configure credentials into their LLM instances at `build()` time — LLM implementations use their own configured credentials. The orchestrator passes an empty string as `api_key` for library calls. LLM implementations that use the supplied `api_key` (rather than their configured credentials) will return `None` on library calls unless they are configured to accept an empty key. LLM implementations that always use their own configured credentials are unaffected. This is consistent with the no-special-casing principle — the orchestrator's call convention is identical in both paths.

**State.**
carapex is stateless between `evaluate()` calls. The caller assembles the messages list and passes it on every call. If the caller includes history, the main LLM sees history. If not, it does not. carapex does not accumulate, inject, or validate message history.

**Construction.**
`build(config)` is the construction entry point. It validates configuration and constructs all components before returning a `Carapex` instance. It does not make network calls — LLM reachability is not verified at startup. No partially constructed instance is returned on configuration failure.

**Serving.**
`serve()` starts the HTTP server. It blocks until shutdown. `serve()` is only valid after `build()` returns and only when `server` config is present — calling `serve()` without a configured `ServerBackend` raises `ConfigurationError`.

**Teardown.**
`close()` terminates the instance and releases all held resources. It MUST be called after all in-flight `evaluate()` calls have completed and after `serve()` has returned. The instance is not reusable after `close()`. Full lifecycle contract in §6; close order and atomicity in §10 and §12.

**Configuration.**
`CarapexConfig.load(path)` deserialises a JSON or YAML configuration file and returns a `CarapexConfig` object. Raises `ConfigurationError` if the file is missing, unreadable, or syntactically invalid. `CarapexConfig.write_default(path)` writes a complete configuration file to `path` with default values for all registered components. Neither method is part of the `Carapex` instance lifecycle — both are class-level operations on `CarapexConfig` used before `build()` is called.

---

## §4 Data Contracts

### EvaluationResult

`EvaluationResult` is the structured return value of every `evaluate()` call. It is the internal result type — the library interface returns it directly. The HTTP layer maps it to a `ChatCompletionResponse` or error response before returning to the HTTP caller.

| Field | Type | Required | Meaning |
|---|---|---|---|
| `safe` | boolean | always | `true` if all checks passed and the LLM was called successfully. `false` in all other cases. |
| `content` | string or null | conditional | The LLM's response text. Present if and only if `safe=true`. Null in all other cases. |
| `failure_mode` | string or null | always | Null if `safe=true`. A failure mode identifier (see §7) if `safe=false`. |
| `reason` | string or null | conditional | A human-readable explanation of why the prompt or response was blocked. Present when `safe=false` and a human-readable explanation is available — typically content refusals where the guard or checker describes what it found. Null when `safe=false` due to infrastructure failures (`guard_unavailable`, `guard_evaluation_corrupt`, `translation_failed`, `llm_unavailable`, `normalisation_unstable`). Free text, not a stable identifier. Callers MUST NOT branch on this value — use `failure_mode` for programmatic handling. |

**Invariants.**
- `safe=true` implies `content` is a non-null string and `failure_mode` is null.
- `safe=false` implies `content` is null and `failure_mode` is a non-null string.
- These are exclusive and exhaustive — no other combinations are valid.

**Unknown fields.** A caller MUST ignore fields not listed above. Future versions MAY add fields.

### Audit Log

The audit log is a JSONL file. Each line is a self-contained JSON object representing one event. Records from a single `evaluate()` call are linked by a shared `audit_id` value. The HTTP layer maps to `evaluate()` — each HTTP request produces one set of audit records linked by one `audit_id`.

**Common fields present on every record.**

| Field | Type | Meaning |
|---|---|---|
| `event` | string | Event name. See event list below. |
| `audit_id` | string | Per-call identifier. Links all records from one `evaluate()` call. Absent on `carapex_init` records — those carry `instance_id` instead. |
| `timestamp` | string | ISO 8601 timestamp at time of emission. |
| `instance_id` | string (UUID4) | Identifies the carapex instance that emitted the record. Generated once at `build()` time and attached to every record including `carapex_init`. Use `(instance_id, audit_id)` as the composite key for multi-instance log aggregation. |

**`carapex_init` carries `instance_id` instead of `audit_id`.** It is emitted once at build time and is not associated with any `evaluate()` call. It carries `event` (always `"carapex_init"`) and `timestamp` in the same positions as all other records. `instance_id` is a UUID4 generated once at `build()` time. It anchors the configuration snapshot to the records produced by that instance — a consumer can determine exactly what components and endpoints were active when any set of `evaluate()` records were produced.

**Event-specific fields.**

The event set is open. Extensions — custom checkers, auditors, observability integrations — MAY emit event types not listed here. Consumers MUST ignore unknown event types rather than treating them as errors.

| Event | Additional fields |
|---|---|
| `carapex_init` | `version` (string), `main_llm` (URL), `input_guard_llm` (URL), `output_guard_llm` (URL), `translator_llm` (URL), `input_checker` (list of strings), `output_checker` (list of strings), `normaliser` (dict: `decoders` list of strings, `max_passes` integer), `debug` (boolean) — LLM URL fields carry the resolved endpoint, not the configured value. When `input_guard_llm`, `output_guard_llm`, or `translator_llm` is null in config and falls back to another LLM, the field carries the URL of whichever LLM is actually serving that role. A null config value never appears here — the record always shows the URL in use. `instance_id` is in common fields — not duplicated here. |
| `input_normalised` | `stable` (boolean), `original_length` (integer), `normalised_length` (integer) |
| `input_safety_check` | `safe` (boolean), `failure_mode` (string), `reason` (string) — `failure_mode` and `reason` absent if `safe=true`. The `reason` field MUST NOT disclose which specific pattern matched — implementations MUST use a generic description (e.g. "Input contains a disallowed pattern"). Disclosing the matched pattern allows an attacker to enumerate the pattern set and craft bypasses. |
| `llm_call` | `role` (string — one of `"translator"`, `"input_guard"`, `"main_llm"`, `"output_guard"`), `success` (boolean), `prompt_tokens` (integer), `completion_tokens` (integer), `total_tokens` (integer), `latency_ms` (integer), `finish_reason` (string) — fields other than `role` and `success` absent if call failed; `prompt_tokens`, `completion_tokens`, and `total_tokens` present on success (zeroed if the provider did not return usage data — see `CompletionResult` in §2) |
| `output_safety_check` | `safe` (boolean), `failure_mode` (string), `reason` (string) — `failure_mode` and `reason` absent if `safe=true`. The `reason` field MUST NOT disclose which specific pattern matched — same constraint as `input_safety_check`. |
| `evaluate_complete` | `safe` (boolean), `failure_mode` (string) — `failure_mode` absent if `safe=true` |

**Encoding.** UTF-8. One JSON object per line. No trailing comma. Records are not guaranteed to be ordered by timestamp within a file under concurrent access — use `audit_id` and event sequence for reconstruction, not line order.

**Emission guarantee.** Best-effort. An audit log failure does not fail the request. A record that was not written is not retried.

**Unknown fields.** A consumer MUST ignore fields not listed above. Future versions MAY add fields to any event.

---

### ChatCompletionRequest and ChatCompletionResponse

carapex's HTTP interface conforms to the OpenAI Chat Completions API as documented at `https://platform.openai.com/docs/api-reference/chat` (snapshotted against the API version current at the time of this spec: the `v1` endpoint, `POST /v1/chat/completions`, as of the 2024-08-06 model snapshot era). carapex does not validate unknown fields in the request — they are passed through to the main LLM unchanged.

**ChatCompletionRequest — fields consumed by carapex**

| Field | Type | Required | Meaning |
|---|---|---|---|
| `messages` | array of `{role: string, content: string}` | required | The conversation messages. carapex extracts the last user message for safety evaluation and forwards the full list to the main LLM if evaluation passes. |
| `model` | string | required by OpenAI spec | Passed through to the main LLM unchanged. Not used by carapex internally. |
| `temperature` | number or null | optional | Passed through to the main LLM unchanged. Not used by carapex internally — guard and translation temperatures are set from carapex config, not from the caller's request. |

All other fields in the OpenAI Chat Completions request (`max_tokens`, `stream`, `tools`, etc.) are passed through to the main LLM unchanged. carapex does not validate or interpret them. `stream` MUST NOT be set to `true` — carapex does not support streaming (§1, §3). If `stream: true` is present in the request, the `ServerBackend` MUST reject the request with HTTP 400 before invoking `ChatHandler`. The response body MUST follow the OpenAI error object shape: `{"error": {"message": "carapex does not support streaming", "type": "invalid_request_error", "code": "streaming_not_supported"}}`. The rejection MUST NOT produce an audit record — the request never entered the pipeline. This applies to the HTTP interface only; the library interface has no equivalent path.

**ChatCompletionResponse — carapex-specific shape**

When `safe=true`, carapex returns the main LLM's response verbatim. The response is the standard OpenAI Chat Completions response object: `id`, `object`, `created`, `model`, `choices` (array with `message.role`, `message.content`, `finish_reason`, `index`), and `usage` (`prompt_tokens`, `completion_tokens`, `total_tokens`).

When `safe=false`, carapex returns a synthetic response. This is a carapex-specific deviation from the OpenAI spec — real OpenAI responses do not use this shape to signal safety blocks:

```json
{
  "id": "<generated>",
  "object": "chat.completion",
  "created": <unix_timestamp>,
  "model": "<model from request>",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "<EvaluationResult.reason, or 'I\'m unable to process this request.' if reason is null>"
    },
    "finish_reason": "content_filter"
  }],
  "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
}
```

`finish_reason: "content_filter"` is an existing OpenAI enum value. It is not customisable. The HTTP status is always 200. `failure_mode` is not exposed in the HTTP response — it is recorded in the audit log only.


---

## §5 Component Contracts

### Component hierarchy

```
Carapex
  ├── Normaliser
  ├── InputCoordinator
  │     ├── EntropyChecker
  │     ├── PatternChecker
  │     ├── ScriptChecker
  │     ├── Translator
  │     └── InputGuardChecker
  ├── Main LLM
  ├── OutputCoordinator
  │     ├── OutputPatternChecker
  │     └── OutputGuardChecker
  ├── Auditor
  └── ServerBackend  (optional — present only when server config supplied)
```

The `Carapex` class owns all components. LLMs are shared resources owned exclusively by the composition root — no checker may close an LLM it did not construct.

### Carapex

```
Component: Carapex
Purpose:   Orchestrates the full input and output safety pipeline. Exposes the library interface
           via evaluate() and the HTTP interface via chat() and serve().

Stateful or stateless: Stateful — holds references to all pipeline components and manages their lifecycle.

Invariants:
  - All pipeline components are fully initialised before any call to evaluate() is accepted.
  - The original messages list is never modified before delivery to the main LLM.
  - The prompt evaluated by the safety pipeline is always the content of the last user-role message.
  - Guards receive the last user message from the messages list.

Preconditions for evaluate():
  - messages is a non-null, non-empty list of dicts, each with 'role' and 'content' string fields.
  - At least one entry in messages has role == "user" — guaranteed by the OpenAI Chat Completions API contract (see §22).
  - Violation of list-null or list-empty: raises a programming error. Does not return an EvaluationResult.

Accepts:
  messages: list[dict] — a non-null, non-empty list. Each dict carries 'role' and 'content'.
    The last entry with role == "user" is extracted as the prompt for the safety pipeline.
    The full list is passed to the main LLM if evaluation passes.

Returns:
  EvaluationResult with safe=true:  all input checks passed, the LLM was called, all output checks passed.
  EvaluationResult with safe=false: any check failed or any component was unavailable. failure_mode identifies the cause.

Postconditions:
  - After evaluate(): an EvaluationResult is returned. evaluate() never raises except as documented in §17.
  - After evaluate(): at least one audit record with the call's audit_id has been emitted (best-effort).

Additional operations:
  chat(request: ChatCompletionRequest, headers: dict) -> ChatCompletionResponse
    HTTP handler. Extracts the caller's Authorization header from headers and forwards it to all LLM calls
    via the api_key parameter on complete(). Maps the incoming request to evaluate(). Maps EvaluationResult
    to ChatCompletionResponse — HTTP 200 in all cases representable in EvaluationResult. When safe=false:
    synthetic assistant message with finish_reason="content_filter". PipelineInternalError maps to HTTP 500.
    Called by ServerBackend for each incoming HTTP request.

  serve() -> None
    Starts the ServerBackend. Blocks until shutdown. Valid only when server config is present.
    Raises ConfigurationError if called without a configured ServerBackend.

Guarantees:
  - Never raises on content or infrastructure failures — all such outcomes are in EvaluationResult.
  - Always delivers the original messages list to the main LLM, unmodified, if input checks pass.
  - The prompt seen by the safety pipeline is always the last user message — never a system message,
    never an assistant message, never injected content.

Failure behaviour:
  All failure modes defined in §7: returned in EvaluationResult.failure_mode. evaluate() does not raise.
  llm_unavailable (defined in §7): main LLM returned no response. safe=false, failure_mode="llm_unavailable".
    The input pipeline ran to completion — the request was evaluated but no response was produced.
```

---

### Normaliser

```
Component: Normaliser
Purpose:   Decodes obfuscated or encoded input into canonical plain-text form for evaluation.

Stateful or stateless: Stateless — holds an immutable decoder set and config. No mutable state between calls.

Invariants:
  - None — this component is stateless.

Preconditions for normalise():
  - Input is a non-null string.
  - Violation behaviour: raises a programming error.

Accepts:
  text: string — any non-null string, including empty string.

Returns:
  NormaliserResult on success: always returned, even if input did not change.

Postconditions:
  - After normalise(): result.text contains the decoded output.
  - After normalise(): result.stable=true if and only if the output did not change on the final pass.
  - After normalise(): result.stable=false if max_passes was exhausted with output still changing on the
    final pass, OR if a decoder cycle was detected (see §2 "decoder cycle"). Both conditions produce
    normalisation_unstable. They are operationally equivalent to the orchestrator but are distinguishable
    in monitoring by examining whether the pass that set stable=false was within max_passes or detected
    a repeat — this distinction is not surfaced in NormaliserResult itself, only in internal logging.

Guarantees:
  - Never raises — all outcomes including cycle detection are represented in NormaliserResult.
  - The decoded output is always equal to or shorter than the input, never longer, for any built-in decoder.
  - A decoder cycle is detected on the pass where the repeat is first found — the normaliser does not
    continue running after cycle detection.

Failure behaviour:
  normalisation_unstable (defined in §7): result.stable=false. The orchestrator rejects the request.
    The normalised text is still present in result.text — it represents the best available decoding.
    Caused by either max_passes exhaustion or decoder cycle — both are treated identically by the orchestrator.
```

---

### PatternChecker

```
Component: PatternChecker
Purpose:   Detects structural injection tokens in the working text using compiled regular expressions.

Stateful or stateless: Stateless — holds an immutable compiled pattern set. No mutable state between calls.

Invariants:
  - None — this component is stateless.
  - The pattern set is non-empty. build() raises ConfigurationError if an empty or null pattern set is supplied — the checker is never constructed with an empty set. See §8 and §15.

Preconditions for inspect():
  - Input text is a non-null string.
  - Violation behaviour: raises a programming error.

Accepts:
  text: string — any non-null string.

Returns:
  SafetyResult with safe=true:  no pattern matched.
  SafetyResult with safe=false: at least one pattern matched. failure_mode="safety_violation".

Guarantees:
  - Never raises on input content — all match outcomes are in SafetyResult.
  - Deterministic — identical input always produces identical output.

Failure behaviour:
  safety_violation (defined in §7): safe=false, failure_mode="safety_violation", reason names the matched pattern.
```

---

### EntropyChecker

```
Component: EntropyChecker
Purpose:   Rejects input whose Shannon entropy after normalisation exceeds the configured threshold.
           See §15 for the default threshold value and the rationale for its selection.

Stateful or stateless: Stateless — holds immutable threshold and minimum-length config. No mutable state between calls.

Invariants:
  - None — this component is stateless.

Preconditions for inspect():
  - Input text is a non-null string.
  - Violation behaviour: raises a programming error.

Accepts:
  text: string — any non-null string.

Returns:
  SafetyResult with safe=true:  entropy is at or below threshold, or input is below minimum length.
  SafetyResult with safe=false: entropy exceeds threshold. failure_mode="entropy_exceeded".

Postconditions:
  - If len(text) < minimum_length: safe=true unconditionally. Entropy is not computed.
  - If entropy_threshold is null: safe=true unconditionally. Entropy is not computed.

Guarantees:
  - Never raises on input content.
  - Deterministic — identical input always produces identical output.

Failure behaviour:
  entropy_exceeded (defined in §7): safe=false, failure_mode="entropy_exceeded".
```

---

### ScriptChecker

```
Component: ScriptChecker
Purpose:   Detects the language of the input and signals whether translation is required before input guard evaluation.

Stateful or stateless: Stateless between calls. Holds immutable configuration. Language detection
  library state (seeding) is initialised once at construction and does not change thereafter.

Invariants:
  - The language detection library is seeded with a fixed value at construction. The seed does not change after construction.

Preconditions for inspect():
  - Input text is a non-null string.
  - Violation behaviour: raises a programming error.

Accepts:
  text: string — any non-null string.

Returns:
  ScriptResult with safe=true always. ScriptChecker never blocks — it is a detector, not a gate.
  ScriptResult carries: detected_language (string or null), translation_needed (boolean).

Postconditions:
  - safe=true unconditionally.
  - translation_needed=true if detected_language is not English or if detection failed or if
    detection confidence is below the configured threshold.
  - translation_needed=false if and only if detected_language is English AND detection confidence
    is at or above the configured threshold (default: 80%, configurable via
    safety.script_confidence_threshold in §15).
  - detected_language=null if detection failed.

Guarantees:
  - Never raises — detection failure sets translation_needed=true rather than raising.
  - Never returns safe=false.

Rationale for 80% default: the cost asymmetry is intentional. Under-confidence produces an
  unnecessary translation call — cheap and safe. Over-confidence on a non-English prompt sends
  untranslated text to the input guard, which evaluates English only. 80% sits above the noise
  floor for short inputs while keeping the false-translation rate acceptable. Operators with
  known-English traffic may raise it; operators handling very short inputs should lower it.
```

---

### Translator

```
Component: Translator
Purpose:   Translates non-English working text to English before input guard evaluation.
           A TextTransformingChecker — may replace the working text for all subsequent pipeline stages.

Stateful or stateless: Stateful — holds a reference to the translator LLM for translation calls.

Invariants:
  - set_prior_result() MUST be called with the ScriptResult from ScriptChecker before inspect() is called.
    A Translator that has not received set_prior_result() is in an invalid state for inspect().

Preconditions for inspect():
  - set_prior_result() has been called with a ScriptResult carrying translation_needed.
  - Input text is a non-null string.
  - Violation behaviour: raises a programming error if set_prior_result() was not called first.

Accepts:
  text: string — any non-null string.

Returns:
  SafetyResult with safe=true:  translation_needed=false (no translation required), or translation succeeded.
  SafetyResult with safe=false: translation_needed=true and translation call failed. failure_mode="translation_failed".

Postconditions:
  - If safe=true and translation occurred: get_output_text() returns the English translation.
  - If safe=true and no translation was needed: get_output_text() returns the original text unchanged.
  - If safe=false: the request is rejected. The untranslated text does not proceed to the input guard.

Invariant on translation result:
  - When translation_needed=true and the translator LLM returns a response that is byte-for-byte
    identical to the input text, the Translator MUST treat this as a failed translation:
    safe=false, failure_mode="translation_failed". An echo response is not a valid translation.
    This is the minimum detectable violation of the §19 guarantee that "the guard always evaluates
    English." It does not guarantee translation quality — it closes only the trivially detectable
    bypass where the translator returns its input unchanged.

Guarantees:
  - Never passes untranslated non-English text to the input guard when identity echo is detectable.
  - The original prompt delivered to the main LLM is never the translated text.

Failure behaviour:
  translation_failed (defined in §7): safe=false, failure_mode="translation_failed". Request rejected.
    Triggered by: translation call returning None, translation call returning the input text unchanged
    when translation_needed=true, translation call returning an empty string when translation_needed=true.
    An empty translation is not a valid English rendering of the input — the guard would evaluate nothing
    rather than the translated content, undermining the §19 English-evaluation guarantee.
```

---

### InputGuardChecker

```
Component: InputGuardChecker
Purpose:   Evaluates the semantic intent of the working text using an LLM call. Detects framing
           manipulation, instruction injection, coercion, and identity redefinition attempts.

Stateful or stateless: Stateful — holds a reference to the input guard LLM.

Invariants:
  - The input guard system prompt is loaded at construction from the configured path, or from the built-in default if no path is supplied. It does not change after construction.
  - Per-call delimiters are generated fresh for every inspect() call using a cryptographically random source. See §19 for the entropy requirement.

Preconditions for inspect():
  - Input text is a non-null string.
  - Violation behaviour: raises a programming error.

Accepts:
  text: string — any non-null string. The working text at this stage (English, post-translation if applicable).

Returns:
  SafetyResult with safe=true:  input guard returned valid JSON with "safe": true (boolean literal only).
  SafetyResult with safe=false: any other outcome. failure_mode identifies the specific cause.

Postconditions:
  - safe=true if and only if the input guard returned a parseable JSON response containing "safe": true.
  - Any other value for "safe" — absent, null, false, an integer — produces safe=false.

Guarantees:
  - Fails closed. A missing, unparseable, or structurally invalid input guard response never produces safe=true.
  - Delimiter values are never written to the audit log.

Failure behaviour:
  guard_unavailable (defined in §7): guard returned no response. safe=false, request rejected. reason=null.
  guard_evaluation_corrupt (defined in §7): guard responded but output was unparseable or missing "safe" key.
    safe=false, request rejected. Distinct from guard_unavailable — treat as a potential security signal. reason=null.
  safety_violation (defined in §7): input guard returned safe=false. safe=false, failure_mode="safety_violation".
    reason is populated from the guard response's "reason" field if present; null if the field is absent.
    The guard response JSON MAY carry a "reason" field alongside "safe" — a string description of why the
    prompt was refused. carapex extracts it when present and surfaces it in SafetyResult.reason and
    ultimately EvaluationResult.reason. The guard is not required to supply it — reason=null is valid.
```

---

### OutputPatternChecker

```
Component: OutputPatternChecker
Purpose:   Detects known jailbreak success indicators, system prompt leakage, and downstream
           injection attempts in the LLM's response text.

Stateful or stateless: Stateless — holds an immutable compiled pattern set. No mutable state between calls.

Invariants:
  - None — this component is stateless.

Preconditions for inspect():
  - Input text is a non-null string.
  - Violation behaviour: raises a programming error.

Accepts:
  text: string — the LLM response text.

Returns:
  SafetyResult with safe=true:  no output pattern matched.
  SafetyResult with safe=false: at least one pattern matched. failure_mode="safety_violation".

Guarantees:
  - Never raises on input content.
  - Deterministic — identical input always produces identical output.

Failure behaviour:
  safety_violation (defined in §7): safe=false, failure_mode="safety_violation".
```

---

### OutputGuardChecker

```
Component: OutputGuardChecker
Purpose:   Evaluates the LLM response semantically — checking for harmful content, injection responses,
           and system configuration disclosure. Uses a separate system prompt from the input guard.

Stateful or stateless: Stateful — holds a reference to the output guard LLM.

Invariants:
  - The output guard system prompt is loaded at construction from the configured path, or from the built-in default if no path is supplied. It does not change after construction and is distinct from the input guard system prompt.
  - OutputGuardChecker may be disabled via configuration. When disabled, this component is not instantiated.
  - Per-call delimiters are generated fresh for every inspect() call using a cryptographically random source. See §19 for the entropy requirement. Delimiter values are never written to the audit log.

Preconditions for inspect():
  - Input text is a non-null string (the LLM response).
  - Violation behaviour: raises a programming error.

Accepts:
  text: string — the full LLM response text, including empty string. Empty string is not a special case — it passes through the full output pipeline unchanged. OutputGuardChecker evaluates whatever the LLM produced.

Returns:
  SafetyResult with safe=true:  output guard evaluated the response as safe.
  SafetyResult with safe=false: output guard evaluated the response as unsafe, or output guard call failed.

Guarantees:
  - Fails closed on guard unavailability or corrupt output, identically to InputGuardChecker.

Failure behaviour:
  guard_unavailable (defined in §7): safe=false, request rejected. reason=null.
  guard_evaluation_corrupt (defined in §7): safe=false, request rejected. reason=null.
  safety_violation (defined in §7): safe=false, failure_mode="safety_violation".
    reason is populated from the guard response's "reason" field if present; null if absent.
    Same contract as InputGuardChecker — the guard MAY supply a reason string; carapex extracts it when present.
```

---

### CheckerCoordinator

```
Component: CheckerCoordinator
Purpose:   Executes an ordered sequence of SafetyCheckers. Stops at the first failure.
           Manages TextTransformingChecker handoff — updates working text when a checker signals transformation.

Stateful or stateless: Stateful — holds an ordered list of checker instances and manages working text.

Invariants:
  - Checker order is fixed at construction and does not change.
  - A failure in any checker stops execution. Subsequent checkers do not run.

Preconditions for inspect():
  - Input text is a non-null string.
  - Violation behaviour: raises a programming error.

Accepts:
  text: string — the initial working text (normalised last user message for input; LLM response for output).

Returns:
  SafetyResult with safe=true:  all checkers passed.
  SafetyResult with safe=false: the first checker that failed. failure_mode from that checker.

Postconditions:
  - If a checker exposes get_output_text() and returns safe=true: the coordinator calls
    set_prior_result() before inspect() and calls get_output_text() after the safe=true result.
    The returned text replaces the working text for all subsequent checkers in the sequence.
  - The coordinator does not know which concrete checker types it holds — it identifies
    TextTransformingChecker behaviour by the presence of get_output_text() on the instance.

Guarantees:
  - Never runs a subsequent checker after a failure.
  - close() propagates to all child checkers.
  - An empty checker sequence is valid. inspect() on an empty sequence returns
    SafetyResult(safe=true) immediately without emitting a check audit record.
    This is the correct behaviour when both output checkers are disabled via config —
    the output coordinator runs with an empty sequence and passes unconditionally.
    The security tradeoff of an empty output pipeline is documented in §15.

Notes:
  - The input_safety_check audit record covers the full InputCoordinator run — it is not per-checker.
    failure_mode identifies which checker produced the block:
      entropy_exceeded      → EntropyChecker
      safety_violation      → PatternChecker or InputGuardChecker
      translation_failed    → Translator
      guard_unavailable     → InputGuardChecker (or OutputGuardChecker in output pipeline)
      guard_evaluation_corrupt → InputGuardChecker (or OutputGuardChecker in output pipeline)
    Per-checker granularity is not in the audit log by design — the failure_mode provides the signal
    needed to distinguish attack categories from infrastructure failures without exposing internal routing.
```

---

### Auditor

```
Component: Auditor
Purpose:   Receives log() calls throughout evaluate() and writes structured audit records.
           The composition root constructs the Auditor and closes it last in teardown.

Stateful or stateless: Stateful — holds a write destination (e.g. a file handle) and
  any buffering state required for async writes.

States: ready → closed

Invariants:
  - log() MUST NOT fail the calling code — a write failure is absorbed internally.
  - Credentials MUST NOT appear in any record written by log().

Preconditions for log():
  - event is a non-null, non-empty string.
  - data is a non-null dict.
  - Violation behaviour: the Auditor detects the violation, silently discards the call,
    and MUST NOT propagate any exception to the caller. log() never raises regardless
    of argument validity. A log failure MUST NOT fail the request.

Operations:
  log(event: string, data: dict) -> None
    Writes one audit record. Valid in ready state. After close(): no-op — a log()
    call on a closed auditor silently does nothing. Never raises regardless of state.

  close() -> None
    Flushes buffered records and releases held resources. Idempotent — a second call
    has no effect. Never raises.

Guarantees:
  - Never raises in log() — write failures are absorbed.
  - close() is idempotent.
  - log() after close() is a no-op (no exception, no write attempt).
```

---

### ServerBackend

```
Component: ServerBackend
Purpose:   Provides the HTTP transport layer. Exposes /v1/chat/completions and delegates each
           incoming request to Carapex via the ChatHandler callable.

Stateful or stateless: Stateful — holds a running HTTP server instance and its connections.

Invariants:
  - serve() blocks until shutdown is initiated.
  - The backend owns transport only — it does not inspect, parse, or modify request or response content
    beyond what is required for HTTP framing.

Preconditions for serve():
  - handler is a non-null callable matching the ChatHandler signature.
  - Violation behaviour: raises a programming error.

Accepts:
  handler: ChatHandler — the callable to invoke for each incoming request.
  Config is supplied at construction time via build() — serve() requires no config parameter.

Operations:
  serve(handler) -> None
    Starts the HTTP server. Blocks until shutdown. Calls handler for each request.
    Never raises on individual request failures — a failed request returns an HTTP error response.
    Raises on startup failure (port already in use, invalid config, etc.).

  close() -> None
    Stops the server and releases resources. Idempotent. Never raises.

Guarantees:
  - A request that handler raises on produces an HTTP 500 response. The exception is not propagated.
  - close() stops accepting new requests. In-flight requests complete before shutdown.

Failure behaviour:
  Startup failure: raises before serve() returns control to the caller. No server is running.
  Individual request failure: HTTP 500. No exception propagation. Logged to stderr.
```

---

## §6 Lifecycle

### Carapex

States: `uninitialised` → `ready` → `closed`

**`uninitialised`** — before `build()` returns. Not a valid state for callers — `build()` never exposes a partially constructed instance.

**`ready`** — after `build()` returns successfully. All components are initialised. LLM clients are constructed and configured but not verified — reachability is confirmed on first call, not at startup.
- `evaluate()` is valid and may be called repeatedly. The instance stays in `ready` between calls.

**`closed`** — after `close()` is called.
- `evaluate()` MUST NOT be called. Behaviour is undefined.
- `serve()` MUST NOT be called. Behaviour is undefined.
- `close()` is idempotent.
- Cannot be reused after `close()`.

**`invalid`** — after `evaluate()` raises `PipelineInternalError`.
- `evaluate()` MUST NOT be called. Behaviour is undefined.
- `close()` SHOULD be called to release resources, but MAY not succeed — component state is unknown.
- The instance cannot be recovered. Discard it and construct a new one via `build()`.
- Distinct from `closed`: `invalid` is reached through a component bug, not through normal teardown.

**Valid call sequence:**
```
instance = build(config)
instance.evaluate(messages)     # repeats as needed
instance.evaluate(messages)
...
instance.close()
```

**Construction failure.** If `build()` raises, no instance is returned. All components constructed before the failure are closed before the exception propagates.

---

### CarapexConfig

`CarapexConfig.load()` and `CarapexConfig.write_default()` are stateless class-level operations — they construct or write a value and return. No instance is created, no resources are held, no teardown is required. §6 lifecycle and §10 atomicity do not apply.

---

### Stateless checkers

Normaliser, PatternChecker, EntropyChecker, ScriptChecker, OutputPatternChecker — no lifecycle to specify. These components hold no mutable state. Valid for use immediately after construction and remain valid indefinitely. No `close()` requirement.

The Normaliser exposes `normalise(text)` as its single operation. No construction-time setup or teardown applies — the method is valid for the lifetime of the object.

---

### Checkers with LLM references

Translator, InputGuardChecker, OutputGuardChecker — states: `ready` → `closed`

**`ready`** — after construction. `inspect()` is valid. These components hold references to injected LLMs. They do not own the LLM lifecycle — the composition root does.

**`closed`** — after the composition root closes the LLMs they reference. These components implement `close()` as a no-op — they own no resources. The method exists because `CheckerCoordinator.close()` calls `close()` on every child checker uniformly, without knowing which ones own resources and which don't. A checker that omits `close()` breaks that uniform teardown. Calling `inspect()` after the LLM is closed is a programming error.

The current checker contract is stateless per call — each `inspect()` is independent with no accumulated context between calls. A future checker that maintains state across calls requires a new contract. The current `SafetyChecker` abstraction does not support it.

---

### CheckerCoordinator

States: `ready` → `closed`

`close()` attempts to close all child checkers regardless of individual failures. If a child raises during `close()`, the exception is logged and closing continues. The last exception raised, if any, is re-raised after all children have been attempted. Idempotent.

---

### LLMs

States: `ready` → `closed`

**`ready`** — after construction. `complete()` is valid. The LLM client is constructed once and its connections are reused across all `evaluate()` calls for the lifetime of the instance.

`complete(messages, api_key)` accepts a messages list and the caller's API key. `messages` is `list[dict]` where each dict carries `role` and `content`. The full list is passed as-is. `api_key` is a string — always provided by the orchestrator, never null, may be empty string when no Authorization header was supplied by the caller. The provider does not extract, inspect, or transform the messages. That is carapex's responsibility, already done before `complete()` is called.

**`closed`** — after `close()` is called by the composition root.
- No operations are valid.
- `close()` is idempotent.
- Cannot be reused after `close()`.

**What `close()` does.** Releases local client resources — connections, connection pools, thread pools, API client handles. It does not terminate the remote LLM server. The server continues running independently of carapex.

**Session state.** The input guard LLM, output guard LLM, and translator LLM are stateless — each call is independent, no context is carried between calls. The main LLM receives whatever history the caller passed in the messages list. carapex does not manage, accumulate, or reset the messages list between calls. Session management is the caller's responsibility — the caller decides whether to include history. See §19 for the implications on multi-turn grooming.


**Ownership.** LLMs are owned by the composition root. `Carapex.close()` closes LLMs after all checkers have been closed. A checker MUST NOT call `close()` on an injected LLM.


---

### Auditor

States: `ready` → `closed`

**`ready`** — after construction. `log()` is valid and writes audit records. Remains in `ready` for the lifetime of the instance until `close()` is called.

**`closed`** — after `close()` is called.
- `log()` is a no-op. No exception is raised. No record is written.
- `close()` is idempotent — a second call has no effect.

Close order: the Auditor is closed last in the `Carapex.close()` sequence, after all pipeline components. This ensures all in-flight pipeline stages can emit audit records before the destination is closed.

---

---

### ServerBackend

States: `uninitialised` → `serving` → `closed`

**`uninitialised`** — before `serve()` is called.

**`serving`** — while `serve()` is blocking. The HTTP server is accepting requests. `close()` transitions to `closed`.

**`closed`** — after `close()` is called or after `serve()` returns. No operations are valid. `close()` is idempotent.

`ServerBackend` is optional. When `server` config is absent, no `ServerBackend` is constructed and `Carapex.serve()` raises `ConfigurationError` if called.

If `serve()` raises at startup (port conflict, invalid config, or other startup failure), the backend transitions directly to `closed` without entering `serving`. `close()` is safe to call on a backend in this state — it is idempotent and never raises. The backend cannot be recovered after a startup failure — discard it and construct a new instance via `build()`.

---

## §7 Failure Taxonomy

**Failure vs. defined operational outcome.** A `SafetyResult` with `safe=false` and `failure_mode="safety_violation"` is a defined outcome — the system evaluated the content and refused it. A failure is a condition that prevented the system from completing evaluation at all. Both surface in `EvaluationResult.failure_mode`, but they are semantically distinct and callers MUST treat them differently.

The failure mode set is open. Extensions — custom checkers, specialised guards, policy evaluators — MAY produce failure modes not listed here. Callers MUST NOT assume this list is exhaustive.

**Handling contract for unknown failure modes.** When a caller receives a `failure_mode` value not in this list, it MUST treat it as an infrastructure failure: `safe=false`, do not retry unchanged, do not pass the request through. This is the most conservative posture and is consistent with the fail-closed design of the system. An unknown failure mode that is silently ignored is a security gap.

**Base failure modes.** The following are defined by the core system. Extensions MAY add to this set — see handling contract above.

| Name | Meaning | Category | Recoverability |
|---|---|---|---|
| `null` | All checks passed. Not a failure. | — | — |
| `"safety_violation"` | Content refused by pattern checker or guard. | Adversarial signal | Permanent for this input. Do not retry unchanged. |
| `"entropy_exceeded"` | Input entropy exceeded threshold after normalisation. | Adversarial signal | Permanent for this input. Do not retry unchanged. |
| `"guard_unavailable"` | Guard LLM returned no response. | Operational failure | Retry after backoff. Monitor for persistence. |
| `"guard_evaluation_corrupt"` | Guard responded but output was unparseable or missing the `"safe"` key. | Security signal. The guard responded with malformed output — possible manipulation. Treat as a potential injection event. | Do not retry without investigation. |
| `"translation_failed"` | Language detection or translation call failed. | Operational failure | Retry after backoff. |
| `"normalisation_unstable"` | Input did not converge after max passes, or a decoder cycle was detected. | Security signal — may indicate a custom decoder defect, or an evasion attempt. | Do not retry unchanged. Review decoder configuration before retrying. |
| `"llm_unavailable"` | Main LLM returned no response. | Operational failure | Retry after backoff. Monitor for persistence. |


**Information carried.** All failure modes surface in `EvaluationResult.failure_mode` as a string. `EvaluationResult.reason` carries a human-readable explanation where one is available — it is present for content refusals and may be null for infrastructure failures. See §4 for the full `EvaluationResult` schema.

**Representation.** Failure modes are always returned as values in `EvaluationResult`. They are never raised as exceptions by `evaluate()`. The exception hierarchy in §17 exists for callers who choose to raise exceptions from `EvaluationResult` values — the exceptions are not raised internally.

**What is NOT done on any failure.** The main LLM is never called if an input check fails. The caller's prompt is never logged in full — only metadata is recorded. A failed input guard call never produces `safe=true`.


---

## §8 Boundary Conditions

### `evaluate(messages)`

**Empty messages list.** Not valid. Raises a programming error before the pipeline runs.

**`null`/`None` messages.** Not valid. Raises a programming error before the pipeline runs.

**Messages list with no user-role entry.** Violates the OpenAI Chat Completions API contract — this is a programming error in the caller. Raises a programming error before the pipeline runs. See §22.

**Messages list with only a system message.** Same as above — raises a programming error.

**Last user message with null content.** Not valid. `content` is typed as string in §3 — a null value violates the type contract. Raises a programming error before the pipeline runs. Same treatment as a null messages list.

**Last user message with empty string content.** Valid input. The pipeline runs normally on empty text. Pattern and input guard checks run on empty text. Entropy check passes unconditionally — empty string is below the minimum length threshold. The normaliser returns the empty string unchanged with `stable=true`.

**Last user message with whitespace-only content.** Valid input. The pipeline runs normally. If the built-in whitespace decoder is active (it is part of the default decoder set — see §15 normaliser config), the normaliser collapses whitespace and subsequent checks run on the resulting empty string. If a custom decoder set is configured that omits the whitespace decoder, whitespace-only content passes through unchanged — subsequent checks run on the original whitespace string. The pipeline does not special-case whitespace-only input independently of the decoder set.

**Messages list with history (multiple turns).** Valid. The full list is passed to the main LLM. Only the last user message is evaluated by the safety pipeline. History is not evaluated. This is a documented limitation — see §19.

**Very short strings (below entropy minimum length).** The entropy check passes unconditionally for strings shorter than the configured minimum length (default: 50 characters). Pattern and input guard checks still run.

**String at exactly entropy minimum length.** Entropy is computed. The check applies normally.

**Strings containing only high-entropy content.** If entropy exceeds the threshold after normalisation, `failure_mode="entropy_exceeded"` regardless of content. This includes legitimate cryptographic material — operators who regularly process such input MUST set `entropy_threshold: null`.

**Mixed encoded and plain text.** Valid input. The normaliser operates on the full string — each decoder pass processes the entire string and the reductive constraint ensures natural language portions are not corrupted. A prompt containing both plain text and an encoded payload (e.g. "summarise this: <base64 blob>") stabilises when the encoded portions are decoded. The natural language prefix was already canonical and is unchanged.

**Strings requiring many decoding passes.** If the input requires more passes than `max_passes` to converge, `stable=false` and `failure_mode="normalisation_unstable"`. The partially decoded text is not forwarded.

---

### `chat(request, headers)` — HTTP path

**Absent `Authorization` header.** Valid. The `authorization` key is not present in the `headers` dict. `ChatHandler` passes an empty string as `api_key` to all LLM calls. Any LLM that requires authentication will return `None` when called with an empty key — `evaluate()` returns the appropriate failure mode (`llm_unavailable` or `guard_unavailable`). No special casing applies: the orchestrator treats an empty key identically to any other key.

**Empty `Authorization` header value.** Equivalent to absent — `api_key` is the empty string. Same behaviour as above.

**`messages` list with no user-role entry, null, or empty — via HTTP.** `evaluate()` raises a programming error (`ValueError`) before the pipeline runs. `chat()` catches the exception and returns HTTP 400 with `{"error": {"message": "<description>", "type": "invalid_request_error", "code": "invalid_messages"}}`. No audit record is produced. See §3 and §17.

---

### `build(config)`

**Valid config, all LLMs reachable.** Returns a fully initialised `Carapex` instance.

**Valid config, LLM URL syntactically valid but unreachable.** `build()` succeeds. First call to `evaluate()` returns the appropriate failure mode — `llm_unavailable` or `guard_unavailable`. No exception at build time.

**Invalid config — bad pattern regex.** Raises `ConfigurationError` before any component is constructed.

**Invalid config — missing or unrecognised `type`.** Raises `ConfigurationError` before any component is constructed.

**Invalid config — `injection_patterns: []` or `null`.** Both raise `ConfigurationError`. An empty pattern set removes the pattern check layer entirely — this is always a misconfiguration in practice and possibly an attack vector: a compromise of the configuration path that clears the pattern list would silently disable structural injection detection if silent restoration were the behaviour. Raising on an empty set forces the operator to make the absence of patterns a deliberate, explicit choice — by setting a known safe default or a non-empty custom set. The error message MUST identify which field caused the failure. To explicitly disable pattern checking, operators MUST supply a configuration with patterns that match nothing — empty set is not a valid representation of that intent.

**`output_guard_enabled: false` with `output_guard_llm` configured.** Raises `ConfigurationError`. Supplying an LLM for a disabled component is a contradiction — it indicates operator confusion about the configuration. Raising forces the operator to resolve the contradiction explicitly rather than silently ignoring an LLM that will never be used.

**`output_guard_llm: null` with `input_guard_llm: null`.** Valid. Both resolve to the main LLM, which serves as guard, output guard, and main LLM. This is a valid single-model deployment — operationally the weakest configuration but not a misconfiguration. See §15 for the tradeoff rationale.

**Duplicate extension name.** If two registered components share the same `name` attribute, `RuntimeError` is raised during autodiscovery at import time — before any `build()` call is possible. The application cannot start. This is not a `build()` failure — it occurs when the module containing the duplicate is first imported.

**Invalid config — `max_passes: 0`.** Raises `ConfigurationError`. §15 requires `max_passes >= 1`. Zero is not valid — a normaliser that is permitted zero passes cannot evaluate any input. The error message MUST identify the field.

**Invalid config — `max_passes` negative.** Raises `ConfigurationError`. Same rationale as zero.

**Invalid config — `entropy_min_length: 0` or negative.** Valid. Any value ≤ 0 means entropy is computed for all inputs — entropy is always ≥ 0, so no string can be shorter than a non-positive minimum length. No `ConfigurationError` is raised.

**Invalid config — `input_guard_temperature: 0.0` or `output_guard_temperature: 0.0`.** Raises `ConfigurationError`. §15 explicitly states "Do not set to 0.0" — a fully greedy model can produce degenerate outputs on ambiguous inputs. This is a safety-relevant constraint. Raising at `build()` time prevents silent degradation rather than emitting an advisory comment. The error message MUST identify the field.

**Invalid config — `translation_temperature` outside range [0.0, 1.0] inclusive or `input_guard_temperature` / `output_guard_temperature` outside range (0.0, 1.0] exclusive.** Raises `ConfigurationError`. Values outside the meaningful range for the respective temperature semantic are always misconfiguration.

---

### EntropyChecker thresholds

**`entropy_threshold: null`.** Entropy check is disabled. All inputs pass unconditionally regardless of entropy value.

**`entropy_threshold` at exactly the threshold value.** The check is strictly greater-than — input MUST exceed the threshold to be rejected. Fail closed means rejecting known bad content, not the boundary itself.

---

### Normaliser passes

**`max_passes: 1`.** One decoding pass. If the output differs from the input, `stable=false`. The decoded output is available in `NormaliserResult.text` but the orchestrator rejects the request.

**`max_passes` exhausted with no change on final pass.** If the last pass produced no change, `stable=true` even if earlier passes did produce changes. Stability is determined by the final pass only.


---

## §9 Sentinel Values and Encoding Conventions

### `safe` (boolean)

Appears in: `EvaluationResult`, `SafetyResult`, audit log records.

`true` — all checks passed. In `EvaluationResult`, implies `content` is present and `failure_mode` is null. In a `SafetyResult`, means the evaluating component found no issue. In audit log records, means the check at that stage passed.

`false` — a check failed or a component was unavailable. Always accompanied by a non-null `failure_mode`. A caller MUST NOT treat `safe=false` as equivalent to a content refusal without inspecting `failure_mode` — infrastructure failures also produce `safe=false`.

`safe` is never null. It is always the boolean literal `true` or `false`.

**Special case — InputGuardChecker and OutputGuardChecker.** Only the boolean literal `true` in the guard's JSON response is accepted as safe. `"true"` (string), `1` (integer), absent, and `null` all produce `safe=false`. This applies equally to both InputGuardChecker and OutputGuardChecker — both parse the same JSON response format and both enforce the same strict boolean-literal contract. This is not a general convention — it is a specific guard contract to prevent type-coercion bypasses.

---

### `failure_mode` (string or null)

Appears in: `EvaluationResult`, `SafetyResult`, audit log records.

`null` — no failure. All checks passed. Present on `EvaluationResult` when `safe=true`. Present on audit records for the `evaluate_complete` event when the run succeeded.

Non-null string — identifies the specific failure cause. See §7 for the base set and the handling contract for unknown values. A caller MUST branch on `failure_mode` to distinguish content refusals from infrastructure failures.

`failure_mode` is null if and only if `safe=true`.

---

### `stable` (boolean)

Appears in: `NormaliserResult`, audit log `input_normalised` event.

`true` — the normalised output did not change on the final pass. The input has been fully decoded by the configured decoder set.

`false` — the output did not converge. Either `max_passes` was exhausted with the output still changing, or a decoder cycle was detected. The orchestrator treats `stable=false` as `failure_mode="normalisation_unstable"`.

`stable` is never null.

---

### `detected_language` (string or null)

Appears in: `ScriptResult` (internal — not in `EvaluationResult`).

Non-null string — the ISO language code detected by the language detection library. Present when detection succeeded and confidence is at or above `safety.script_confidence_threshold`.

`null` — detection failed, or confidence was below `safety.script_confidence_threshold`. When `detected_language` is null, `translation_needed` is always `true` — the safe fallback.

`detected_language` is never used directly by anything downstream of `ScriptChecker`. It is surfaced for diagnostic purposes only. All routing decisions are made on `translation_needed`.

---

### `translation_needed` (boolean)

Appears in: `ScriptResult` (internal — not in `EvaluationResult`).

`true` — the detected language is not English, detection failed, or detection confidence is below `safety.script_confidence_threshold`. `Translator` will attempt translation.

`false` — the detected language is English and detection confidence is at or above `safety.script_confidence_threshold` (default: 0.80). `Translator` passes the text through unchanged.

Failure-safe: any ambiguity maps to `true`. An untranslated non-English prompt reaching the input guard is the worse outcome.

---

### `content` (string or null)

Appears in: `EvaluationResult`.

Non-null string — the LLM's response text. Present if and only if `safe=true`.

`null` — the LLM was not called (input check failed), or the LLM call failed. `failure_mode` identifies which.

`content` is null if and only if `safe=false`. These are not independent fields — a caller MUST NOT check `content` for null as a proxy for check status. Use `safe` and `failure_mode`.

---

### `reason` (string or null)

Appears in: `EvaluationResult`, `SafetyResult` (internal).

Non-null string — a human-readable explanation of why the prompt or response was blocked. Present when `safe=false` and the blocking component could produce a description (content refusals: `safety_violation`, `entropy_exceeded`).

`null` — no explanation available. Present when `safe=true`, or when `safe=false` due to an infrastructure failure mode (`guard_unavailable`, `guard_evaluation_corrupt`, `translation_failed`, `llm_unavailable`, `normalisation_unstable`).

`reason` is free text. It is not a stable identifier. Callers MUST NOT branch on its value — use `failure_mode` for programmatic handling.

---

### `finish_reason` (string — fixed value `"content_filter"`)

Appears in: `ChatCompletionResponse` (HTTP interface only) when `safe=false`.

Always the string `"content_filter"` when carapex blocks a request. This is an existing OpenAI Chat Completions enum value — not a carapex-specific invention. It signals to the caller that the response was stopped by a content filter.

`finish_reason` is not customisable. Callers who switch on `finish_reason` see expected behaviour. Callers who do not switch on it receive a safe assistant message in `content` regardless.

`finish_reason` is absent from `EvaluationResult` — it appears only in the HTTP response shape. It is never `null` in a blocked response — it is always the literal string `"content_filter"`.

---

### Token count fields (`prompt_tokens`, `completion_tokens`)

Appear in: `llm_call` audit log records only.

Token counts are not present on `EvaluationResult`. They are recorded exclusively in the audit log `llm_call` records, one record per LLM call.

In audit log `llm_call` records: `prompt_tokens`, `completion_tokens`, and `total_tokens` are present when the call succeeded. All three fields are absent when the call failed. When the call succeeded but the provider returned no usage data, all three fields are present and zeroed — absent usage data is not a failure condition (see `CompletionResult` in §2). Field names match the OpenAI API.

Library callers who need per-call token counts MUST read them from the audit log, not from `EvaluationResult`.


---

## §10 Atomicity and State on Failure

**Stateless components.** Normaliser, PatternChecker, EntropyChecker, ScriptChecker, and OutputPatternChecker are stateless — they hold no mutable state. This section is not applicable to them. A failure in any of these components during a `evaluate()` call leaves no internal state to recover from — they are valid for immediate reuse.

---

### `evaluate()`

`evaluate()` is not atomic with respect to side effects. The HTTP layer calls `evaluate()` for each incoming request — the atomicity contract is the same whether the call originates from HTTP or from a direct library call. If `evaluate()` fails midway through the pipeline, audit records for completed stages have already been emitted. They are not retracted. A caller who retries after a failure will produce a second set of audit records for the same logical request — both sets share different `audit_id` values and are independent.

The `Carapex` instance remains in the `ready` state after any `evaluate()` failure, including infrastructure failures. `evaluate()` is safe to call again after a failure.

**Partial pipeline execution is visible in the audit log.** If `evaluate()` fails at the input guard stage, audit records for normalisation and pattern checking are present. The `evaluate_complete` record is always the final record for a call — its absence indicates the process was killed mid-run or that `evaluate()` raised `PipelineInternalError`.

**No durable state is modified by `evaluate()`.** The audit log is append-only. No rollback is possible or necessary.

---

### `build()`

`build()` is atomic with respect to the returned instance. If it raises, no `Carapex` instance is returned and no cleanup is required by the caller. Internally, any components constructed before the failure are cleaned up by `build()` before raising.

**`build()` MUST clean up any components it constructed before raising.** If construction fails partway through, all successfully constructed components are closed before the exception propagates. The caller receives a clean exception with no resource leak.

---

### `close()`

`close()` is not atomic across components. Close order: ServerBackend first (stops accepting new requests), then checkers, then LLMs, then Auditor last (so all pipeline components can emit audit records until teardown completes). Full rationale in §12.

`close()` MUST attempt to close all components regardless of individual failures. If a component raises during `close()`, the exception is logged and closing continues. All components are closed on a best-effort basis. If any component raised, the last exception is re-raised after all components have been attempted.

---

### `serve()`

`serve()` is a blocking call that holds no durable state of its own — the `Carapex` instance and pipeline remain in `ready` state while `serve()` is running. When `serve()` returns for any reason (SIGTERM, unrecoverable server error, operator-initiated shutdown), `ServerBackend` transitions to `closed`. The pipeline is not affected — `Carapex` remains in `ready`. The operator MUST call `close()` explicitly after `serve()` returns to release all pipeline resources. `serve()` returning does not constitute teardown.

If `serve()` raises (startup failure — port conflict, invalid config, or other `ServerBackend` startup error), `Carapex` also remains in `ready` state. The `ServerBackend` transitions directly to `closed` without entering `serving` — see §6a. The pipeline is unaffected and `evaluate()` remains valid. The operator MUST NOT call `serve()` again on the same instance — the `ServerBackend` cannot be recovered. Discard the instance and construct a new one via `build()`.

---

### Ungraceful shutdown

The audit log is append-only JSONL. A process killed mid-write may produce a partial final record — a JSON object that is not valid JSON. Consumers of the audit log MUST handle malformed final records gracefully by skipping them. All preceding records are intact.

No other durable state exists. There is no recovery procedure required on next startup. A new `build()` call produces a clean instance.


---

## §11 Ordering and Sequencing

### Input pipeline

`Carapex` orchestrates the full pipeline. The input pipeline runs first:

```
messages: List[dict]
  → extract last user message (content: str)
  → Normaliser
  → EntropyChecker
  → PatternChecker
  → ScriptChecker
  → Translator
  → InputGuardChecker
  → [main LLM call — receives original messages list, unmodified]
```

**Normaliser MUST precede all checkers** because checkers evaluate decoded content. A pattern check on encoded input would miss encoded injection tokens. An entropy check on encoded input would measure encoding overhead, not content entropy. Normalisation is the prerequisite for meaningful evaluation.

**EntropyChecker precedes PatternChecker** because entropy computation is cheaper than regex iteration on the hot path. The vast majority of inputs are legitimate — entropy passes in a single O(n) string pass with one threshold comparison. Pattern matching iterates compiled regexes against the full string. Failing fast on high entropy avoids the regex pass entirely. Both checks are deterministic and operate on normalised text — reversing them produces the same correctness outcome but at higher cost on legitimate input. The ordering is a performance decision, not a correctness one: no strict ordering is required between these two checks for the pipeline to produce correct results, but cost-ordering applies when there is no strict dependency. Strings below the minimum length threshold skip entropy — PatternChecker runs next regardless.

**PatternChecker MUST precede ScriptChecker** — cheap deterministic check before an external library call.

**ScriptChecker MUST precede Translator** because Translator reads `ScriptResult.translation_needed` from ScriptChecker's output. Translator cannot operate without it.

**Translator MUST precede InputGuardChecker** because the guard evaluates English text only. Passing non-English text to the guard risks misclassification. Translation is a correctness prerequisite for input guard evaluation, not a performance optimisation.

**InputGuardChecker MUST be last in the input pipeline** because it is the most expensive check (one LLM call). All cheaper checks must have passed before it runs.

**The main LLM receives the original messages list, not the working text.** The working text is derived from the last user message and may be transformed by the Translator. The main LLM MUST receive the full messages list as the caller sent it — normalisation and translation are evaluation tools, not input transformations. The caller's history, system prompt, and assistant turns are passed through unchanged.

---

### Output pipeline

After the main LLM returns a response, `Carapex` passes it to the `OutputCoordinator` (`CheckerCoordinator`) for output evaluation:

```
[LLM response]
  → OutputPatternChecker
  → OutputGuardChecker (if enabled)
```

**OutputPatternChecker MUST precede OutputGuardChecker** because pattern matching is cheaper and deterministic. A known jailbreak success indicator in the response is caught without an LLM call.

**OutputGuardChecker is the final stage.** It requires the complete response text. It cannot be moved earlier.

**Empty output pipeline.** If `output_guard_enabled: false` and no output patterns are configured, `OutputCoordinator` holds an empty checker sequence and passes the LLM response unconditionally. This is a deliberate deployment choice — the security tradeoff is documented in §15. The empty coordinator does not emit an `output_safety_check` audit record.

---

### Early termination

If any stage produces `safe=false`, the pipeline stops. No subsequent stage runs. The LLM is not called if any input check fails. The output pipeline does not run if the LLM call fails.

This is a correctness and efficiency constraint — running subsequent checks after a failure provides no additional signal and consumes resources unnecessarily.


---

## §12 Interaction Contracts

### Normaliser → CheckerCoordinator (input)

The normaliser runs before the coordinator. The coordinator receives the normalised text as its initial working text. If `stable=false`, the orchestrator rejects the request before the coordinator runs — the coordinator never sees unstable output.

The orchestrator owns this handoff. Neither the normaliser nor the coordinator knows about the other.

---

### ScriptChecker → Translator

This is the only direct inter-checker dependency in the pipeline.

**Initiator:** CheckerCoordinator, on behalf of Translator.
**Mechanism:** Before calling `Translator.inspect()`, the coordinator calls `Translator.set_prior_result()` with the `SafetyResult` returned by ScriptChecker.

**What Translator receives:** A `ScriptResult` — which MUST be a `SafetyResult` subclass (see §2) — carrying `detected_language` and `translation_needed`. Translator reads `translation_needed` directly. It does not re-detect the language.

**What the coordinator knows:** That Translator is a `TextTransformingChecker`. It does not know it is a `Translator` specifically. The `set_prior_result()` call is part of the `TextTransformingChecker` contract, not a `Translator`-specific call.

**Postcondition the coordinator relies on:** After a `safe=true` result from any `TextTransformingChecker`, calling `get_output_text()` returns the transformed text. The coordinator switches the working text to this value for all subsequent checkers.

**What the caller guarantees:** ScriptChecker immediately precedes Translator in the pipeline. The coordinator MUST call `set_prior_result()` before `inspect()` on any `TextTransformingChecker`. Calling `inspect()` without `set_prior_result()` is a programming error.

**Resource ownership:** Translator holds a reference to the translator LLM for translation calls. It does not own the LLM — the composition root does. Translator MUST NOT call `close()` on the translator LLM.

---

### InputGuardChecker / OutputGuardChecker → Guard LLMs

InputGuardChecker holds a reference to the input guard LLM. OutputGuardChecker holds a reference to the output guard LLM — which MAY be the same instance as the main LLM (when `output_guard_llm` is null in config), or a distinct instance (when `output_guard_llm` is explicitly configured). Neither checker owns the LLM it holds.

**Who constructs:** The composition root (`build()`).
**Who closes:** `Carapex.close()`, after all checkers have been closed. If the output guard LLM is the same instance as the main LLM, it is closed once. If they are distinct instances, each is closed separately.
**Who MUST NOT close:** InputGuardChecker, OutputGuardChecker, Translator.

If each checker called `close()` on the shared LLM, it would be closed on the first call and subsequent calls would operate on a closed resource. Centralised ownership in the composition root avoids this without reference counting. In the distinct-instance case, the composition root tracks both instances independently.

---

### Carapex → LLMs

**Main LLM** — receives the original prompt after all input checks pass. Called via `complete(messages, api_key)` with the caller's key. Returns the LLM response text and token usage.

**Input Guard LLM** — receives safety evaluation requests via InputGuardChecker. Called via `complete_with_temperature(messages, temperature, api_key)` with the caller's key. Receives one call per `evaluate()` invocation for input evaluation. Stateless — each call is independent.

**Output Guard LLM** — receives safety evaluation requests via OutputGuardChecker. Called via `complete_with_temperature(messages, temperature, api_key)` with the caller's key. Receives one call per `evaluate()` invocation where output checking is enabled. When `output_guard_llm` is null in config, this role is served by the main LLM instance — not the input guard LLM.

**Translator LLM** — receives translation requests (via Translator) when `translation_needed=true`. Called via `complete_with_temperature(messages, temperature, api_key)` with the caller's key. Receives one call per `evaluate()` invocation where translation was required. If `translator_llm` is not configured, falls back to the main LLM.

The orchestrator applies the same call convention to every LLM. No special casing on role, credential type, or whether the LLM is the main instance or a fallback. The `api_key` parameter is always the caller's key — what each LLM implementation does with it is its own concern.

**Ordering guarantee:** The main LLM is never called before all input checkers have returned `safe=true`. The output pipeline never runs before the main LLM has returned a response.

**Callback contract:** All LLM calls are synchronous request-response. No callbacks.

---

### Carapex → Auditor

The auditor receives `log()` calls throughout `evaluate()`. It does not affect control flow — a `log()` failure does not fail the request.

**Who constructs:** The composition root.
**Who closes:** `Carapex.close()`, after checkers and LLMs.
**Ownership:** The composition root owns the auditor lifecycle exclusively.

**Thread safety:** The auditor MUST be safe for concurrent `log()` calls if `evaluate()` may be called concurrently. See §13.

---

### Composition Root → All Components

The composition root is the only location that knows about concrete classes. All components below it depend only on abstractions. The composition root constructs components in dependency order — a component is never constructed before its dependencies.

**Close order:** ServerBackend is closed first (stops accepting new requests). Then checkers, then LLMs. The auditor is closed last. This ensures no component attempts to log after the auditor is closed, no checker attempts an LLM call after the LLM is closed, and no new requests arrive while pipeline components are shutting down. Closing ServerBackend first is the correct intake-stop step even though graceful drain of in-flight requests is not currently implemented — see §21.


---

## §13 Concurrency and Re-entrancy

### `Carapex.evaluate()`

`evaluate()` is safe to call concurrently from multiple threads. Each call is an independent pipeline execution with its own working text, `audit_id`, and delimiter values. No mutable state is shared between concurrent calls.

The input guard LLM and main LLM MUST be safe for concurrent calls if concurrent `evaluate()` invocations are expected — LLM implementations are responsible for their own thread safety.

Re-entrant calls to `evaluate()` (calling `evaluate()` from within a callback invoked by `evaluate()`) are not possible — `evaluate()` accepts no callbacks.

---

### `Carapex.close()`

`close()` MUST NOT be called concurrently with `evaluate()`. The caller is responsible for ensuring all in-flight `evaluate()` calls complete before calling `close()`. The consequence of violating this constraint is undefined behaviour — a `evaluate()` call may attempt to use a closed LLM or checker.

---

### ChatHandler concurrency

The `ServerBackend` MAY call `ChatHandler` concurrently from multiple threads. `ChatHandler` is re-entrant — concurrent and nested invocations are safe. This is safe because `ChatHandler` calls `evaluate()`, and `evaluate()` is safe for concurrent calls (see `Carapex.evaluate()` above). Each concurrent handler invocation produces an independent `evaluate()` execution with its own working text, `audit_id`, and delimiter values. No handler-level locking is required or expected.

---

### Registries

Registries are written during autodiscovery at import time, before any `build()` call. After autodiscovery completes, registries are read-only. Concurrent reads after startup are safe. Concurrent writes — dynamic registration after startup — are not supported and produce undefined behaviour.

---

### Auditor

The built-in file auditor is thread-safe. Concurrent `log()` calls from concurrent `evaluate()` invocations are safe — records are written atomically per line. Record interleaving across concurrent calls is possible — use `audit_id` to correlate records from a single call, not line order.

Custom `Auditor` implementations MUST be thread-safe if concurrent `evaluate()` calls are expected.

---

### Language detection library seeding

The language detection library is seeded once at `ScriptChecker` construction with a fixed value. The seed does not change after construction. If the library exposes seeding only as a process-level global rather than a per-instance parameter, the implementation MUST treat construction of `ScriptChecker` as a globally-visible side effect. This is safe if `ScriptChecker` is constructed before any concurrent use begins — i.e. before `build()` returns. Constructing `ScriptChecker` concurrently with another construction or with an active `evaluate()` call produces undefined behaviour.

The composition root constructs all components sequentially. Concurrent construction is not supported.

---

### Cross-reference §22

§22 Assumptions declares that `build()` is called from a single thread before any concurrent `evaluate()` calls begin. §13 and §22 are consistent — the concurrency guarantees above hold only if that assumption is satisfied.


---

## §14 External Dependencies

### Input Guard LLM (optional)

An LLM service reachable over the network or in-process. Optional — when `input_guard_llm` is null in config, the main LLM serves this role.

**Credentials:** The caller's `Authorization` header is passed as `api_key` to all calls on this LLM instance. carapex never manages or stores guard credentials separately.
**Misconfigured URL:** `build()` raises `ConfigurationError`. No `Carapex` instance is returned.
**Unreachable at runtime:** `evaluate()` returns `failure_mode="guard_unavailable"`. The request is rejected. The main LLM is not called.
**Verified:** Not at startup. First call failure surfaces as `guard_unavailable`. This is intentional — see the rationale below.

---

### Main LLM (required)

An LLM service reachable over the network or in-process. Required for producing responses — without it, `evaluate()` can evaluate prompts but cannot return content.

**Credentials:** The caller's `Authorization` header is passed as `api_key` to all calls on this LLM instance.
**Misconfigured URL:** `build()` raises `ConfigurationError`. No `Carapex` instance is returned.
**Unreachable at runtime:** `evaluate()` returns `failure_mode="llm_unavailable"`. The pipeline ran fully — input was evaluated — but no response was produced.
**Verified:** Not at startup. First call failure surfaces as `llm_unavailable`. This is intentional — see the rationale below.

---

### Output Guard LLM (optional)

An LLM service reachable over the network or in-process. Optional — absent when `output_guard_enabled: false`. When enabled and `output_guard_llm` is null in config, this role is served by the main LLM instance; when `output_guard_llm` is explicitly configured, it is a distinct instance.

**Credentials:** The caller's `Authorization` header is passed as `api_key` to all calls on this LLM instance.
**Misconfigured URL (when enabled):** `build()` raises `ConfigurationError`. No `Carapex` instance is returned.
**Unreachable at runtime (when enabled):** `evaluate()` returns `failure_mode="guard_unavailable"`. The response is rejected.
**Verified:** Not at startup. First call failure surfaces as `guard_unavailable`. This is intentional — see the rationale below.
**When disabled:** Not instantiated.

---

### Translator LLM (optional)

An LLM service reachable over the network or in-process. Optional — absent when `translator_llm` is null in config, in which case the main LLM serves as the fallback for translation calls. When `translator_llm` is explicitly configured, it is a distinct instance.

**Credentials:** The caller's `Authorization` header is passed as `api_key` to all calls on this LLM instance.
**Misconfigured URL (when configured):** `build()` raises `ConfigurationError`. No `Carapex` instance is returned.
**Unreachable at runtime:** `evaluate()` returns `failure_mode="translation_failed"`. The request is rejected before the input guard runs.
**Verified:** Not at startup. First call failure surfaces as `translation_failed`. This is intentional — see the rationale below.

---

### Language detection library (required)

A library for detecting the language of input text. Required for script detection and translation routing.

**Absent at startup:** `build()` raises `ConfigurationError` or equivalent import/initialisation error. The application cannot start.
**Becomes unavailable at runtime:** Not applicable — the language detection library is an in-process dependency, not a network service. If it raises unexpectedly, `ScriptChecker` catches the failure and sets `translation_needed=true` — the safe fallback.

---

### In-process LLM inference library (optional)

An in-process library for running LLM inference locally, without a network call. Required only when an in-process LLM backend is configured.

**Absent when configured:** `build()` raises `ConfigurationError` at provider construction.
**Absent when not configured:** No impact.

---

### HTTP server framework (optional)

A library for running an HTTP server. Required only when an HTTP server backend is configured.

**Absent when configured:** `build()` raises `ConfigurationError` at server backend construction.
**Absent when not configured:** No impact.

---

### Audit log file (required when FileAuditor is configured)

A writable file path on the local filesystem.

**Absent or unwritable at startup:** `build()` raises `ConfigurationError`.
**Becomes unwritable at runtime:** Audit log writes fail silently. The request is not failed. The failure is logged to stderr if possible. stderr is the only non-circular channel available — the auditor cannot call `log()` on itself to report its own failure, and it cannot raise without breaking the no-raise contract. stderr is the one output channel that is always available regardless of auditor state.


---

### Why carapex does not health-check LLMs at startup

A health check at startup creates a false readiness guarantee. Reachability at `build()` time is not reachability at call time. The check passes, the instance is returned, and the LLM goes down one second later — the guarantee was never real. An operator who relies on `build()` succeeding as evidence the system is operational has a false belief.

The fail-closed contract makes health checks redundant. Every `complete()` call that returns `None` produces `failure_mode="llm_unavailable"` or `"guard_unavailable"`. The pipeline rejects the request. The system behaves correctly whether the LLM was unreachable at startup or went down mid-operation. There is no failure mode that health check prevents that the runtime contract does not already handle.

Health checks at startup would also require credentials — API keys that don't exist at `build()` time in a pass-through auth model. This is not a workaround problem. It is evidence that startup verification and pass-through auth are fundamentally incompatible. Removing health checks resolves the incompatibility cleanly.

A misconfigured URL is caught at `build()` time via `ConfigurationError` — URL format validation requires no network call. Reachability failures surface on the first real call with a specific failure mode. Operators who want periodic liveness verification MUST implement that externally.

---

## §15 Configuration

All configuration is supplied to `build()` via a `CarapexConfig` object. `CarapexConfig.load(path)` deserialises a JSON or YAML file into `CarapexConfig`. `CarapexConfig.write_default(path)` writes a complete configuration file with default values for all registered components. Both are class-level operations on `CarapexConfig` — they are not free functions.

Invalid configuration always raises `ConfigurationError` at `build()` time — never silently at runtime.

---

### Top-level fields

| Field | Type | Default | Meaning |
|---|---|---|---|
| `main_llm` | dict | required | Main LLM config. MUST contain `"type"` matching a registered LLM name. |
| `input_guard_llm` | dict or null | null | Input guard LLM config. If null, falls back to the main LLM. MUST contain `"type"` if present. |
| `output_guard_llm` | dict or null | null | Output guard LLM config. If null, falls back to the main LLM directly. Separate from `input_guard_llm` — input and output guards may have different training and different evaluation goals. Providing a distinct output guard model is preferred for deployments where response evaluation requires different capabilities from prompt evaluation. |
| `translator_llm` | dict or null | null | Translator LLM config. If null, falls back to the main LLM directly. Independent of `input_guard_llm` — translation and input guard evaluation are separate operations with no dependency between them. The main LLM is the universal fallback for any role not explicitly configured. Roles do not inherit from each other except where explicitly stated. |
| `safety` | dict | `{}` | Safety checker configuration. See safety fields below. |
| `normaliser` | dict | `{}` | Normaliser configuration. See normaliser fields below. |
| `audit` | dict | `{}` | Auditor configuration. See audit fields below. |
| `debug` | boolean | `false` | Enables additional diagnostic output. Does not affect safety behaviour. |
| `server` | dict or null | null | Server backend configuration. If null or absent, carapex runs as a library only — `serve()` raises `ConfigurationError`. MUST contain `"type"` if present. See server config fields below. |

`null` and absent mean the same thing for `input_guard_llm` — both result in the main LLM being used as the guard. This is a convenience for single-model deployments where running a second model is impractical. The tradeoff is acknowledged: a separate guard model with different training provides stronger defence-in-depth than self-evaluation. `input_guard_llm: null` is a valid configuration but operators in security-sensitive deployments SHOULD supply a separate guard.

The same equivalence applies to `output_guard_llm` and `translator_llm` — null and absent both result in the main LLM serving that role directly.

---

### LLM config fields (applies to `main_llm`, `input_guard_llm`, `output_guard_llm`, and `translator_llm`)

**LLM extensibility.** The LLM abstraction is designed to be swapped entirely. The translator LLM, input guard LLM, output guard LLM, and main LLM are all resolved through the same registry. This means the input guard can be replaced with a policy-driven evaluator, a RAG-augmented classifier, a fine-tuned safety model, or any implementation that satisfies the `LLMProvider` contract in §16. The same applies to the translator and output guard. No pipeline changes are required — only configuration and the new implementation. See §24 for future directions.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `type` | string | required | Registry name of the LLM implementation. Missing or unrecognised value raises `ConfigurationError`. |
Additional fields are implementation-specific. See §16 Extension Contracts for the LLM extension contract, and individual LLM implementations for their fields.

---

### Safety config fields

| Field | Type | Default | Meaning |
|---|---|---|---|
| `injection_patterns` | list of strings or null | absent (built-in defaults) | Regex patterns for the input PatternChecker. Absent or omitted: built-in default pattern set is used. `null`: raises `ConfigurationError` — see §8 for the full rationale. `[]` also raises `ConfigurationError`. Supply a non-empty list of patterns to use a custom set. Custom patterns replace the built-in set entirely. |
| `entropy_threshold` | float or null | `5.8` | Shannon entropy threshold in bits/character. Input above this value is rejected. `null` disables entropy checking entirely. **Security field** — setting to `null` removes a check layer. The default of 5.8 is chosen to sit in the gap between dense legitimate technical content (code, JSON, URLs: ~4.5–5.5 bits/char) and encoded or obfuscated attack content that survived normalisation (custom ciphers, undecoded binary: ~6.5–8.0 bits/char). Natural language sits lower at ~3.5–4.5 bits/char. Override downward only with evidence that your application's legitimate inputs fall below the threshold — doing so increases false positives. Override upward only if your application regularly handles genuinely high-entropy content such as cryptographic material — doing so reduces detection coverage. |
| `entropy_min_length` | integer | `50` | Minimum character length for entropy evaluation. Inputs below this length skip entropy checking. |
| `script_confidence_threshold` | float | `0.80` | Minimum confidence score (0.0–1.0) for the language detection result to be treated as English. Below this threshold, `translation_needed=true` regardless of detected language. **Security field** — the failure direction is deliberate: under-confidence triggers an unnecessary translation call (cheap, safe); over-confidence on a non-English prompt sends untranslated text to the input guard (the worse outcome). Raise for known-English traffic; lower for very short inputs where detectors produce lower confidence scores. |
| `output_guard_enabled` | boolean | `true` | Whether `OutputGuardChecker` runs. Disabling removes semantic evaluation of LLM responses. **Security field** — disabling weakens output safety coverage. |
| `input_guard_temperature` | float | `0.1` | Temperature for input guard safety classification calls. Near-zero but not zero — a fully greedy model can produce degenerate outputs on genuinely ambiguous edge cases where correct classification is uncertain. 0.1 provides enough variation to escape these without materially affecting determinism. Value MUST be in range (0.0, 1.0] exclusive of 0.0 — `build()` raises `ConfigurationError` for 0.0 or any value outside this range. See §8. |
| `output_guard_temperature` | float | `0.1` | Temperature for output guard safety classification calls. Same rationale as `input_guard_temperature`. Independent — input and output guard models may be configured separately and may have different optimal temperature values. Same valid range and `ConfigurationError` behaviour as `input_guard_temperature`. See §8. |
| `translation_temperature` | float | `0.0` | Temperature for translation calls. Zero — translation must be deterministic. The same input must produce the same English output on every call for consistent input guard evaluation. Value MUST be in range [0.0, 1.0] inclusive — `build()` raises `ConfigurationError` for any value outside this range. **Warning:** any non-zero value violates the §23 translation determinism correctness contract. Non-zero temperature allows the translator to produce different English renderings of the same input on different calls — the input guard may then classify the same prompt differently across calls, undermining the consistency guarantee that deterministic translation provides. Non-zero values are permitted by the range validator but MUST NOT be used in production deployments. |
| `input_guard_system_prompt_path` | string or null | `null` | Path to a file containing the input guard system prompt. `null` uses the built-in default prompt. The file is read once at `build()` time and not re-read at runtime. **Security field** — a custom prompt replaces the built-in prompt entirely. The security properties in §19 apply to the built-in prompt. Operators supplying a custom prompt take full responsibility for its security properties. |
| `output_guard_system_prompt_path` | string or null | `null` | Path to a file containing the output guard system prompt. `null` uses the built-in default. Separate from `input_guard_system_prompt_path` — input and output guards have different evaluation goals. The input guard evaluates intent; the output guard evaluates response content. They may be trained differently, require different instructions, and may run on different models via `output_guard_llm`. Same file-path constraints as `input_guard_system_prompt_path`. |

**Input guard system prompt constraints.** The prompt is loaded at startup and fixed for the lifetime of the instance. It cannot be changed at runtime. A missing or unreadable file raises `ConfigurationError` at `build()` time. An empty file raises `ConfigurationError` — an empty input guard system prompt is a misconfiguration. Changing the input guard system prompt is a deployment operation, not a runtime operation. Operators who require prompt updates must redeploy.

---

### Normaliser config fields

| Field | Type | Default | Meaning |
|---|---|---|---|
| `max_passes` | integer | `5` | Maximum normalisation passes before declaring unstable. Must be >= 1. The default handles all realistic legitimate inputs — most multi-encoded content requires 2–3 passes at most. Raising this value increases the computational cost of processing adversarial inputs designed to exhaust the normaliser. Override only if your application legitimately produces encoding chains deeper than 5 layers. |
| `decoders` | list of strings or null | null | Ordered list of decoder names to apply. `null` uses the full built-in decoder set. `null` is the safe default — any reduction in decoder coverage reduces the normaliser's ability to decode obfuscated payloads. A custom list is appropriate when an operator can demonstrate that their application's inputs will never contain certain encoding types, and the reduced set measurably improves throughput. Any reduction in coverage is a deliberate security tradeoff that MUST be justified. |

---

### Audit config fields

| Field | Type | Default | Meaning |
|---|---|---|---|
| `type` | string | `"file"` | Registry name of the auditor implementation. |
| `path` | string | required when `type` is `"file"` | `FileAuditor` only. File path for JSONL audit log output. |


---

### Server config fields

| Field | Type | Default | Meaning |
|---|---|---|---|
| `type` | string | required | Registry name of the server backend implementation. Missing or unrecognised value raises `ConfigurationError`. |
| `host` | string | `"0.0.0.0"` | Host address to bind. |
| `port` | integer | `8000` | Port to listen on. |

Additional fields are implementation-specific. The built-in `fastapi` backend accepts `workers` (integer, default 1) for the number of uvicorn worker processes.

`server: null` and `server` absent are equivalent — both mean library-only mode. `serve()` raises `ConfigurationError` in both cases.

---

### Security fields

Fields marked **Security field** above affect the security posture of the system. Setting `entropy_threshold: null` or `output_guard_enabled: false` reduces the number of active checks. Supplying a custom `input_guard_system_prompt_path` or `output_guard_system_prompt_path` transfers responsibility for guard security properties to the operator. These MUST be explicitly justified in deployment configuration — they are not safe defaults for general use.


---

## §16 Extension Contracts

carapex is designed so that adding a new implementation requires no changes to any existing component. The only files that change are the new implementation itself and its registration point.

---

### LLMProvider

The abstraction for all LLM implementations — both the main LLM and the input guard LLM.

**MUST implement:**

`complete(messages, api_key) → CompletionResult | None`
Sends a list of messages to the LLM and returns a `CompletionResult`. `api_key` is the caller's API key, always supplied by the orchestrator — never null, but may be an empty string when the HTTP caller supplied no Authorization header. The implementation MUST accept the parameter. Whether it uses the supplied key or its own configured credentials is implementation-specific — the orchestrator applies the same call convention to every LLM with no special casing. Returns `None` if the LLM is unavailable or returns no usable content. "No usable content" follows the OpenAI Chat Completions spec: `content` is `null` when `finish_reason` is `"content_filter"` or `"function_call"` — intentional withholding, not a string value. `content: ""` (empty string) is not a specified OpenAI response state and is treated as malformed. In both cases the implementation MUST return `None`. `CompletionResult` is only constructed when `content` is a non-empty string. Never raises on infrastructure failures — return `None` instead. The caller treats `None` as the appropriate failure mode (`guard_unavailable` or `llm_unavailable`).

`close() → None`
Releases any resources held by the LLM implementation. Idempotent — a second call has no effect. Never raises.

**MUST set:**

`name: string` — the registry key. Must be unique across all registered LLM implementations. This is also the value required in the `"type"` field of LLM configuration. Must be set as a class attribute, not an instance attribute.

**MAY override:**

`from_config(raw: dict) → LLMProvider` — constructs an instance from a raw config dict. The base implementation constructs from the base config fields. Override to handle implementation-specific fields.

`default_config() → dict` — returns the default configuration dict for this LLM implementation. Used by `write_default()` to generate a complete config file. Override to add implementation-specific fields. The base implementation returns the base fields only.

`complete_with_temperature(messages: list[dict], temperature: float, api_key: str) → CompletionResult | None` — variant called by the pipeline when a specific temperature must be applied (guard calls, translation calls). The pipeline passes the temperature value from carapex config (§15 `input_guard_temperature`, `output_guard_temperature`, `translation_temperature`) and the caller's key. The base implementation calls `complete(messages, api_key)` and ignores the `temperature` parameter — an implementation that does not override this method will silently use whatever temperature the backend was configured with, rather than the carapex-specified temperature. Override this method if the backend supports per-call temperature control; doing so is RECOMMENDED for any LLM implementation used as a guard or translator. A conformant override MUST apply the `temperature` argument to the underlying LLM call and MUST pass the `api_key` argument through. An override that ignores either parameter does not satisfy this contract.

**MUST NEVER do:**
- Raise on infrastructure failures in `complete()` — return `None` instead. A raising LLM implementation breaks the fail-closed contract. If `complete()` raises, the orchestrator cannot distinguish an LLM failure (which should produce `llm_unavailable` or `guard_unavailable` in `EvaluationResult`) from a component bug (which produces `PipelineInternalError`). The result is that an infrastructure failure surfaces as `PipelineInternalError` — the instance is treated as invalid and must be discarded. Returning `None` from `complete()` instead of raising ensures the failure is handled in-band as an `EvaluationResult`-level failure mode, and the instance remains in `ready` state. Note: if an LLM implementation raises despite this contract, the orchestrator will catch the exception and wrap it in `PipelineInternalError` — §17 describes the propagation path. This is the consequence of violating the no-raise contract, not a supported pattern.
- Call `close()` on injected dependencies — the implementation owns only what it constructs.
- Log secret values — API keys, tokens, passwords must not appear in log output.

**Registration:** Place the implementation file in `backends/`. Autodiscovery picks it up at import time. No existing files need to be edited.

**Duplicate names** raise `RuntimeError` during autodiscovery at import time, before any `build()` call is possible.

---

### Auditor

The abstraction for audit log destinations.

**MUST implement:**

`log(event: string, data: dict) → None`
Writes one audit record. Never raises — a log failure must not fail the request. If writing fails, the failure MAY be reported to stderr.

`close() → None`
Flushes and releases any held resources. Idempotent. Never raises.

**MUST set:**

`name: string` — the registry key. Must be unique across all registered auditors.

**MAY override:**

`from_config(raw: dict) → Auditor`
`default_config() → dict`

**MUST NEVER do:**
- Raise in `log()` — ever. A raising auditor breaks every `evaluate()` call.
- Block `log()` indefinitely — log writes must be async. A synchronous write that blocks on I/O or a full buffer stalls `evaluate()` for every caller. Buffer writes and flush asynchronously.
- Log the prompt content — only metadata is recorded. Full prompt text must not appear in audit records.

**Thread safety:** MUST be safe for concurrent `log()` calls. See §13.

**Registration:** Place the implementation file in `audit/`. Autodiscovered.

**Built-in auditors:**

`FileAuditor` — the default and required auditor for production deployments. Writes JSONL records to a configured file path. `build()` raises `ConfigurationError` if no auditor is configured. There is no null or discard auditor — the audit log is the only structured record of pipeline execution, and no deployment context exists in which discarding it is correct.

`InMemoryAuditor` — a test helper. Stores records in memory for direct assertion in tests. Deliberately not selectable via `type:` in configuration — it is not a valid deployment auditor. Tests MUST use this rather than a file auditor.

---

### Decoder

The abstraction for normaliser decoders.

**MUST implement:**

`decode(text: string) → string`
Applies one decoding transformation to the input. Returns the decoded text. If no transformation applies, returns the input unchanged. Never raises on content — return the input unchanged instead.

**MUST set:**

`name: string` — used to reference the decoder in normaliser configuration.

**MUST NEVER do:**
- Re-encode output. A decoder that produces output which another decoder would expand further than the original input creates a cycle. The normaliser detects and terminates cycles, but the root cause is the decoder. Violating the strictly reductive constraint is the most common cause of `normalisation_unstable` on legitimate input.
- Raise on content — return the input unchanged instead.

**Registration:** Place the implementation file in `normaliser/`. Autodiscovered.

---

### SafetyChecker

The abstraction for all pipeline checkers — both input and output.

**MUST implement:**

`inspect(text: string) → SafetyResult`
Evaluates the text and returns a result. `safe=true` to pass, `safe=false` to block with a `failure_mode`. Never raises on content or infrastructure failures — return a `SafetyResult` with `safe=false` and the appropriate `failure_mode` instead.

`close() → None`
Releases any resources the checker constructed. Idempotent. Never raises. Checkers that hold only injected LLMs MUST NOT call `close()` on them — they do not own the LLM lifecycle.

**MUST set:**

`name: string` — identifies the checker in logs and error messages.

**MAY implement (TextTransformingChecker only):**

`set_prior_result(result: SafetyResult) → None`
Receives the preceding checker's result before `inspect()` is called. Implement this if the checker needs to read state from a preceding checker. The coordinator calls this before `inspect()` on any checker that implements it.

`get_output_text() → string`
Returns the transformed working text after a `safe=true` result. The coordinator calls this after a successful `inspect()` and switches the working text for all subsequent checkers. Only implement if the checker may transform the working text.

**MUST NEVER do:**
- Raise in `inspect()` on content or infrastructure failures — return `SafetyResult` instead. If `inspect()` raises despite this contract, the orchestrator catches the exception and wraps it in `PipelineInternalError` — the instance is treated as invalid and must not be reused.
- Modify the original prompt — checkers evaluate text, they do not rewrite it for the LLM.
- Call `close()` on injected LLMs. A checker that calls `close()` on an LLM it did not construct will close a shared resource on first call — subsequent checkers or calls using the same LLM will operate on a closed resource. The failure is silent unless the LLM raises on use-after-close. There is no runtime enforcement of this constraint — it is enforced by code review. Any `close()` call on an injected dependency is a bug. See §12 for the ownership model.

**Checker order is a security decision.** The pipeline order is determined by the composition root. A new checker's position in the pipeline must be deliberately chosen — it is not arbitrary. See §11 for the ordering rationale.

**Custom failure modes.** A custom checker MAY return failure modes not in the base set defined in §7. No registration is required — a new failure mode is valid as any non-null string returned in `SafetyResult.failure_mode`. The extension MUST document any failure modes it produces. Callers who receive an undocumented failure mode will treat it as an infrastructure failure per the §7 handling contract — this is safe, but uninformative. Failure to document is the extension author's responsibility, not a spec violation.

**Registration:** Place the implementation file in `safety/`. Unlike LLM implementations and decoders, safety checkers require explicit positioning in the composition root — autodiscovery alone is insufficient because order is security-critical. The composition root must be updated to insert the checker at the correct position in the input or output pipeline. Autodiscovery cannot determine where in the security pipeline a checker belongs — that is a human security decision. A checker placed incorrectly may run after the check it was meant to gate, or before data it depends on is available. Position is not an implementation detail; it is part of the security contract. New checkers MUST be reviewed for pipeline position as part of the implementation review, not as an afterthought.


---

### ServerBackend

The abstraction for HTTP server implementations.

**MUST implement:**

`serve(handler: ChatHandler) → None`
Starts the HTTP server. Blocks until shutdown. Invokes `handler(request, headers)` for each incoming request, passing the parsed request body and the HTTP headers dict. The headers dict MUST have all keys normalised to lowercase (e.g. `"authorization"`, `"content-type"`) before being passed to the handler — the handler relies on lowercase keys for credential extraction. Never raises on individual request failures — those produce HTTP error responses. Raises on startup failure (port conflict, invalid config). Config (host, port, implementation-specific settings) is supplied at construction time — not passed to `serve()`.

`close() → None`
Stops the server and releases resources. Idempotent. Never raises.

**MUST set:**

`name: string` — the registry key. Must be unique across all registered server backends.

**MAY override:**

`from_config(raw: dict) → ServerBackend`
`default_config() → dict`

**MUST NEVER do:**
- Inspect or modify the content of requests or responses — that is carapex's responsibility.
- Raise on individual request handler failures — return HTTP 500 instead.
- Hold references to pipeline components directly — all interaction is through `ChatHandler`.

**Registration:** Place the implementation file in `server/`. Autodiscovered.

**Built-in implementations:**

`FastAPIBackend` — the default. Uses FastAPI + uvicorn. Registered under `type: fastapi`. Not production-hardened for high-concurrency deployments — replace with a production-grade backend for those use cases.

---

### Resource ownership summary

| Resource | Constructed by | Closed by | MUST NOT be closed by |
|---|---|---|---|
| Main LLM | Composition root | `Carapex.close()` | Any checker |
| Input Guard LLM | Composition root | `Carapex.close()` | Any checker |
| Output Guard LLM | Composition root | `Carapex.close()` — as a distinct instance if `output_guard_llm` is configured; the same instance as main LLM if not | Any checker |
| Translator LLM | Composition root | `Carapex.close()` — as a distinct instance if `translator_llm` is configured; the main LLM instance if not | Any checker |
| Auditor | Composition root | `Carapex.close()` | Any checker |
| Checker-owned resources | The checker itself | The checker's `close()` | Composition root directly |
| ServerBackend | Composition root | `Carapex.close()` — first, before checkers, LLMs, and auditor | Any checker or LLM |

Cross-reference §12 — resource ownership here is consistent with the interaction contracts documented there.


---

## §17 Error Propagation

### How failure modes reach `EvaluationResult`

Every failure mode originates in a component and travels to `EvaluationResult.failure_mode` via the orchestrator. The orchestrator never swallows failures — every failure produces an `EvaluationResult` with `safe=false` and a non-null `failure_mode`.

| Failure mode | Origin | Path to caller |
|---|---|---|
| `"safety_violation"` | PatternChecker, InputGuardChecker, OutputPatternChecker, OutputGuardChecker | Component returns `SafetyResult(safe=false, failure_mode="safety_violation")` → CheckerCoordinator returns on first failure → orchestrator sets `EvaluationResult.failure_mode` |
| `"entropy_exceeded"` | EntropyChecker | Same path as above |
| `"normalisation_unstable"` | Normaliser returns `NormaliserResult(stable=false)` | Orchestrator inspects `stable` before running the coordinator. Sets `EvaluationResult.failure_mode` directly. Coordinator never runs. |
| `"guard_unavailable"` | InputGuardChecker, OutputGuardChecker — guard returns `None` | Component returns `SafetyResult(safe=false, failure_mode="guard_unavailable")` → coordinator → orchestrator |
| `"guard_evaluation_corrupt"` | InputGuardChecker, OutputGuardChecker — guard returns unparseable output | Same path as `guard_unavailable` |
| `"translation_failed"` | Translator — translation call fails | Component returns `SafetyResult(safe=false, failure_mode="translation_failed")` → coordinator → orchestrator |
| `"llm_unavailable"` | Orchestrator — main LLM returns `None` | Orchestrator sets `EvaluationResult.failure_mode` directly after the LLM call. |

---

### `PipelineInternalError`

If a component raises an unexpected exception — one not covered by its contract — the orchestrator catches it, wraps it in `PipelineInternalError` carrying the component name and original exception, and propagates it to the caller. This is the only exception that `evaluate()` raises.

**Origin:** Any pipeline component — Normaliser, any checker, any LLM — that raises an exception not covered by its contract. The most likely sources are bugs in custom extension implementations.

**Path to caller:** Component raises → orchestrator catches → wraps in `PipelineInternalError` → propagates out of `evaluate()` to the caller. No `EvaluationResult` is returned. No `evaluate_complete` audit record is emitted for this call. The HTTP layer catches `PipelineInternalError` and returns HTTP 500 — it does not propagate to the HTTP caller as an unhandled exception.

**System state after propagation:** Undefined. The component that raised may have left internal state corrupt. The `Carapex` instance MUST NOT be reused after a `PipelineInternalError` — the caller should treat the instance as invalid and construct a new one via `build()`. This is distinct from all `EvaluationResult`-based failures, after which the instance remains in `ready` state and `evaluate()` may be called again.

`PipelineInternalError` indicates a bug in a component, not an operational failure. It is not represented in `EvaluationResult` — it propagates as an exception because returning an `EvaluationResult` would imply the pipeline evaluated normally, which it did not.

Callers MUST handle `PipelineInternalError` separately from `EvaluationResult`-based failures.

---

### Precondition violations — HTTP path

When `evaluate()` raises `ValueError` or `TypeError` due to a precondition violation (null messages, empty messages, no user-role entry, null content), `chat()` catches the exception and returns HTTP 400. This is a caller error — a conformant OpenAI client will never trigger this path (see §22). The response body follows the OpenAI error object shape: `{"error": {"message": "<description>", "type": "invalid_request_error", "code": "invalid_messages"}}`. No audit record is produced — the pipeline did not run. No `audit_id` was generated. This is distinct from `PipelineInternalError` (HTTP 500, component bug) — a precondition violation is unambiguously a malformed request, not a system failure.

---

### Exception hierarchy — caller-raised only

The following exceptions are never raised by `evaluate()` internally. They exist for callers who prefer exception-based control flow and choose to raise them from `EvaluationResult` values. HTTP callers receive HTTP error responses — these exceptions are relevant only to library callers.

| Exception | When to raise | Distinct from |
|---|---|---|
| `CarapexViolation` | `failure_mode` indicates a content refusal (`"safety_violation"`, `"entropy_exceeded"`) | The prompt was evaluated and refused |
| `IntegrityFailure` | `failure_mode` indicates a check component failed (`"guard_unavailable"`, `"guard_evaluation_corrupt"`, `"translation_failed"`) | The prompt was not evaluated — do not retry as a content decision |
| `NormalisationError` | `failure_mode="normalisation_unstable"` | The prompt was not evaluated — investigate decoders before retrying |

A catch block that conflates `IntegrityFailure` with `CarapexViolation` may retry a request on content when the actual problem is a downed input guard LLM. The exception types enforce the distinction at the language level.

---

### Audit log failures

Audit log write failures are absorbed by the auditor. They do not propagate to the orchestrator or the caller. A `evaluate()` call that completes successfully with a failed audit write returns a normal `EvaluationResult`. The audit failure MAY be reported to stderr.


---

## §18 Observability Contract

The audit log is the primary observability surface. Every `evaluate()` call produces a structured sequence of JSONL records linked by `audit_id`. HTTP requests map 1:1 to `evaluate()` calls — each HTTP request produces one set of audit records. The audit log is append-only and best-effort — see §4 for the full schema.

---

### Event sequence per `evaluate()` call

For a call that completes normally (input passes, LLM called, output passes):

```
carapex_init        (once per build(), not per evaluate() call)
input_normalised
llm_call            (role="translator", if translation was needed)
llm_call            (role="input_guard")
input_safety_check
llm_call            (role="main_llm")
llm_call            (role="output_guard", if enabled)
output_safety_check (if output_guard_enabled)
evaluate_complete
```

For a call blocked at normalisation:

```
input_normalised    (stable=false)
evaluate_complete        (failure_mode="normalisation_unstable")
```

For a call blocked at input:

```
input_normalised
llm_call            (role="translator", if translation was needed)
llm_call            (role="input_guard")
input_safety_check  (safe=false)
evaluate_complete
```

This sequence applies when the input guard produces the block. If a cheaper checker blocks first — EntropyChecker or PatternChecker — the pipeline exits before any LLM call. The sequence in that case is:

```
input_normalised
input_safety_check  (safe=false, failure_mode="entropy_exceeded" or "safety_violation")
evaluate_complete
```

No `llm_call` records appear — neither translator nor input guard ran. Early termination is guaranteed (§11, §23): once any checker returns `safe=false`, no subsequent stage runs.

If translation itself failed, the event sequence is:

```
input_normalised
llm_call            (role="translator", success=false)
input_safety_check  (safe=false, failure_mode="translation_failed")
evaluate_complete
```

The translator LLM was called but returned no response — `llm_call` has `success=false`. The input guard never ran — `llm_call (role="input_guard")` is absent. This mirrors the `guard_unavailable` pattern: a failed LLM call always emits a `llm_call` record with `success=false`.

For a call blocked at input due to guard unavailability (`guard_unavailable`):

```
input_normalised
llm_call            (role="translator", if translation was needed)
llm_call            (role="input_guard", success=false)
input_safety_check  (safe=false, failure_mode="guard_unavailable")
evaluate_complete
```

The guard LLM returned no response — `llm_call` has `success=false`. Token count and other fields are absent.

For a call blocked at input due to corrupt guard output (`guard_evaluation_corrupt`):

```
input_normalised
llm_call            (role="translator", if translation was needed)
llm_call            (role="input_guard", success=true)
input_safety_check  (safe=false, failure_mode="guard_evaluation_corrupt")
evaluate_complete
```

The guard LLM responded but its output was unparseable or missing the `"safe"` key — `llm_call` has `success=true` (the call completed) but the result was unusable. This distinction matters: `guard_unavailable` and `guard_evaluation_corrupt` are both rejections, but one indicates a network/availability problem and the other is a potential security signal (the guard may have been manipulated into producing malformed output).

For a call where the main LLM was unreachable:

```
input_normalised
llm_call            (role="translator", if translation was needed)
llm_call            (role="input_guard")
input_safety_check
llm_call            (role="main_llm", success=false)
evaluate_complete        (failure_mode="llm_unavailable")
```

The output pipeline does not run — no `output_safety_check` record is emitted.

For a call blocked at output:

```
input_normalised
llm_call            (role="translator", if translation was needed)
llm_call            (role="input_guard")
input_safety_check
llm_call            (role="main_llm")
llm_call            (role="output_guard", if enabled)
output_safety_check (safe=false)
evaluate_complete
```

`evaluate_complete` is always the final record for a call. Its absence in the audit log for a given `audit_id` indicates one of two conditions: the process was killed mid-run, or `evaluate()` raised `PipelineInternalError`. Both produce an incomplete record set with no `evaluate_complete`. A consumer cannot distinguish these two conditions from the audit log alone — `PipelineInternalError` propagates as an exception to the caller, not as an audit record. See §17.

`carapex_init` is emitted synchronously as the final act of `build()`, before `build()` returns. It is guaranteed to appear in the audit log before any `evaluate()` record from the same instance. This guarantee holds because `build()` is called from a single thread before any concurrent `evaluate()` calls begin (§22) — no `evaluate()` call can produce audit records until `build()` has returned, by which time `carapex_init` has already been written.

---

### Emission timing

Every event is emitted after the operation it describes completes — not before. A record in the log means the operation finished. The exception is `evaluate_complete` — it is emitted as the final act of `evaluate()`, after all other operations.

This matters for crash recovery: if the process dies between `llm_call` (role="main_llm") and `output_safety_check`, the audit log contains the main LLM call record but not `output_safety_check` or `evaluate_complete`. The incomplete record set is the crash signal.

---

### Token counts

Each `llm_call` event carries its own token counts — `prompt_tokens` and `completion_tokens` — for that specific LLM interaction. Token counts are self-contained per call and are not carried forward to subsequent events. Fields are absent if the call failed before token data was available. Field names match the OpenAI API — `prompt_tokens` for input, `completion_tokens` for output.

---

### `audit_id`

A short string generated once per `evaluate()` call. Unique within a deployment for practical purposes — not guaranteed globally unique. All records from a single `evaluate()` share the same `audit_id`. Use `audit_id` to reconstruct the full record set for a call, not line order — concurrent calls interleave records in the log.

---

### What is not logged

The prompt text is never written to the audit log. Response content is never written to the audit log. Delimiter values used in input guard calls are never logged. API keys and credentials MUST NOT be logged. This is a hard constraint — not a recommendation. An auditor implementation that logs credential values violates this contract regardless of how the field is named or encoded. Only metadata — lengths, flags, failure modes, token counts — is recorded.

This is a deliberate constraint. The audit log is designed to be safe to ship to external observability systems without exposing prompt content or credentials.

---

### Schema versioning

`carapex_init` carries a `version` field (string, semver format, always present). This is the only record that carries a version. All records emitted by the same running instance conform to the schema for that version. Consumers who need to handle records from multiple carapex versions use the `version` field from `carapex_init` to determine which schema applies — not field presence inference.

Individual `evaluate_complete`, `input_safety_check`, and other per-call records do not carry a `version` field. They share the schema declared by the `carapex_init` record from the same process lifetime.

A custom `Auditor` MAY emit additional event types not in the base set. Consumers of the audit log MUST ignore unknown event types — see §4. A custom `Auditor` that emits the same event names as the base set MUST preserve the common field schema (`event`, `audit_id`, `timestamp`). Adding fields to a base event is permitted. Removing or renaming fields is a breaking change.


---

## §19 Security Properties

### What carapex guarantees

**Fail closed on all infrastructure failures.** If the input guard is unavailable, returns a corrupt response, or produces a response missing the `"safe"` key, the request is rejected. There is no condition under which an unevaluated prompt produces `safe=true`. A caller who receives `safe=true` in `EvaluationResult` can rely on the fact that all configured checks ran and passed.

**Only boolean `true` is accepted as a safe guard result.** Type coercion — `"true"` (string), `1` (integer), any truthy value — does not pass. This prevents bypass via JSON type ambiguity.

**Per-call random delimiters.** Each input and output guard call wraps evaluated content in unique 64-character hex delimiter tokens generated from a cryptographically random source. Each hex character encodes 4 bits, so 64 characters carry 256 bits of entropy. At 256 bits, the probability of an attacker correctly guessing a delimiter within a single call is negligible. An attacker cannot construct input that closes the delimiter boundary early without knowing the delimiter value in advance. Delimiter values are not logged.

**The input guard is stateless when a separate model is configured.** Each prompt is evaluated from scratch with no knowledge of prior calls. Grooming attacks — which build up shared context across turns to shift a model's framing — are not effective against a dedicated input guard model. The input guard cannot be primed. This guarantee does not hold when `input_guard_llm` is null and the main LLM serves as the input guard — the main LLM may be influenced by the conversation history the caller passed in the messages list. In that configuration, a grooming attack that shapes the main LLM's context may affect guard evaluation. See §15 for the single-model deployment tradeoff.

**Guards receive the last user message from the messages list.** This is structural — `evaluate()` extracts the last user message and passes only that to guards. No prior turns are visible to the guard. This eliminates grooming of the guard itself. It does not prevent grooming of the main LLM via the messages list — that is a documented limitation (§1, §19 "What carapex does not guarantee").

**Normalisation precedes evaluation.** All checks run on decoded content. An attacker cannot bypass pattern or entropy checks by encoding their payload — the normaliser decodes it before evaluation reaches those checks.

**Translation before input guard evaluation.** Non-English prompts are translated to English before the input guard sees them. The input guard always evaluates English. This eliminates the attack surface of low-resource language misclassification. The enforcement mechanisms are in the Translator contract (§5): a translator response that is byte-for-byte identical to the input is treated as a failed translation (`translation_failed`), and a translator response that is an empty string when translation was required is also treated as `translation_failed` — an empty translation is not a valid English rendering and would cause the guard to evaluate nothing rather than the content. Translation quality beyond these detectable failures is not guaranteed — a translator LLM that produces plausible-looking but incorrect English is outside the threat model.

**The original prompt is never modified.** The LLM receives exactly what the caller sent. Normalisation and translation are evaluation tools only.

---

### What carapex does not guarantee

**Semantic manipulation is probabilistic.** The input guard reduces the risk of semantic attacks — framing, hypotheticals, narrative wrapping. It does not eliminate it. A well-crafted prompt that distributes harmful intent across innocent-looking structure may pass. LLM-based classification is not deterministic and can be fooled by distribution-shifted inputs.

**Static defences degrade.** The pattern checker and entropy threshold are fixed at deployment. An attacker who can probe the system can iterate bypass attempts. A static system provides diminishing returns against an active adversary specifically targeting it. Pattern set review, input guard model currency, and threshold tuning are operational concerns — not correctness contracts. Guidance on maintaining defence effectiveness over time is in the Deployment Guidance section of this spec.

**Multi-turn grooming of the main LLM is not prevented.** The guard evaluates each prompt in isolation. If an attacker has shaped the main LLM's context across multiple apparently legitimate turns, the input guard may not detect this — it sees only the current prompt, not the accumulated context.

**No cross-session visibility.** carapex operates per-call. Distributed attacks that fragment intent across sessions or accounts are invisible at the per-call level.

---

### Custom input guard prompts and LLM implementations

The security guarantees above apply to the built-in input guard system prompt and the built-in input guard model. Operators who supply a custom prompt via `input_guard_system_prompt_path` or `output_guard_system_prompt_path` take full responsibility for the security properties of that prompt. The guarantees above do not automatically transfer to custom prompts. In particular: a custom prompt that does not instruct the guard to fail closed, or that can be manipulated to abandon its output format, weakens or eliminates the guarantees above.

The same applies to `output_guard_llm`. Substituting a different model for output guard evaluation is a security operation — not a routine configuration change. A model that was not trained or evaluated for safety classification may produce unreliable `safe` values, fail to detect injection responses, or be more susceptible to manipulation via the response content it is evaluating. Operators supplying a custom output guard LLM take full responsibility for verifying that the model satisfies the fail-closed contract in §5 (OutputGuardChecker).

---

### Multi-caller isolation

carapex operates per-call with no shared mutable state between concurrent calls. One caller's prompt does not affect another caller's evaluation result. One caller cannot exhaust per-call resources (delimiter generation, working text) for another caller.

Shared resources — LLMs, auditor — are shared across concurrent calls. A caller who sends a very large prompt or triggers a slow input guard response does not block other callers, but does consume LLM capacity for the duration of that call. carapex does not enforce per-caller resource quotas. Resource exhaustion protection is the responsibility of the deployment infrastructure.

---

### Fail-closed rationale

Failing open on input guard unavailability would mean requests pass with only pattern and entropy checking while `EvaluationResult.safe` reports `true`. The caller has no signal that the input guard did not run. This is a false guarantee — the system claims to have checked something it did not check. carapex does not provide false guarantees. Every `safe=true` result means all configured checks ran and passed.


---

## §20 Versioning and Evolution

### Stability levels

| Interface | Stability | Notes |
|---|---|---|
| `evaluate(messages: list[dict])` signature | Stable | Breaking change requires major version increment |
| `EvaluationResult` fields | Stable | New fields MAY be added. Removing or renaming fields is a breaking change. |
| `build(config)` signature | Stable | |
| `CarapexConfig` fields | Evolving | New fields MAY be added with minor version increment. Removing fields is a breaking change. Field semantics MAY change with documented notice. |
| `output_guard_llm` config field | Evolving | Adding this field is non-breaking. Removing it or changing its fallback semantics (null → main LLM) is a breaking change. Given its security significance, changes to its resolution logic MUST be documented explicitly, not subsumed into a general config changelog entry. |
| Failure mode base set | Stable | New failure modes from extensions are not breaking — see §7. Removing a base failure mode is a breaking change. |
| Audit log base event set | Evolving | New events and new fields on existing events MAY be added. Removing events or fields is a breaking change. |
| Extension abstractions (`LLMProvider`, `SafetyChecker`, `Auditor`) | Evolving | New MUST-implement operations are a breaking change for existing extensions. New MAY-override operations are not. The `api_key` parameter added to `complete()` and `complete_with_temperature()` in v11 is a breaking change for existing `LLMProvider` implementations. |
| `Decoder` extension abstraction | Evolving | New MUST-implement operations are a breaking change for existing decoders. New MAY-override operations are not. The strictly reductive constraint is Stable — any relaxation is a breaking change to the security model, not merely an API change. |
| `ServerBackend` extension abstraction | Evolving | New MUST-implement operations are a breaking change for existing server backends. New MAY-override operations are not. |
| `TextTransformingChecker` contract (`set_prior_result`, `get_output_text`) | Evolving | New MUST-implement operations on this sub-interface are a breaking change for existing `TextTransformingChecker` implementations. |
| Built-in input guard system prompt | Evolving | MAY change between versions. Not a stable contract — callers who depend on specific guard behaviour MUST supply their own prompt via config. |
| `ChatCompletionRequest` / `ChatCompletionResponse` shapes | Follows OpenAI Chat Completions API versioning | carapex passes unknown request fields through unchanged. The synthetic blocked response shape (§4) is Stable — `finish_reason: "content_filter"`, HTTP 200, `usage` zeroed. Changes to the OpenAI spec that affect fields carapex passes through are not breaking changes to carapex. |
| Exception hierarchy | Stable | New exception types MAY be added. Removing or reclassifying existing types is a breaking change. |

---

### Breaking changes

A breaking change is one that requires callers or extension authors to modify their code to maintain correct behaviour. Breaking changes require a major version increment and MUST be documented in the changelog with the previous contract and the migration path.

---

### Spec maintenance

This spec is a versioned document. Each substantive change MUST be recorded: what changed, why, and what the previous contract was.

Current version: 0.13.

---

### Data format versioning

The `EvaluationResult` schema and the audit log JSONL format are versioned contracts subject to the stability levels above. See §4 for the full schemas. The `carapex_init` audit record carries a `version` field (semver string) that declares the schema version for all records emitted by that instance. Per-call records do not carry a version field — consumers use the `carapex_init` version to determine the applicable schema.


---

## §21 What Is Not Specified

The following are intentionally left to implementers. Any approach that satisfies the contracts above is conformant.

**Internal data structures.** How `SafetyResult`, `NormaliserResult`, `ScriptResult` are represented internally. They are pipeline-internal types — their shape is an implementation detail as long as the component contracts in §5 are satisfied.

**Decoder algorithm implementations.** How whitespace collapse, unicode escape decoding, HTML entity decoding, URL decoding, base64 decoding, and homoglyph substitution are implemented, provided they are strictly reductive and produce the correct canonical output.

**Entropy computation implementation.** Any correct Shannon entropy computation over the character distribution of the input string is conformant.

**Language detection implementation.** How language detection is performed internally. The contract is the output — `detected_language` and `translation_needed` — not the algorithm or library used.

**Guard response parsing.** How the guard's JSON response is parsed, provided the contract in §5 (InputGuardChecker) is satisfied — only boolean `true` passes, all other values and parse failures produce `safe=false`.

**Logging verbosity and format.** Internal logging (not the audit log) — levels, message format, handler configuration. The audit log schema in §4 is specified. Internal diagnostic logging is not.

**Retry strategies.** Whether and how LLM implementations retry failed calls internally. The contract is the return value — `None` on failure — not how many attempts were made.

**Concurrency model.** Whether LLM implementations use connection pools, thread pools, async I/O, or any other concurrency mechanism internally. Whether the server backend uses async or threaded request handling — only the contract (blocking `serve()`, handler invoked per request) is specified. The thread-safety contracts in §13 are specified. The implementation is not.

**Memory layout and object lifecycle.** How objects are allocated, pooled, or garbage collected.

**Config file format.** Whether configuration is supplied as JSON, YAML, TOML, or another format. The `CarapexConfig` field contract in §15 is specified. The serialisation format is not — `load()` implementations MAY support any format.

**Graceful drain on shutdown.** Shutdown is initiated by SIGTERM. Graceful drain of in-flight `evaluate()` calls before component teardown is not implemented. A SIGTERM mid-evaluation may leave an incomplete audit trail — `evaluate_complete` will be absent for any call interrupted by shutdown. Completing in-flight calls before closing components is tech debt. Until implemented, operators MUST accept that shutdown under load produces incomplete audit records for the interrupted calls.


---

## §22 Assumptions

### Environmental assumptions

**Character encoding.** All string inputs are UTF-8. The entropy computation, pattern matching, and normaliser decoders operate on Unicode code points. Platform default encoding is never assumed — all file I/O specifies UTF-8 explicitly.

**Filesystem atomicity.** Writes to the audit log file are atomic at the line level on a single writer. The built-in file auditor does not guarantee atomic writes under concurrent access from multiple processes writing to the same file — only within a single process. If multiple processes share an audit log file, records may interleave or corrupt. Violation is detectable as malformed JSON on a line.

**Clock.** The system clock is approximately correct and moves forward. Audit log timestamps are for human readability and correlation, not for strict ordering. Record ordering within a file is not guaranteed to match timestamp order under concurrent access.

**In-process library availability.** The language detection library and all other in-process dependencies are installed and available at startup. Missing dependencies fail at startup with a clear error — they do not fail silently at first use.

---

### Caller assumptions

**The messages list always contains at least one entry with `role == "user"`.** This is a requirement of the OpenAI Chat Completions API contract — any conformant client will satisfy it. Via the HTTP interface, a request without a user message is a malformed API call that violates the OpenAI spec before reaching carapex. Via the library interface, a messages list with no user entry is a programming error in the caller. carapex raises a programming error on null or empty lists; a non-empty list with no user entry is treated the same way — a programming error, not a runtime safety condition. carapex does not define a failure mode for this case.

**`build()` is called from a single thread before any concurrent `evaluate()` calls begin.** Construction is not concurrent. Registry reads, component construction, and any process-level library initialisation all occur before any `evaluate()` call or `serve()` call. If `build()` is called concurrently with another `build()` or with an active `evaluate()`, behaviour is undefined.

**`close()` is called after all in-flight `evaluate()` calls complete and after `serve()` has returned.** The caller is responsible for draining concurrent calls before closing and for initiating server shutdown before calling `close()`. carapex does not track in-flight calls or block `close()` until they complete.

**The caller does not call `evaluate()` after `close()`.** The Carapex instance is not reusable after `close()`. Calling `evaluate()` on a closed instance produces undefined behaviour.

**The caller handles `PipelineInternalError`.** This indicates a component bug. The caller must not retry the same request without investigation — the error will recur.

**The caller treats unknown `failure_mode` values as infrastructure failures.** See §7. A caller who ignores unknown failure modes and passes the request through creates a security gap.

**carapex does not model caller intent or enforce caller behaviour.** Input may be adversarial or legitimate. Call volume may be low or extreme. carapex evaluates each call in isolation — it has no visibility into patterns across calls, sessions, or callers. Protection against abusive call patterns, API misuse, or distributed load is the responsibility of the deployment infrastructure.

---

### Operational assumptions

**The input guard LLM responds within a bounded time.** carapex does not enforce a timeout on guard calls internally — see §23. If the guard hangs indefinitely, `evaluate()` hangs. Timeout enforcement is the responsibility of the LLM implementation or the deployment infrastructure.

**The deployment environment does not share the audit log file across multiple carapex processes** unless the operator has verified that concurrent writes are safe at the filesystem level. Violation produces interleaved or corrupt records — detectable as malformed JSON.

**carapex makes no assumption about call rate.** It applies no rate limiting or backpressure. Under sustained load, LLMs become the bottleneck and `evaluate()` will block until the LLM responds. The caller or deployment infrastructure is responsible for managing concurrency and call volume.

**Configuration files are readable at startup.** A missing or unreadable config file raises `ConfigurationError` at `build()` time. Config files are not re-read at runtime.

**The language detection implementation uses a fixed seed, producing deterministic results per call.** The seed does not change after construction — the same input always produces the same `translation_needed` routing decision. A random seed would mean identical prompts might route differently on different calls, producing inconsistent input guard evaluation. If the language detection library changes its internal algorithm in a future release, detection results may change for the same seed. This is an upstream dependency assumption — not enforced by carapex.

---

### Assumptions whose violation is undetectable

**The input guard has not been fine-tuned to ignore its system prompt.** carapex cannot verify that the guard model respects its evaluation instructions. An input guard model that has been compromised or fine-tuned to always return `safe=true` will pass all prompts. This is outside the threat model.


---

## §23 Performance Contracts

### What is not a performance characteristic — it is a correctness contract

**Translation determinism.** Translation MUST produce the same English output for the same input on every call. This is a correctness contract — not a performance goal. The input guard evaluates the translated text. If translation is non-deterministic, the input guard may evaluate different English representations of the same prompt on different calls, producing inconsistent safety decisions. Translation temperature is set to 0.0 to enforce this. See §15.

**Input guard evaluation non-determinism is bounded.** The input guard operates at temperature 0.1. For unambiguous inputs, the result MUST be consistent across calls. Variation is permitted only on genuinely ambiguous edge cases where the correct classification is uncertain. A caller MUST NOT rely on variation as a mechanism to retry a blocked prompt to a pass — the system is designed so that identical prompts produce identical results in all but edge cases.

**Early termination is guaranteed.** If any pipeline stage returns `safe=false`, subsequent stages MUST NOT run. This is a correctness contract — a caller who instruments the pipeline and observes downstream stages running after an upstream failure has observed a defect, not a performance issue.

**`evaluate_complete` is always the final audit record.** If `evaluate_complete` is present in the audit log for a given `audit_id`, the call completed. This is a correctness contract for audit consumers reconstructing call outcomes.

---

### Genuine performance characteristics — not specified

The following are implementation choices. No values are mandated.

- Latency of individual pipeline stages
- Throughput under concurrent load
- Memory usage per call
- Backend connection pool sizing
- Retry counts and backoff intervals within LLM implementations
- Normaliser pass duration

These are deployment and tuning concerns. They do not affect the correctness contracts above.

---

### Timeout

carapex does not enforce timeouts on LLM calls internally. If an LLM call hangs, `evaluate()` hangs. This is a deliberate deferral — timeout policy is deployment-specific and depends on the latency characteristics of the configured LLMs. Timeout enforcement belongs in the LLM implementation or in the deployment infrastructure wrapping `evaluate()`.

This is stated explicitly so callers know the absence of a timeout contract is intentional, not an omission. A caller who requires bounded `evaluate()` latency MUST implement timeout enforcement externally.

---

## §24 Future Directions

This section documents known extension directions for carapex. Nothing here is a current contract. It exists so that future design decisions are made with awareness of intended extensibility, and so that current architectural choices — particularly in §15 and §16 — can be understood in the context of what they are designed to accommodate.

---

### Policy-driven input guards

The current input guard model is a single LLM configured by a static system prompt. The architecture already supports replacement: both the input guard and output guard resolve through the LLM registry, and their system prompts are operator-supplied files. The natural evolution is a policy engine that selects input guard behaviour at call time based on caller context, topic, or risk tier — rather than applying a single fixed policy to every request.

A policy-driven input guard would satisfy the `LLMProvider` contract in §16. No pipeline changes would be required. The policy resolution — which system prompt applies, which model to use, what temperature to set — would be internal to the LLM implementation.

---

### Context-aware evaluation

The input guard currently evaluates each prompt in isolation. It has no knowledge of the conversation history, the caller's session, or patterns across calls. Grooming attacks that distribute intent across multiple turns are invisible to it — this is a documented limitation in §19.

Context-aware evaluation would extend what the input guard receives to include a sanitised representation of prior turns. The `evaluate()` interface is a natural extension point — an optional `context` parameter carrying prior call metadata would allow the input guard to evaluate the current prompt in light of prior behaviour without requiring session-level state in the pipeline.

This requires decisions about what context to include, how to prevent context injection, and what the performance implications of larger input guard inputs are. Those are design questions, not current contracts.

---

### RAG-augmented input guards

A RAG-augmented input guard — pulling from a curated knowledge base of known attack patterns, policy documents, or domain-specific constraints — can reason about novel attack variants and domain-specific constraints that a static system prompt cannot encode. This is a LLM-level extension: the RAG lookup is internal to the input guard LLM implementation. The pipeline contract is unchanged.

---

### Separate output guard model

The current spec supports separate `output_guard_llm` configuration — see §15. The intent is that input and output evaluation can run on different models with different training and different system prompts. Input evaluation is primarily about intent; output evaluation is about content, compliance, and injection response detection. These are different classification tasks that may benefit from specialisation.

This is already supported at the configuration level. The future direction is first-class documentation of what properties a good output guard model should have, distinct from an input guard model.

---

### Per-call input guard selection

A further extension of policy-driven input guards: selecting not just the guard configuration but the guard model itself at call time. High-stakes requests route to a more capable (and more expensive) input guard; routine requests to a faster one. This requires a routing layer ahead of the input guard LLM — an LLM implementation that itself dispatches to one of several underlying models.

The `LLMProvider` contract supports this: the routing logic is internal to the LLM implementation. The pipeline sees a single LLM call.

---

### Contextual output guard

The current output guard evaluates the LLM response in isolation. It has no knowledge of the original prompt. This limits its ability to reason about whether the response is appropriate *for that input* — a compliant response to a refused prompt, or a response that reveals something the prompt was designed to extract, is invisible to a guard that sees only the response text.

The natural extension is to pass both the original prompt and the LLM response to the output guard, enabling it to evaluate the pair rather than the response alone.

The complication: the main LLM is stateful. A short prompt like "proceed" is meaningful only in the context of the conversation history the main LLM holds — context the output guard does not have and cannot safely receive. Passing the original prompt without that context may produce noise rather than signal for stateful interactions.

This suggests two possible directions: a contextual output guard that receives `(prompt, response)` for stateless or single-turn deployments; or a stateful output checker extension point that can receive sanitised conversation context alongside the response. The latter intersects with the context-aware evaluation direction above and carries the same design questions around context injection and performance.

Neither direction changes the current `OutputGuardChecker` contract. Both are LLM-level extensions — the pipeline interface is unchanged.

---

### Stateful input guard with rolling attention window

The input guard is stateless — each prompt is evaluated independently. A stateful guard would maintain a bounded window of prior turns, enabling detection of grooming attacks that distribute intent across multiple exchanges.

The rolling window is the critical design constraint. LLM instruction-following degrades as context length grows — a guard receiving a very long context may fail to apply its own system prompt reliably. The window size therefore bounds both the attack surface visible to the guard and the guard's own evaluation quality. Longer windows see more of the attack; shorter windows keep the guard sharp. The right tradeoff depends on the deployment context.

This intersects with the context-aware evaluation direction above. The distinction: context-aware evaluation passes sanitised prior turn metadata; a stateful guard maintains a sliding window of actual content. The latter is more powerful and carries more risk — injecting adversarial content into the window is a new attack surface that must be addressed.

---

### Confidence scoring

The current evaluation contract is binary: `safe=true` or `safe=false`. This is deliberate — fail-closed semantics require a hard boundary. The binary gate is the right default.

Some deployments need graduated responses. A prompt that scores at the edge of the safety boundary may warrant human review rather than hard rejection. A high-confidence pass is different from a marginal one for risk management purposes.

Confidence scoring would expose a numeric signal alongside the binary result. The binary result remains the authoritative safety decision — confidence is additive, not a replacement. Applications that need graduated handling can use it; applications that don't can ignore it. The fail-closed contract is unchanged: a low-confidence `safe=true` is still a pass, not a rejection.

---

### Adaptive defence

Pattern sets and entropy thresholds are static after deployment. An adversary who can probe the system will eventually find bypass paths. Static defences provide diminishing returns against a targeted, iterative attacker.

Adaptive defence would introduce adversarial testing as an operational loop: fuzzed prompt generation, mutation-based attack discovery, and defence drift detection against the deployed configuration. The output is not runtime behaviour change — it is data for human review: which patterns are being bypassed, which thresholds are drifting, what the guard model is failing to catch. Pattern updates and model replacement remain deliberate operator decisions, not automatic responses.

This is an operational capability layered on top of carapex, not a pipeline extension. The audit log is the primary data source.

---

### Specialised guard models

The input guard currently uses a general-purpose LLM. General models are not trained specifically against jailbreak taxonomies, injection signatures, or obfuscation patterns. A model fine-tuned on adversarial prompt corpora may outperform a general model on safety classification — lower latency, stronger detection, more consistent output format.

This is already architecturally supported: `LLMProvider` is swappable. The future direction is first-class guidance on what properties a good specialised guard model must have — training data requirements, output format stability, behaviour under adversarial input — so that operators can evaluate candidate models against a defined standard rather than by intuition.

---

### Streaming evaluation

The current design requires the complete LLM response before output checks run. This is a correctness requirement: the output guard and output pattern checker need the full response to evaluate it reliably. It is also a documented limitation — see §1.

Streaming evaluation would allow partial safety checking during generation, enabling early termination of unsafe responses. The fundamental tension is that semantic evaluation on a partial response is unreliable — a response that looks unsafe at token 50 may be safe at completion, and vice versa. Any streaming approach must define a contract for what "safe so far" means and what the false positive and false negative implications are.

This is not an extension of the current pipeline — it requires a different safety model for output evaluation. The current `OutputGuardChecker` contract assumes complete input.

---

### Scope extension: LLM security gateway

The directions above extend carapex within its current scope — prompt execution safety. A broader direction is scope expansion: carapex as a control plane for LLM application security, integrating RAG governance, retrieval filtering, enterprise policy enforcement, and ecosystem connectors.

This would position carapex differently relative to tools like Guardrails AI and NVIDIA NeMo Guardrails — not as a focused safety layer but as a security gateway. The `LLMProvider`, `SafetyChecker`, and `Auditor` abstractions are designed for composability and would support this extension without architectural rewrite.

Whether to pursue this direction is a product decision, not an architecture one. The current design does not foreclose it.

---

### Notes for future design

These directions share a common constraint: they MUST preserve the fail-closed contract in §19. Any extension that introduces a code path where an unevaluated prompt produces `safe=true` breaks the primary security guarantee. Future designs MUST be reviewed against this constraint first.


---

## Deployment Guidance

This section covers operational concerns — what you need to do to keep carapex's security guarantees meaningful over time. Nothing here is a correctness contract. These are the practices that separate a system that is secure on day one from one that remains secure at day 90.

---

### Deployment patterns

**Docker container.** carapex runs as a standalone container. The operator mounts a config file and exposes the server port. The application container points its LLM client at carapex's address.

```
[application container] → carapex:8000 → [LLM API]
```

**Kubernetes sidecar.** carapex runs as a sidecar container in the same pod as the application. The application calls `localhost:8000`. No network hop between application and carapex. The API key never leaves the pod.

```yaml
containers:
  - name: app
    # calls localhost:8000
  - name: carapex
    image: carapex:latest
    ports:
      - containerPort: 8000
```

The sidecar pattern is the strongest deployment posture. carapex and the application share a network namespace — the application's API key travels only over loopback. The operator controls the carapex config directly in the pod spec.

---

### Key trust model

carapex uses pass-through authentication — the caller's `Authorization` header is extracted by `ChatHandler` and passed as `api_key` to every LLM call via `complete()` and `complete_with_temperature()`. carapex never stores or logs the key. In Docker and sidecar deployments the key never leaves the caller's infrastructure.


---

### Verify the deployment is evaluating

Before accepting traffic, confirm that carapex is actually running and evaluating every prompt. The `carapex_init` audit record is emitted once when `build()` completes. Its presence in the audit log confirms the instance initialised successfully. Its absence, or the absence of `input_safety_check` records for known requests, indicates that carapex is not in the call path.

Do not rely on application-level logging to confirm evaluation. Verify through the audit log directly. A misconfigured deployment that bypasses carapex entirely will not surface any error — it will simply be absent from the audit log.

---

### Wire the audit log into your observability stack

The audit log is the only structured record of pipeline execution. Left as a flat file it has limited operational value. Operators SHOULD ship audit records to their existing observability infrastructure — log aggregation, SIEM, or analytics platform.

At minimum, monitor for:

- `failure_mode="guard_unavailable"` — indicates the input guard LLM is unreachable. If sustained, input semantic evaluation has stopped.
- `failure_mode="llm_unavailable"` — main LLM is unreachable. Requests are being rejected rather than served.
- `safe=false` volume — sudden spikes may indicate a coordinated attack or a misconfigured pattern set generating false positives.
- Missing `evaluate_complete` records — indicates `evaluate()` calls that did not complete. A process killed mid-run leaves an incomplete audit trail.

Anomaly detection on these signals does not require carapex-specific tooling — standard log alerting is sufficient.

---

### Pattern set review cadence

The input pattern checker is static after deployment. Attack techniques evolve. A pattern set that provides strong coverage at launch will drift from the threat landscape over time.

Establish a review cadence before deployment, not after an incident. At minimum:

- Review the pattern set when a new attack technique is publicly disclosed that targets LLM applications.
- Review after any confirmed bypass — a prompt that passed pattern checking and should not have.
- Scheduled review at a cadence appropriate to your threat model. High-value deployments warrant more frequent review than internal tooling.

Pattern updates require redeployment. Treat them as security operations, not routine config changes. Test updated patterns against your known-good traffic before deploying to production — overly broad patterns increase false positives.

---

### Input guard model currency

The input guard LLM is your primary semantic defence. LLMs age — a model that classifies reliably at deployment may weaken against newer attack variants as the threat landscape evolves.

Track when your input guard model was last evaluated against current attack techniques. When you update the model:

- Evaluate the candidate model against your known attack corpus before deploying.
- Verify output format stability — the guard must return parseable JSON with a boolean `safe` field. A model update that changes output format will cause `guard_evaluation_corrupt` failures.
- Update the `input_guard_system_prompt_path` if the new model requires different instruction format.
- Treat model replacement as a security operation requiring explicit sign-off, not a routine dependency update.

---

### Threshold tuning

The entropy threshold default (5.8 bits/character) is chosen for general applicability. It may not be correct for your application's traffic profile.

Before deploying to production, measure the entropy distribution of your legitimate traffic. If your application regularly handles high-entropy content — cryptographic material, base64-encoded data, dense code — the default threshold may produce unacceptable false positive rates. If your application handles only natural language, you may be able to tighten the threshold.

Threshold changes are security-relevant. Raising the threshold reduces detection coverage. Lowering it increases false positives. Document the rationale for any deviation from the default. Re-evaluate if your traffic profile changes significantly.

---

### Config change process

Security-relevant config fields — `input_guard_system_prompt_path`, `output_guard_system_prompt_path`, `output_guard_enabled`, `entropy_threshold` — affect the security posture of the deployment directly. They MUST NOT be changed through the same process as routine operational config.

Treat changes to these fields as security operations:

- Changes require explicit justification documented before deployment.
- Changes should be reviewed by someone other than the author.
- The previous config should be retained until the new config is confirmed operational.
- Audit log monitoring should be heightened after a security config change to catch unexpected behaviour.

---

### Multi-instance deployments

Horizontal scaling produces multiple carapex instances, each with its own audit log. The `instance_id` field is present on every audit record — including `carapex_init` and all per-call records. The `audit_id` field in per-call records is unique per call within an instance but is not globally unique across instances.

When aggregating audit logs from multiple instances, use `(instance_id, audit_id)` as the composite key for correlating records from a single call. Do not use `audit_id` alone across instances.

Ensure all instances are running the same configuration. A deployment where some instances have `output_guard_enabled: false` and others do not will produce inconsistent safety evaluation across requests — load balancing will route some requests through a weaker configuration transparently.

---

### Incident response

If you observe a confirmed bypass — a prompt that produced `safe=true` and should not have:

1. Retrieve the full audit trail for the call using `audit_id`. Confirm which checks ran and what they returned.
2. Determine the bypass mechanism: pattern evasion, entropy below threshold, semantic manipulation of the guard, or infrastructure failure.
3. If pattern evasion: add a pattern, test against known-good traffic, redeploy.
4. If semantic manipulation: evaluate whether a guard model update or system prompt change is warranted. Do not assume the bypass is reproducible without testing.
5. If infrastructure failure: the audit log will show `guard_unavailable` or `guard_evaluation_corrupt`. Treat this as an operational outage, not a safety failure — the request was rejected, not passed unsafely.

carapex's fail-closed design means infrastructure failures are rejections, not bypasses. A `guard_unavailable` failure is not a security incident — it is an availability incident. Distinguish the two in your incident response process.

---

### Defence-in-depth

carapex is one layer. It targets a specific gap: structured, auditable safety evaluation at the prompt boundary. It does not replace:

- Application-level input validation before carapex receives the prompt.
- Output sanitisation in the application layer after carapex returns a response.
- Rate limiting and abuse detection at the API gateway level.
- Access control and authentication upstream of carapex.
- Monitoring of the main LLM's outputs for content policy violations outside the carapex pipeline.

A deployment that relies solely on carapex for LLM application security has a single layer where multiple are warranted. The guarantees in §19 are strongest when carapex operates within a defence-in-depth stack, not as the only defence.
