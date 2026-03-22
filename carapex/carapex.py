"""
Carapex — the top-level orchestrator and composition root.

This is the only file that knows about concrete classes. All components
below it depend only on abstractions.

Public API:
  build(config)       → Carapex   construct a fully initialised instance
  carapex.evaluate(messages)      library interface
  carapex.chat(request, headers)  HTTP handler (called by ServerBackend)
  carapex.serve()                 start the HTTP server (blocks)
  carapex.close()                 release all resources

Pipeline execution order (§11):
  messages
    → extract last user message
    → Normaliser
    → EntropyChecker
    → PatternChecker
    → ScriptChecker → Translator (TextTransformingChecker handoff)
    → InputGuardChecker
    → Main LLM (receives original messages list, unmodified)
    → OutputPatternChecker
    → OutputGuardChecker (if enabled)
    → EvaluationResult
"""

from __future__ import annotations

import logging
import secrets
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from carapex.audit.base import Auditor
from carapex.core.config import CarapexConfig
from carapex.core.exceptions import ConfigurationError, PipelineInternalError
from carapex.core.registry import get_auditor, get_decoder, get_llm, get_server, all_decoder_names
from carapex.core.types import EvaluationResult, UsageResult
from carapex.llm.base import LLMProvider
from carapex.normaliser.base import Normaliser
from carapex.safety.coordinator import CheckerCoordinator
from carapex.safety.entropy import EntropyChecker
from carapex.safety.guard import InputGuardChecker, OutputGuardChecker
from carapex.safety.pattern import OutputPatternChecker, PatternChecker
from carapex.safety.script import ScriptChecker
from carapex.safety.translator import Translator
from carapex.server.base import ServerBackend

log = logging.getLogger(__name__)

# Failure modes that are infrastructure failures (not content refusals)
_INFRA_FAILURE_MODES = frozenset({
    "guard_unavailable",
    "guard_evaluation_corrupt",
    "translation_failed",
    "llm_unavailable",
    "normalisation_unstable",
    "no_user_message",
})

_BLOCKED_CONTENT = "I'm unable to process this request."


class Carapex:
    """Orchestrates the full safety pipeline. Owns all component lifecycles.

    Do not construct directly — use build(config).
    """

    def __init__(
        self,
        *,
        normaliser: Normaliser,
        input_coordinator: CheckerCoordinator,
        output_coordinator: CheckerCoordinator,
        main_llm: LLMProvider,
        # Guard checkers kept separately so the orchestrator can call
        # inspect_with_key() to forward the caller's api_key.
        input_guard: InputGuardChecker | None,
        output_guard: OutputGuardChecker | None,
        auditor: Auditor,
        server: ServerBackend | None,
        instance_id: str,
        # All LLM instances that need closing (deduped at build time)
        _llm_instances: list[LLMProvider],
    ) -> None:
        self._normaliser = normaliser
        self._input_coordinator = input_coordinator
        self._output_coordinator = output_coordinator
        self._main_llm = main_llm
        self._input_guard = input_guard
        self._output_guard = output_guard
        self._auditor = auditor
        self._server = server
        self._instance_id = instance_id
        self._llm_instances = _llm_instances

    # ------------------------------------------------------------------
    # Library interface
    # ------------------------------------------------------------------

    def evaluate(self, messages: list[dict[str, Any]]) -> EvaluationResult:
        """Evaluate a messages list through the full safety pipeline.

        Returns EvaluationResult. Never raises on content or infrastructure
        failures — those are in EvaluationResult.failure_mode.

        Raises:
          ValueError / TypeError  — precondition violation (caller bug)
          PipelineInternalError   — component bug (instance must be discarded)
        """
        # Precondition validation
        if messages is None:
            raise ValueError("messages must not be None")
        if not isinstance(messages, list):
            raise TypeError(f"messages must be a list, got {type(messages).__name__}")
        if len(messages) == 0:
            raise ValueError("messages must not be empty")

        # Extract last user message
        prompt = _extract_last_user_message(messages)
        if prompt is None:
            raise ValueError(
                "messages list contains no entry with role='user'. "
                "At least one user message is required."
            )

        audit_id = secrets.token_hex(8)
        return self._run_pipeline(
            messages=messages,
            prompt=prompt,
            api_key="",  # library path: empty key per spec §3
            audit_id=audit_id,
        )

    # ------------------------------------------------------------------
    # HTTP interface
    # ------------------------------------------------------------------

    def chat(
        self,
        request: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        """HTTP handler. Called by ServerBackend for each incoming request.

        Maps the request to evaluate(). Maps EvaluationResult to a
        ChatCompletionResponse dict. HTTP 200 for all EvaluationResult outcomes.

        Raises:
          ValueError / TypeError  — malformed request (→ HTTP 400 in ServerBackend)
          PipelineInternalError   — component bug (→ HTTP 500 in ServerBackend)
        """
        api_key = headers.get("authorization", "")
        # Strip "Bearer " prefix if present
        if api_key.lower().startswith("bearer "):
            api_key = api_key[7:]

        messages = request.get("messages")
        if messages is None:
            raise ValueError("Request missing required field 'messages'")

        # Precondition checks (raises ValueError — caught by ServerBackend → HTTP 400)
        if not isinstance(messages, list):
            raise TypeError("'messages' must be an array")
        if len(messages) == 0:
            raise ValueError("'messages' must not be empty")

        prompt = _extract_last_user_message(messages)
        if prompt is None:
            raise ValueError(
                "messages contains no entry with role='user'"
            )

        audit_id = secrets.token_hex(8)
        result = self._run_pipeline(
            messages=messages,
            prompt=prompt,
            api_key=api_key,
            audit_id=audit_id,
        )

        return _result_to_http_response(result, request)

    # ------------------------------------------------------------------
    # HTTP serve
    # ------------------------------------------------------------------

    def serve(self) -> None:
        """Start the HTTP server. Blocks until shutdown.

        Raises ConfigurationError if no ServerBackend is configured.
        """
        if self._server is None:
            raise ConfigurationError(
                "serve() called without a configured ServerBackend. "
                "Add a 'server' block to your config."
            )
        self._server.serve(self.chat)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release all held resources.

        Close order (§12): ServerBackend → checkers → LLMs → Auditor.
        Attempts close on all components regardless of individual failures.
        Re-raises the last exception after all components have been attempted.
        """
        last_exc: Exception | None = None

        def _try_close(component: Any, label: str) -> None:
            nonlocal last_exc
            try:
                component.close()
            except Exception as e:  # noqa: BLE001
                log.warning("Error closing %s: %s", label, e)
                last_exc = e

        if self._server is not None:
            _try_close(self._server, "ServerBackend")

        _try_close(self._input_coordinator, "InputCoordinator")
        _try_close(self._output_coordinator, "OutputCoordinator")

        for llm in self._llm_instances:
            _try_close(llm, f"LLM({llm!r})")

        _try_close(self._auditor, "Auditor")

        if last_exc is not None:
            raise last_exc

    # ------------------------------------------------------------------
    # Internal pipeline
    # ------------------------------------------------------------------

    def _run_pipeline(
        self,
        messages: list[dict[str, Any]],
        prompt: str,
        api_key: str,
        audit_id: str,
    ) -> EvaluationResult:
        """Execute the full pipeline. Returns EvaluationResult.

        Wraps unexpected component exceptions in PipelineInternalError.
        """
        # --- Normalisation ---
        try:
            norm_result = self._normaliser.normalise(prompt)
        except Exception as e:
            raise PipelineInternalError("Normaliser", e) from e

        self._log(audit_id, "input_normalised", {
            "stable": norm_result.stable,
            "original_length": len(prompt),
            "normalised_length": len(norm_result.text),
        })

        if not norm_result.stable:
            result = EvaluationResult(safe=False, failure_mode="normalisation_unstable")
            self._log(audit_id, "evaluate_complete", {
                "safe": False, "failure_mode": "normalisation_unstable"
            })
            return result

        working = norm_result.text

        # --- Input coordinator (EntropyChecker, PatternChecker, ScriptChecker,
        #     Translator, InputGuardChecker) ---
        # The coordinator handles the ScriptChecker→Translator prior_result handoff.
        # Guard checkers are in the coordinator but we call inspect_with_key on them
        # separately — so they are NOT in the coordinator; they run after it.
        # Actually per spec, all checkers run through the coordinator, but the
        # orchestrator owns api_key forwarding. The guard checkers implement both
        # inspect() (no key, used by coordinator) and inspect_with_key() (used here).
        # We run the coordinator for pre-guard checks, then call guard directly.

        try:
            coord_result, working = self._input_coordinator.inspect_with_prior_handoff(working)
        except Exception as e:
            raise PipelineInternalError("InputCoordinator", e) from e

        if not coord_result.safe:
            self._log(audit_id, "input_safety_check", {
                "safe": False,
                "failure_mode": coord_result.failure_mode,
                **( {"reason": coord_result.reason} if coord_result.reason else {}),
            })
            result = EvaluationResult(
                safe=False,
                failure_mode=coord_result.failure_mode,
                reason=coord_result.reason,
            )
            self._log(audit_id, "evaluate_complete", {
                "safe": False,
                "failure_mode": coord_result.failure_mode,
            })
            return result

        # --- Input guard (with api_key) ---
        if self._input_guard is not None:
            try:
                t0 = time.monotonic()
                guard_result = self._input_guard.inspect_with_key(working, api_key)
                latency_ms = int((time.monotonic() - t0) * 1000)
            except Exception as e:
                raise PipelineInternalError("InputGuardChecker", e) from e

            # Determine success for audit: guard_unavailable means call failed
            guard_call_success = guard_result.failure_mode != "guard_unavailable"
            self._log(audit_id, "llm_call", {
                "role": "input_guard",
                "success": guard_call_success,
                "latency_ms": latency_ms,
            })

            if not guard_result.safe:
                self._log(audit_id, "input_safety_check", {
                    "safe": False,
                    "failure_mode": guard_result.failure_mode,
                    **( {"reason": guard_result.reason} if guard_result.reason else {}),
                })
                result = EvaluationResult(
                    safe=False,
                    failure_mode=guard_result.failure_mode,
                    reason=guard_result.reason,
                )
                self._log(audit_id, "evaluate_complete", {
                    "safe": False,
                    "failure_mode": guard_result.failure_mode,
                })
                return result

            self._log(audit_id, "input_safety_check", {"safe": True})

        # --- Main LLM ---
        try:
            t0 = time.monotonic()
            completion = self._main_llm.complete(messages, api_key)
            latency_ms = int((time.monotonic() - t0) * 1000)
        except Exception as e:
            raise PipelineInternalError("MainLLM", e) from e

        if completion is None:
            self._log(audit_id, "llm_call", {
                "role": "main_llm", "success": False, "latency_ms": latency_ms,
            })
            result = EvaluationResult(safe=False, failure_mode="llm_unavailable")
            self._log(audit_id, "evaluate_complete", {
                "safe": False, "failure_mode": "llm_unavailable"
            })
            return result

        self._log(audit_id, "llm_call", {
            "role": "main_llm",
            "success": True,
            "latency_ms": latency_ms,
            "prompt_tokens": completion.usage.prompt_tokens,
            "completion_tokens": completion.usage.completion_tokens,
            "total_tokens": completion.usage.total_tokens,
            "finish_reason": completion.finish_reason,
        })

        # --- Output coordinator (OutputPatternChecker) ---
        try:
            out_coord_result, _ = self._output_coordinator.inspect_with_prior_handoff(
                completion.content
            )
        except Exception as e:
            raise PipelineInternalError("OutputCoordinator", e) from e

        if not out_coord_result.safe:
            self._log(audit_id, "output_safety_check", {
                "safe": False,
                "failure_mode": out_coord_result.failure_mode,
                **( {"reason": out_coord_result.reason} if out_coord_result.reason else {}),
            })
            result = EvaluationResult(
                safe=False,
                failure_mode=out_coord_result.failure_mode,
                reason=out_coord_result.reason,
            )
            self._log(audit_id, "evaluate_complete", {
                "safe": False, "failure_mode": out_coord_result.failure_mode,
            })
            return result

        # --- Output guard (with api_key) ---
        if self._output_guard is not None:
            try:
                t0 = time.monotonic()
                out_guard_result = self._output_guard.inspect_with_key(
                    completion.content, api_key
                )
                latency_ms = int((time.monotonic() - t0) * 1000)
            except Exception as e:
                raise PipelineInternalError("OutputGuardChecker", e) from e

            out_guard_success = out_guard_result.failure_mode != "guard_unavailable"
            self._log(audit_id, "llm_call", {
                "role": "output_guard",
                "success": out_guard_success,
                "latency_ms": latency_ms,
            })

            if not out_guard_result.safe:
                self._log(audit_id, "output_safety_check", {
                    "safe": False,
                    "failure_mode": out_guard_result.failure_mode,
                    **( {"reason": out_guard_result.reason} if out_guard_result.reason else {}),
                })
                result = EvaluationResult(
                    safe=False,
                    failure_mode=out_guard_result.failure_mode,
                    reason=out_guard_result.reason,
                )
                self._log(audit_id, "evaluate_complete", {
                    "safe": False, "failure_mode": out_guard_result.failure_mode,
                })
                return result

            self._log(audit_id, "output_safety_check", {"safe": True})

        # --- All checks passed ---
        result = EvaluationResult(safe=True, content=completion.content)
        self._log(audit_id, "evaluate_complete", {"safe": True})
        return result

    def _log(self, audit_id: str, event: str, data: dict[str, Any]) -> None:
        try:
            self._auditor.log(event, {"audit_id": audit_id, "instance_id": self._instance_id, **data})
        except Exception:  # noqa: BLE001 — log failures never propagate
            pass


# ---------------------------------------------------------------------------
# Composition root
# ---------------------------------------------------------------------------

def build(config: CarapexConfig) -> Carapex:
    """Construct a fully initialised Carapex instance.

    Validates all configuration and constructs all components before returning.
    Raises ConfigurationError on any invalid configuration.
    Cleans up any successfully constructed components before re-raising.
    Does not make network calls — LLM reachability is not verified at startup.
    """
    # Ensure all built-in implementations are registered
    import carapex.llm        # noqa: F401 — triggers autodiscovery
    import carapex.audit      # noqa: F401
    import carapex.normaliser # noqa: F401
    import carapex.server     # noqa: F401

    constructed: list[Any] = []  # track for cleanup on failure

    def _cleanup() -> None:
        for component in reversed(constructed):
            try:
                component.close()
            except Exception:  # noqa: BLE001
                pass

    try:
        instance_id = str(uuid.uuid4())

        # --- Main LLM ---
        main_llm = _build_llm(config.main_llm, "main_llm")
        constructed.append(main_llm)

        # --- Optional LLM roles — fall back to main_llm if not configured ---
        guard_llm = _build_llm(config.input_guard_llm, "input_guard_llm") if config.input_guard_llm else main_llm
        if config.input_guard_llm:
            constructed.append(guard_llm)

        output_guard_llm_cfg = config.output_guard_llm
        safety_cfg = config.safety or {}
        output_guard_enabled = bool(safety_cfg.get("output_guard_enabled", True))

        if output_guard_enabled and output_guard_llm_cfg:
            raise ConfigurationError(
                "output_guard_llm is configured but output_guard_enabled is True. "
                "If output_guard_enabled is True and you want a dedicated output guard LLM, "
                "set output_guard_llm separately. "
                "If you want to disable output guard entirely, set output_guard_enabled: false "
                "and remove output_guard_llm."
            )

        # Resolve output guard LLM
        if output_guard_enabled:
            out_guard_llm = (
                _build_llm(output_guard_llm_cfg, "output_guard_llm")
                if output_guard_llm_cfg
                else main_llm
            )
            if output_guard_llm_cfg:
                constructed.append(out_guard_llm)
        else:
            if output_guard_llm_cfg:
                raise ConfigurationError(
                    "output_guard_llm is configured but output_guard_enabled is false. "
                    "Remove output_guard_llm or set output_guard_enabled: true."
                )
            out_guard_llm = None

        translator_llm = (
            _build_llm(config.translator_llm, "translator_llm")
            if config.translator_llm
            else main_llm
        )
        if config.translator_llm:
            constructed.append(translator_llm)

        # --- Normaliser ---
        normaliser = _build_normaliser(config.normaliser or {})

        # --- Safety configuration ---
        entropy_threshold = safety_cfg.get("entropy_threshold", 5.8)
        entropy_min_length = int(safety_cfg.get("entropy_min_length", 50))
        script_confidence = float(safety_cfg.get("script_confidence_threshold", 0.80))
        input_guard_temp = float(safety_cfg.get("input_guard_temperature", 0.1))
        output_guard_temp = float(safety_cfg.get("output_guard_temperature", 0.1))
        translation_temp = float(safety_cfg.get("translation_temperature", 0.0))
        input_guard_prompt_path = safety_cfg.get("input_guard_system_prompt_path")
        output_guard_prompt_path = safety_cfg.get("output_guard_system_prompt_path")
        injection_patterns = safety_cfg.get("injection_patterns")  # None → use default

        # Validate temperatures (§8)
        if not (0.0 < input_guard_temp <= 1.0):
            raise ConfigurationError(
                f"safety.input_guard_temperature must be in (0.0, 1.0] exclusive of 0.0, "
                f"got {input_guard_temp}"
            )
        if not (0.0 < output_guard_temp <= 1.0):
            raise ConfigurationError(
                f"safety.output_guard_temperature must be in (0.0, 1.0] exclusive of 0.0, "
                f"got {output_guard_temp}"
            )
        if not (0.0 <= translation_temp <= 1.0):
            raise ConfigurationError(
                f"safety.translation_temperature must be in [0.0, 1.0], "
                f"got {translation_temp}"
            )

        # Validate injection_patterns (§8 — empty list or explicit null raises)
        if injection_patterns is not None and len(injection_patterns) == 0:
            raise ConfigurationError(
                "safety.injection_patterns must not be an empty list. "
                "Supply a non-empty list or omit the field to use the built-in default set."
            )

        # --- Build checkers ---
        try:
            pattern_checker = PatternChecker(patterns=injection_patterns)
        except ValueError as e:
            raise ConfigurationError(f"Invalid injection pattern: {e}") from e

        entropy_checker = EntropyChecker(
            threshold=entropy_threshold,
            min_length=entropy_min_length,
        )
        script_checker = ScriptChecker(confidence_threshold=script_confidence)
        translator = Translator(llm=translator_llm, temperature=translation_temp)

        input_guard: InputGuardChecker | None = InputGuardChecker(
            llm=guard_llm,
            temperature=input_guard_temp,
            system_prompt_path=input_guard_prompt_path,
        )

        # Input coordinator: EntropyChecker, PatternChecker, ScriptChecker, Translator
        # InputGuardChecker runs separately (needs api_key forwarding)
        input_coordinator = CheckerCoordinator([
            entropy_checker,
            pattern_checker,
            script_checker,
            translator,
        ])

        # Output coordinator: OutputPatternChecker
        # OutputGuardChecker runs separately (needs api_key forwarding)
        output_pattern_checker = OutputPatternChecker()
        output_coordinator = CheckerCoordinator([output_pattern_checker])

        output_guard: OutputGuardChecker | None = None
        if output_guard_enabled:
            output_guard = OutputGuardChecker(
                llm=out_guard_llm,  # type: ignore[arg-type]
                temperature=output_guard_temp,
                system_prompt_path=output_guard_prompt_path,
            )

        # --- Auditor ---
        auditor = _build_auditor(config.audit or {})
        constructed.append(auditor)

        # --- ServerBackend (optional) ---
        server: ServerBackend | None = None
        if config.server:
            server = _build_server(config.server)
            constructed.append(server)

        # Deduplicated LLM instances for close()
        llm_instances: list[LLMProvider] = _dedup_llms(
            main_llm,
            guard_llm if config.input_guard_llm else None,
            out_guard_llm if config.output_guard_llm else None,
            translator_llm if config.translator_llm else None,
        )

        instance = Carapex(
            normaliser=normaliser,
            input_coordinator=input_coordinator,
            output_coordinator=output_coordinator,
            main_llm=main_llm,
            input_guard=input_guard,
            output_guard=output_guard,
            auditor=auditor,
            server=server,
            instance_id=instance_id,
            _llm_instances=llm_instances,
        )

        # Emit carapex_init record (§18)
        _emit_init(instance, config, instance_id)

        return instance

    except ConfigurationError:
        _cleanup()
        raise
    except Exception as e:
        _cleanup()
        raise ConfigurationError(f"Build failed: {e}") from e


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------

def _build_llm(cfg: dict[str, Any], field_name: str) -> LLMProvider:
    llm_type = cfg.get("type")
    if not llm_type:
        raise ConfigurationError(
            f"'{field_name}' config missing required field 'type'. "
            f"Registered LLM types are resolved from the llm registry."
        )
    try:
        cls = get_llm(llm_type)
    except KeyError as e:
        raise ConfigurationError(
            f"'{field_name}.type' value {llm_type!r} is not a registered LLMProvider. "
            f"{e}"
        ) from e
    try:
        return cls.from_config(cfg)
    except Exception as e:
        raise ConfigurationError(f"Failed to construct '{field_name}': {e}") from e


def _build_normaliser(cfg: dict[str, Any]) -> Normaliser:
    max_passes = int(cfg.get("max_passes", 5))
    if max_passes < 1:
        raise ConfigurationError(
            f"normaliser.max_passes must be >= 1, got {max_passes}"
        )

    decoder_names: list[str] | None = cfg.get("decoders")

    # Import to trigger decoder autodiscovery
    import carapex.normaliser  # noqa: F401

    from carapex.core.registry import all_decoder_names, get_decoder

    if decoder_names is None:
        # Use all registered decoders in registration order
        decoder_names = all_decoder_names()

    decoders = []
    for name in decoder_names:
        try:
            cls = get_decoder(name)
        except KeyError as e:
            raise ConfigurationError(
                f"normaliser.decoders references unknown decoder {name!r}. {e}"
            ) from e
        decoders.append(cls())

    return Normaliser(decoders=decoders, max_passes=max_passes)


def _build_auditor(cfg: dict[str, Any]) -> Auditor:
    auditor_type = cfg.get("type", "file")
    try:
        cls = get_auditor(auditor_type)
    except KeyError as e:
        raise ConfigurationError(
            f"audit.type value {auditor_type!r} is not a registered Auditor. {e}"
        ) from e
    try:
        return cls.from_config(cfg)
    except Exception as e:
        raise ConfigurationError(f"Failed to construct auditor: {e}") from e


def _build_server(cfg: dict[str, Any]) -> ServerBackend:
    server_type = cfg.get("type")
    if not server_type:
        raise ConfigurationError("server config missing required field 'type'")
    try:
        cls = get_server(server_type)
    except KeyError as e:
        raise ConfigurationError(
            f"server.type value {server_type!r} is not a registered ServerBackend. {e}"
        ) from e
    try:
        return cls.from_config(cfg)
    except Exception as e:
        raise ConfigurationError(f"Failed to construct ServerBackend: {e}") from e


def _dedup_llms(*llms: LLMProvider | None) -> list[LLMProvider]:
    """Return a deduplicated list of LLM instances (by identity, not equality)."""
    seen_ids: set[int] = set()
    result: list[LLMProvider] = []
    for llm in llms:
        if llm is not None and id(llm) not in seen_ids:
            seen_ids.add(id(llm))
            result.append(llm)
    return result


def _extract_last_user_message(messages: list[dict[str, Any]]) -> str | None:
    """Return the content of the last message with role='user', or None."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content")
            if content is None:
                raise ValueError("Last user message has null content")
            return content
    return None


def _result_to_http_response(
    result: EvaluationResult,
    request: dict[str, Any],
) -> dict[str, Any]:
    """Map EvaluationResult to a ChatCompletionResponse dict (§3)."""
    import time as _time

    model = request.get("model", "unknown")
    created = int(_time.time())
    response_id = f"carapex-{secrets.token_hex(8)}"

    if result.safe:
        return {
            "id": response_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": result.content},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    # Blocked — synthetic assistant message with finish_reason="content_filter"
    content = result.reason if result.reason else _BLOCKED_CONTENT
    return {
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "content_filter",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _emit_init(instance: Carapex, config: CarapexConfig, instance_id: str) -> None:
    """Emit the carapex_init audit record (§18)."""
    main_url = config.main_llm.get("url", "unknown")
    instance._auditor.log("carapex_init", {
        "instance_id": instance_id,
        "version": "0.13.0",
        "main_llm": main_url,
        "input_guard_llm": (config.input_guard_llm or {}).get("url", main_url),
        "output_guard_llm": (config.output_guard_llm or {}).get("url", main_url),
        "translator_llm": (config.translator_llm or {}).get("url", main_url),
        "debug": config.debug,
    })
