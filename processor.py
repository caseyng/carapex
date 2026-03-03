"""
processor.py
------------
Carapex — core orchestrator.

Pipeline for every run() call:
    1. Normalise      — expose encoding attacks, return NormaliserResult
    2. Stability check — if unstable, fail closed
    3. Input check    — pattern → entropy → script → translate → guard
    4. If unsafe      → return Response(safe=False, ...)
    5. Call LLM       — original prompt, not normalised or translated
    6. Output check   — pattern + semantic (if enabled)
    7. If unsafe      → return Response(safe=False, ...)
    8. Return         → Response(safe=True, output=raw)

Response.failure_mode distinguishes every outcome:

    safe=True,  failure_mode=None
        Clean pass. All layers ran. Output verified.

    safe=False, failure_mode="safety_violation"
        Genuine content refusal. Pattern or guard found adversarial intent.

    safe=False, failure_mode="entropy_exceeded"
        Input entropy above threshold. Possible encoding attack.

    safe=False, failure_mode="guard_unavailable"
        Guard backend returned no response. Shell integrity failure.

    safe=False, failure_mode="guard_evaluation_corrupt"
        Guard responded but output unparseable. Possible injection.

    safe=False, failure_mode="translation_failed"
        Translation layer could not process input language. Shell integrity failure.

    safe=False, failure_mode="normalisation_unstable"
        Input did not converge after max_passes. Possible adversarial input.

    safe=False, failure_mode="backend_unavailable"
        Main LLM returned no response. Infrastructure failure.

Callers MUST branch on failure_mode.
safe=False alone does not distinguish content refusal from infrastructure failure.
These require different operational responses — do not conflate them.
"""

import logging
import uuid
from dataclasses import dataclass
from typing import Optional

from .backends.base         import LLMBackend
from .safety.base           import SafetyChecker
from .normaliser.normaliser import Normaliser
from .audit.base            import AuditBackend
from .exceptions            import BackendUnavailableError, PipelineInternalError

logger = logging.getLogger(__name__)


@dataclass
class Response:
    """
    Result of Carapex.run().

    output       : LLM response. None if refused or any layer failed.
    safe         : False if any layer failed or refused. True only on full
                   clean pass through all layers.
    refusal      : human-readable explanation when safe=False.
    failure_mode : machine-readable category. Always read before acting on safe.

                   None                        — clean pass (safe=True only)
                   "safety_violation"          — genuine content refusal
                   "entropy_exceeded"          — statistical anomaly detected
                   "guard_unavailable"         — guard infrastructure failure
                   "guard_evaluation_corrupt"  — guard output integrity failure
                   "translation_failed"        — language processing failure
                   "normalisation_unstable"    — input did not stabilise
                   "backend_unavailable"       — main LLM infrastructure failure

    audit_id     : trace ID linking to audit log entries.
    tokens_in    : prompt tokens (if backend reports usage).
    tokens_out   : completion tokens.
    tokens_total : combined.

    Caller contract — branch on failure_mode:

        r = cp.run(prompt)

        if r.failure_mode in (
            "guard_unavailable",
            "guard_evaluation_corrupt",
            "translation_failed",
        ):
            # Shell integrity failure — alert and investigate
            alert_oncall(r.failure_mode)
            return service_unavailable()

        elif r.failure_mode == "normalisation_unstable":
            # Possible adversarial input — log and reject
            log_security_event(r)
            return service_unavailable()

        elif not r.safe:
            # Content refusal (safety_violation, entropy_exceeded)
            return r.refusal

        else:
            return r.output
    """
    output:        Optional[str]
    safe:          bool
    refusal:       Optional[str]  = None
    failure_mode:  Optional[str]  = None
    audit_id:      Optional[str]  = None
    tokens_in:     Optional[int]  = None
    tokens_out:    Optional[int]  = None
    tokens_total:  Optional[int]  = None


class Carapex:
    """
    Hardened prompt execution boundary.
    All implementations injected — never imports concrete classes.
    """

    def __init__(
        self,
        system_prompt:  str,
        backend:        LLMBackend,
        input_checker:  SafetyChecker,
        output_checker: SafetyChecker,
        normaliser:     Normaliser,
        audit:          AuditBackend,
        debug:          bool = False,
        guard_backend:  Optional[LLMBackend] = None,
    ):
        self._system_prompt  = system_prompt
        self._backend        = backend
        self._guard_backend  = guard_backend
        self._input_checker  = input_checker
        self._output_checker = output_checker
        self._normaliser     = normaliser
        self._audit          = audit
        self._debug          = debug

        self._audit.log({
            "event"          : "carapex_init",
            "backend"        : repr(backend),
            "input_checker"  : repr(input_checker),
            "output_checker" : repr(output_checker),
            "normaliser"     : repr(normaliser),
            "debug"          : debug,
        })

    def health_check(self) -> bool:
        """Verify main backend is reachable. Raises BackendUnavailableError if not."""
        if not self._backend.health_check():
            raise BackendUnavailableError(
                f"{self._backend!r} is not ready."
            )
        return True

    def run(self, prompt: str) -> Response:
        """
        Process prompt through full pipeline.

        Returns Response for all outcomes — including infrastructure failures.
        Raises PipelineInternalError on component bugs.
        Never raises on expected operational failures.

        Always read Response.failure_mode before acting on Response.safe.
        """
        audit_id = str(uuid.uuid4())

        # ── 1. Normalise ───────────────────────────────────────────────────
        norm_result = self._normaliser.normalise(prompt)

        self._audit.log({
            "event"    : "input_normalised",
            "audit_id" : audit_id,
            "stable"   : norm_result.stable,
            "original_length"   : len(prompt),
            "normalised_length" : len(norm_result.text),
        })

        # ── 2. Stability check ─────────────────────────────────────────────
        if not norm_result.stable:
            return self._refused(
                reason       = (
                    "Input could not be normalised to a stable form. "
                    "Request rejected for safety."
                ),
                audit_id     = audit_id,
                stage        = "normalisation",
                failure_mode = "normalisation_unstable",
            )

        normalised = norm_result.text

        # ── 3. Input safety ────────────────────────────────────────────────
        input_result = self._input_checker.check(normalised)

        self._audit.log({
            "event"        : "input_safety_check",
            "audit_id"     : audit_id,
            "safe"         : input_result.safe,
            "reason"       : input_result.reason,
            "failure_mode" : input_result.failure_mode,
        })

        if not input_result.safe:
            return self._refused(
                reason       = input_result.reason,
                audit_id     = audit_id,
                stage        = "input",
                failure_mode = input_result.failure_mode or "safety_violation",
            )

        # ── 4. Call LLM ────────────────────────────────────────────────────
        # Original prompt — not normalised, not translated.
        # Normalisation and translation are for evaluation only.
        raw = self._call_llm(prompt, audit_id)

        if raw is None:
            tokens_in, tokens_out, tokens_total = self._extract_tokens()
            self._audit.log({
                "event"        : "run_complete",
                "audit_id"     : audit_id,
                "failure_mode" : "backend_unavailable",
                "tokens_in"    : tokens_in,
                "tokens_out"   : tokens_out,
                "tokens_total" : tokens_total,
            })
            return Response(
                output       = None,
                safe         = False,
                refusal      = "The language model did not return a response.",
                failure_mode = "backend_unavailable",
                audit_id     = audit_id,
            )

        # ── 5. Output safety ───────────────────────────────────────────────
        output_result = self._output_checker.check(raw)
        tokens_in, tokens_out, tokens_total = self._extract_tokens()

        self._audit.log({
            "event"        : "output_safety_check",
            "audit_id"     : audit_id,
            "safe"         : output_result.safe,
            "reason"       : output_result.reason,
            "failure_mode" : output_result.failure_mode,
            "tokens_in"    : tokens_in,
            "tokens_out"   : tokens_out,
            "tokens_total" : tokens_total,
        })

        if not output_result.safe:
            return self._refused(
                reason       = output_result.reason,
                audit_id     = audit_id,
                stage        = "output",
                failure_mode = output_result.failure_mode or "safety_violation",
                tokens_in    = tokens_in,
                tokens_out   = tokens_out,
                tokens_total = tokens_total,
            )

        # ── 6. Return ──────────────────────────────────────────────────────
        self._audit.log({
            "event"        : "run_complete",
            "audit_id"     : audit_id,
            "failure_mode" : None,
            "tokens_in"    : tokens_in,
            "tokens_out"   : tokens_out,
            "tokens_total" : tokens_total,
        })

        return Response(
            output       = raw,
            safe         = True,
            failure_mode = None,
            audit_id     = audit_id,
            tokens_in    = tokens_in,
            tokens_out   = tokens_out,
            tokens_total = tokens_total,
        )

    def close(self) -> None:
        self._backend.close()
        if self._guard_backend is not None:
            self._guard_backend.close()
        self._audit.close()

    def __enter__(self) -> "Carapex":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __repr__(self) -> str:
        parts = [
            f"backend={self._backend!r}",
            f"input_checker={self._input_checker!r}",
            f"normaliser={self._normaliser!r}",
        ]
        if self._guard_backend is not None:
            parts.append(f"guard_backend={self._guard_backend!r}")
        return f"Carapex({', '.join(parts)})"

    def _refused(
        self,
        reason:       str,
        audit_id:     str,
        stage:        str,
        failure_mode: str,
        tokens_in:    Optional[int] = None,
        tokens_out:   Optional[int] = None,
        tokens_total: Optional[int] = None,
    ) -> Response:
        self._audit.log({
            "event"        : "safety_refused",
            "audit_id"     : audit_id,
            "stage"        : stage,
            "reason"       : reason,
            "failure_mode" : failure_mode,
            "tokens_in"    : tokens_in,
            "tokens_out"   : tokens_out,
            "tokens_total" : tokens_total,
        })
        return Response(
            output       = None,
            safe         = False,
            refusal      = reason,
            failure_mode = failure_mode,
            audit_id     = audit_id,
            tokens_in    = tokens_in,
            tokens_out   = tokens_out,
            tokens_total = tokens_total,
        )

    def _call_llm(self, prompt: str, audit_id: str) -> Optional[str]:
        response = self._backend.chat(self._system_prompt, prompt)
        self._audit.log({
            "event"    : "llm_call",
            "audit_id" : audit_id,
            "success"  : response is not None,
        })
        return response

    def _extract_tokens(self) -> tuple[Optional[int], Optional[int], Optional[int]]:
        # _extract_tokens() is called immediately after _call_llm() and before
        # any subsequent guard or output checker calls. This ensures last_usage()
        # reflects the main LLM call even when guard_backend shares the same
        # backend instance (development config). Do not move this call later in
        # the pipeline without revisiting token attribution correctness.
        try:
            usage        = self._backend.last_usage()
            tokens_in    = usage.get("prompt_tokens")
            tokens_out   = usage.get("completion_tokens")
            tokens_total = usage.get("total_tokens")
            return tokens_in, tokens_out, tokens_total
        except (AttributeError, TypeError, KeyError) as e:
            logger.debug("Token usage unavailable from %r: %s", self._backend, e)
            return None, None, None
