"""
Tests for Carapex.close() contract (§12).

Close order per spec: ServerBackend → coordinators → guard checkers → LLMs → Auditor.
Every component must have close() called regardless of failures in others.
The last exception is re-raised after all components are attempted.
"""

from __future__ import annotations

from carapex.audit.memory_auditor import InMemoryAuditor
from carapex.carapex import Carapex
from carapex.normaliser.base import Normaliser
from carapex.safety.coordinator import CheckerCoordinator
from carapex.safety.entropy import EntropyChecker
from carapex.safety.guard import InputGuardChecker, OutputGuardChecker
from carapex.safety.pattern import OutputPatternChecker, PatternChecker
from tests.conftest import SafeGuardLLM, StubLLM


# ---------------------------------------------------------------------------
# Spy helpers
# ---------------------------------------------------------------------------

class CloseSpy:
    """Wraps close() to record whether it was called."""
    def __init__(self):
        self.closed = False

    def mark_closed(self):
        self.closed = True


class SpyLLM(StubLLM):
    name = "_spy_llm"

    def __init__(self, spy: CloseSpy):
        super().__init__()
        self._spy = spy

    def close(self):
        self._spy.mark_closed()


class SpyAuditor(InMemoryAuditor):
    def __init__(self, spy: CloseSpy):
        super().__init__()
        self._spy = spy

    def close(self):
        self._spy.mark_closed()
        super().close()


class SpyCoordinator(CheckerCoordinator):
    def __init__(self, spy: CloseSpy, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._spy = spy

    def close(self):
        self._spy.mark_closed()
        super().close()


class SpyGuard(InputGuardChecker):
    def __init__(self, spy: CloseSpy, **kwargs):
        super().__init__(**kwargs)
        self._spy = spy

    def close(self):
        self._spy.mark_closed()


class SpyOutputGuard(OutputGuardChecker):
    def __init__(self, spy: CloseSpy, **kwargs):
        super().__init__(**kwargs)
        self._spy = spy

    def close(self):
        self._spy.mark_closed()


def _make_spy_carapex(
    llm_spy: CloseSpy | None = None,
    auditor_spy: CloseSpy | None = None,
    input_coord_spy: CloseSpy | None = None,
    input_guard_spy: CloseSpy | None = None,
    output_guard_spy: CloseSpy | None = None,
) -> Carapex:
    guard = SafeGuardLLM()
    main_llm = SpyLLM(llm_spy) if llm_spy else StubLLM()
    auditor = SpyAuditor(auditor_spy) if auditor_spy else InMemoryAuditor()
    input_coord = (
        SpyCoordinator(input_coord_spy, [EntropyChecker(), PatternChecker()])
        if input_coord_spy
        else CheckerCoordinator([EntropyChecker(), PatternChecker()])
    )
    input_guard = (
        SpyGuard(input_guard_spy, llm=guard, temperature=0.1)
        if input_guard_spy
        else InputGuardChecker(llm=guard, temperature=0.1)
    )
    output_guard = (
        SpyOutputGuard(output_guard_spy, llm=guard, temperature=0.1)
        if output_guard_spy
        else OutputGuardChecker(llm=guard, temperature=0.1)
    )

    import carapex.normaliser  # noqa: F401
    from carapex.core.registry import all_decoder_names, get_decoder
    decoders = [get_decoder(n)() for n in all_decoder_names()]
    normaliser = Normaliser(decoders=decoders, max_passes=5)

    return Carapex(
        normaliser=normaliser,
        input_coordinator=input_coord,
        output_coordinator=CheckerCoordinator([OutputPatternChecker()]),
        main_llm=main_llm,
        input_guard=input_guard,
        output_guard=output_guard,
        auditor=auditor,
        server=None,
        instance_id="test-close",
        _llm_instances=[main_llm, guard],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCarapexClose:
    def test_close_calls_llm_close(self):
        spy = CloseSpy()
        cx = _make_spy_carapex(llm_spy=spy)
        cx.close()
        assert spy.closed

    def test_close_calls_auditor_close(self):
        spy = CloseSpy()
        cx = _make_spy_carapex(auditor_spy=spy)
        cx.close()
        assert spy.closed

    def test_close_calls_input_coordinator_close(self):
        spy = CloseSpy()
        cx = _make_spy_carapex(input_coord_spy=spy)
        cx.close()
        assert spy.closed

    def test_close_calls_input_guard_close(self):
        """BUG-4: InputGuardChecker.close() must be called."""
        spy = CloseSpy()
        cx = _make_spy_carapex(input_guard_spy=spy)
        cx.close()
        assert spy.closed

    def test_close_calls_output_guard_close(self):
        """BUG-4: OutputGuardChecker.close() must be called."""
        spy = CloseSpy()
        cx = _make_spy_carapex(output_guard_spy=spy)
        cx.close()
        assert spy.closed

    def test_close_with_none_output_guard_does_not_raise(self):
        guard = SafeGuardLLM()
        import carapex.normaliser  # noqa: F401
        from carapex.core.registry import all_decoder_names, get_decoder
        decoders = [get_decoder(n)() for n in all_decoder_names()]
        cx = Carapex(
            normaliser=Normaliser(decoders=decoders, max_passes=5),
            input_coordinator=CheckerCoordinator([EntropyChecker(), PatternChecker()]),
            output_coordinator=CheckerCoordinator([OutputPatternChecker()]),
            main_llm=StubLLM(),
            input_guard=InputGuardChecker(llm=guard, temperature=0.1),
            output_guard=None,  # explicitly None
            auditor=InMemoryAuditor(),
            server=None,
            instance_id="test-no-output-guard",
            _llm_instances=[StubLLM(), guard],
        )
        cx.close()  # must not raise

    def test_close_with_failing_component_still_closes_others(self):
        """All components must have close() attempted even if one raises."""
        auditor_spy = CloseSpy()

        class FailingCoordinator(CheckerCoordinator):
            def close(self):
                raise RuntimeError("coordinator close failed")

        guard = SafeGuardLLM()
        import carapex.normaliser  # noqa: F401
        from carapex.core.registry import all_decoder_names, get_decoder
        decoders = [get_decoder(n)() for n in all_decoder_names()]
        auditor = SpyAuditor(auditor_spy)
        cx = Carapex(
            normaliser=Normaliser(decoders=decoders, max_passes=5),
            input_coordinator=FailingCoordinator([]),  # will raise on close()
            output_coordinator=CheckerCoordinator([OutputPatternChecker()]),
            main_llm=StubLLM(),
            input_guard=InputGuardChecker(llm=guard, temperature=0.1),
            output_guard=None,
            auditor=auditor,
            server=None,
            instance_id="test-failing-close",
            _llm_instances=[StubLLM(), guard],
        )
        try:
            cx.close()
        except RuntimeError:
            pass  # expected — re-raised after all others are attempted

        assert auditor_spy.closed, "Auditor must be closed even if coordinator raised"

    def test_close_re_raises_last_exception(self):
        """When multiple components fail, the last exception must be re-raised."""
        import pytest

        class FailingCoordinator(CheckerCoordinator):
            def close(self):
                raise RuntimeError("first failure")

        class FailingAuditor(InMemoryAuditor):
            def close(self):
                raise RuntimeError("last failure")

        guard = SafeGuardLLM()
        import carapex.normaliser  # noqa: F401
        from carapex.core.registry import all_decoder_names, get_decoder
        decoders = [get_decoder(n)() for n in all_decoder_names()]
        cx = Carapex(
            normaliser=Normaliser(decoders=decoders, max_passes=5),
            input_coordinator=FailingCoordinator([]),
            output_coordinator=CheckerCoordinator([OutputPatternChecker()]),
            main_llm=StubLLM(),
            input_guard=InputGuardChecker(llm=guard, temperature=0.1),
            output_guard=None,
            auditor=FailingAuditor(),
            server=None,
            instance_id="test-multi-fail",
            _llm_instances=[StubLLM(), guard],
        )
        with pytest.raises(RuntimeError, match="last failure"):
            cx.close()

    def test_auditor_closed_after_llm(self):
        """Close order must be: components → LLMs → Auditor."""
        close_order = []

        class OrderTrackingLLM(StubLLM):
            name = "_order_llm"
            def close(self):
                close_order.append("llm")

        class OrderTrackingAuditor(InMemoryAuditor):
            def close(self):
                close_order.append("auditor")

        guard = SafeGuardLLM()
        import carapex.normaliser  # noqa: F401
        from carapex.core.registry import all_decoder_names, get_decoder
        decoders = [get_decoder(n)() for n in all_decoder_names()]
        main_llm = OrderTrackingLLM()
        auditor = OrderTrackingAuditor()
        cx = Carapex(
            normaliser=Normaliser(decoders=decoders, max_passes=5),
            input_coordinator=CheckerCoordinator([EntropyChecker(), PatternChecker()]),
            output_coordinator=CheckerCoordinator([OutputPatternChecker()]),
            main_llm=main_llm,
            input_guard=InputGuardChecker(llm=guard, temperature=0.1),
            output_guard=None,
            auditor=auditor,
            server=None,
            instance_id="test-order",
            _llm_instances=[main_llm, guard],
        )
        cx.close()
        assert close_order.index("llm") < close_order.index("auditor")
