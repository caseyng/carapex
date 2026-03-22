"""
Shared test fixtures and stub implementations.

Stubs are minimal — they implement only what is needed for the test.
No third-party mocking framework required for the stubs themselves;
pytest fixtures wire them up.
"""

from __future__ import annotations

from typing import Any

import pytest

from carapex.audit.memory_auditor import InMemoryAuditor
from carapex.core.types import CompletionResult, UsageResult
from carapex.llm.base import LLMProvider


# ---------------------------------------------------------------------------
# Stub LLM implementations
# ---------------------------------------------------------------------------

class StubLLM(LLMProvider):
    """Returns a fixed response for every call. Never raises."""

    name = "_stub"  # underscore prefix — not registered in the global registry

    def __init__(self, response: str = "Hello, world!") -> None:
        self._response = response
        self._calls: list[dict[str, Any]] = []

    def complete(
        self,
        messages: list[dict[str, str]],
        api_key: str,
    ) -> CompletionResult | None:
        self._calls.append({"messages": messages, "api_key": api_key, "temperature": None})
        if self._response is None:
            return None
        return CompletionResult(
            content=self._response,
            finish_reason="stop",
            usage=UsageResult(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

    def complete_with_temperature(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        api_key: str,
    ) -> CompletionResult | None:
        self._calls.append({"messages": messages, "api_key": api_key, "temperature": temperature})
        if self._response is None:
            return None
        return CompletionResult(
            content=self._response,
            finish_reason="stop",
            usage=UsageResult(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

    def close(self) -> None:
        pass

    @property
    def calls(self) -> list[dict[str, Any]]:
        return list(self._calls)


class UnavailableLLM(LLMProvider):
    """Always returns None — simulates an unreachable LLM."""

    name = "_unavailable"

    def complete(self, messages, api_key):
        return None

    def complete_with_temperature(self, messages, temperature, api_key):
        return None

    def close(self) -> None:
        pass


class SafeGuardLLM(StubLLM):
    """Guard LLM that always returns safe=true."""

    name = "_safe_guard"

    def __init__(self) -> None:
        super().__init__(response='{"safe": true}')


class UnsafeGuardLLM(StubLLM):
    """Guard LLM that always returns safe=false with a reason."""

    name = "_unsafe_guard"

    def __init__(self, reason: str = "Detected injection attempt") -> None:
        super().__init__(response=f'{{"safe": false, "reason": "{reason}"}}')


class CorruptGuardLLM(StubLLM):
    """Guard LLM that returns unparseable output."""

    name = "_corrupt_guard"

    def __init__(self) -> None:
        super().__init__(response="this is not json at all")


class StringTrueGuardLLM(StubLLM):
    """Guard LLM that returns 'true' as a string — must NOT be accepted."""

    name = "_string_true_guard"

    def __init__(self) -> None:
        super().__init__(response='{"safe": "true"}')


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def auditor() -> InMemoryAuditor:
    return InMemoryAuditor()


@pytest.fixture
def safe_guard() -> SafeGuardLLM:
    return SafeGuardLLM()


@pytest.fixture
def unsafe_guard() -> UnsafeGuardLLM:
    return UnsafeGuardLLM()


@pytest.fixture
def stub_llm() -> StubLLM:
    return StubLLM()


@pytest.fixture
def unavailable_llm() -> UnavailableLLM:
    return UnavailableLLM()


def make_messages(content: str = "Hello") -> list[dict[str, str]]:
    return [{"role": "user", "content": content}]


def make_carapex(
    *,
    main_llm: LLMProvider | None = None,
    guard_llm: LLMProvider | None = None,
    output_guard_llm: LLMProvider | None = None,
    auditor: InMemoryAuditor | None = None,
    output_guard_enabled: bool = True,
) -> Any:
    """Construct a Carapex instance directly (bypassing build()) for unit testing.

    Uses InMemoryAuditor and stub LLMs by default.
    Skips ScriptChecker/Translator to avoid lingua dependency in fast tests.
    """
    from carapex.carapex import Carapex
    from carapex.normaliser.base import Normaliser
    from carapex.safety.coordinator import CheckerCoordinator
    from carapex.safety.entropy import EntropyChecker
    from carapex.safety.guard import InputGuardChecker, OutputGuardChecker
    from carapex.safety.pattern import OutputPatternChecker, PatternChecker

    _main = main_llm or StubLLM()
    _guard = guard_llm or SafeGuardLLM()
    _out_guard = output_guard_llm or SafeGuardLLM()
    _auditor = auditor or InMemoryAuditor()

    # Import normaliser decoders to register them
    import carapex.normaliser  # noqa: F401

    from carapex.core.registry import all_decoder_names, get_decoder
    decoders = [get_decoder(n)() for n in all_decoder_names()]
    normaliser = Normaliser(decoders=decoders, max_passes=5)

    input_guard = InputGuardChecker(llm=_guard, temperature=0.1)
    output_guard = OutputGuardChecker(llm=_out_guard, temperature=0.1) if output_guard_enabled else None

    input_coordinator = CheckerCoordinator([
        EntropyChecker(),
        PatternChecker(),
    ])
    output_coordinator = CheckerCoordinator([OutputPatternChecker()])

    return Carapex(
        normaliser=normaliser,
        input_coordinator=input_coordinator,
        output_coordinator=output_coordinator,
        main_llm=_main,
        input_guard=input_guard,
        output_guard=output_guard,
        auditor=_auditor,
        server=None,
        instance_id="test-instance",
        _llm_instances=[_main, _guard, _out_guard],
    )
