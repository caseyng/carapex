"""
Tests for build() config validation.

Uses minimal stub LLM and auditor implementations registered at module level
so build() can be exercised without real network access.

All tests must clean up after themselves if they modify global registry state.
"""

from __future__ import annotations

import pytest

from carapex.core.exceptions import ConfigurationError
from carapex.core.registry import (
    _llm_registry,
    _auditor_registry,
    register_llm,
    register_auditor,
)
from carapex.llm.base import LLMProvider
from carapex.audit.base import Auditor
from carapex.audit.memory_auditor import InMemoryAuditor


# ---------------------------------------------------------------------------
# Minimal implementations registered once for this module
# ---------------------------------------------------------------------------

class _CfgTestLLM(LLMProvider):
    """Minimal LLM stub for build() tests."""
    name = "_cfg_test_llm"

    def complete(self, messages, api_key):
        return None

    def complete_with_temperature(self, messages, temperature, api_key):
        return None

    def close(self) -> None:
        pass

    @classmethod
    def from_config(cls, raw):
        return cls()

    @classmethod
    def default_config(cls):
        return {"type": "_cfg_test_llm", "url": "http://test"}

    def __repr__(self):
        return "_CfgTestLLM()"


class _CfgTestAuditor(Auditor):
    """Minimal auditor stub for build() tests. Stores records in list."""
    name = "_cfg_test_auditor"

    def __init__(self):
        self.records: list = []

    def log(self, event, data):
        self.records.append({"event": event, **data})

    def close(self) -> None:
        pass

    @classmethod
    def from_config(cls, raw):
        return cls()

    def __repr__(self):
        return "_CfgTestAuditor()"


# Register once; guard against re-registration when pytest re-imports
if "_cfg_test_llm" not in _llm_registry:
    register_llm(_CfgTestLLM)

if "_cfg_test_auditor" not in _auditor_registry:
    register_auditor(_CfgTestAuditor)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_config(**safety_overrides):
    """Return a CarapexConfig using the test stub types."""
    from carapex.core.config import CarapexConfig

    safety = {
        "input_guard_temperature": 0.1,
        "output_guard_temperature": 0.1,
        "translation_temperature": 0.0,
        **safety_overrides,
    }
    return CarapexConfig(
        main_llm={"type": "_cfg_test_llm", "url": "http://test"},
        safety=safety,
        audit={"type": "_cfg_test_auditor"},
    )


def _build_and_close(**safety_overrides):
    """Build and immediately close an instance — asserts no ConfigurationError."""
    from carapex.carapex import build
    instance = build(_minimal_config(**safety_overrides))
    instance.close()


# ---------------------------------------------------------------------------
# Temperature validation
# ---------------------------------------------------------------------------

class TestTemperatureValidation:
    def test_input_guard_temp_zero_raises(self):
        with pytest.raises(ConfigurationError, match="input_guard_temperature"):
            _build_and_close(input_guard_temperature=0.0)

    def test_input_guard_temp_small_positive_ok(self):
        _build_and_close(input_guard_temperature=0.0001)

    def test_input_guard_temp_at_one_ok(self):
        _build_and_close(input_guard_temperature=1.0)

    def test_input_guard_temp_above_one_raises(self):
        with pytest.raises(ConfigurationError, match="input_guard_temperature"):
            _build_and_close(input_guard_temperature=1.1)

    def test_output_guard_temp_zero_raises(self):
        with pytest.raises(ConfigurationError, match="output_guard_temperature"):
            _build_and_close(output_guard_temperature=0.0)

    def test_translation_temp_zero_ok(self):
        # translation_temperature=0.0 is explicitly allowed (deterministic)
        _build_and_close(translation_temperature=0.0)

    def test_translation_temp_one_ok(self):
        _build_and_close(translation_temperature=1.0)

    def test_translation_temp_above_one_raises(self):
        with pytest.raises(ConfigurationError, match="translation_temperature"):
            _build_and_close(translation_temperature=1.1)

    def test_translation_temp_negative_raises(self):
        with pytest.raises(ConfigurationError, match="translation_temperature"):
            _build_and_close(translation_temperature=-0.1)


# ---------------------------------------------------------------------------
# injection_patterns validation
# ---------------------------------------------------------------------------

class TestInjectionPatternsValidation:
    def test_empty_list_raises(self):
        with pytest.raises(ConfigurationError, match="injection_patterns"):
            _build_and_close(injection_patterns=[])

    def test_invalid_regex_raises(self):
        with pytest.raises(ConfigurationError):
            _build_and_close(injection_patterns=["[unclosed bracket"])

    def test_null_uses_defaults(self):
        # injection_patterns=None means use built-in defaults
        _build_and_close(injection_patterns=None)

    def test_custom_patterns_accepted(self):
        _build_and_close(injection_patterns=[r"CUSTOM_BANNED"])


# ---------------------------------------------------------------------------
# output_guard_enabled + output_guard_llm conflict
# ---------------------------------------------------------------------------

class TestOutputGuardConfig:
    def test_output_guard_disabled_with_llm_config_raises(self):
        from carapex.core.config import CarapexConfig
        from carapex.carapex import build

        config = CarapexConfig(
            main_llm={"type": "_cfg_test_llm", "url": "http://test"},
            output_guard_llm={"type": "_cfg_test_llm", "url": "http://guard"},
            safety={
                "input_guard_temperature": 0.1,
                "output_guard_temperature": 0.1,
                "output_guard_enabled": False,
            },
            audit={"type": "_cfg_test_auditor"},
        )
        with pytest.raises(ConfigurationError, match="output_guard"):
            build(config)

    def test_output_guard_disabled_without_llm_config_ok(self):
        _build_and_close(output_guard_enabled=False)


# ---------------------------------------------------------------------------
# Normaliser config validation
# ---------------------------------------------------------------------------

class TestNormaliserConfig:
    def test_max_passes_zero_raises(self):
        from carapex.core.config import CarapexConfig
        from carapex.carapex import build

        config = CarapexConfig(
            main_llm={"type": "_cfg_test_llm", "url": "http://test"},
            normaliser={"max_passes": 0},
            safety={"input_guard_temperature": 0.1, "output_guard_temperature": 0.1},
            audit={"type": "_cfg_test_auditor"},
        )
        with pytest.raises(ConfigurationError, match="max_passes"):
            build(config)

    def test_max_passes_negative_raises(self):
        from carapex.core.config import CarapexConfig
        from carapex.carapex import build

        config = CarapexConfig(
            main_llm={"type": "_cfg_test_llm", "url": "http://test"},
            normaliser={"max_passes": -1},
            safety={"input_guard_temperature": 0.1, "output_guard_temperature": 0.1},
            audit={"type": "_cfg_test_auditor"},
        )
        with pytest.raises(ConfigurationError, match="max_passes"):
            build(config)

    def test_unknown_decoder_raises(self):
        from carapex.core.config import CarapexConfig
        from carapex.carapex import build

        config = CarapexConfig(
            main_llm={"type": "_cfg_test_llm", "url": "http://test"},
            normaliser={"decoders": ["nonexistent_decoder"]},
            safety={"input_guard_temperature": 0.1, "output_guard_temperature": 0.1},
            audit={"type": "_cfg_test_auditor"},
        )
        with pytest.raises(ConfigurationError, match="nonexistent_decoder"):
            build(config)

    def test_max_passes_one_ok(self):
        from carapex.core.config import CarapexConfig
        from carapex.carapex import build

        config = CarapexConfig(
            main_llm={"type": "_cfg_test_llm", "url": "http://test"},
            normaliser={"max_passes": 1},
            safety={"input_guard_temperature": 0.1, "output_guard_temperature": 0.1},
            audit={"type": "_cfg_test_auditor"},
        )
        instance = build(config)
        instance.close()


# ---------------------------------------------------------------------------
# Registry errors
# ---------------------------------------------------------------------------

class TestBuildRegistryErrors:
    def test_unknown_llm_type_raises(self):
        from carapex.core.config import CarapexConfig
        from carapex.carapex import build

        config = CarapexConfig(
            main_llm={"type": "nonexistent_llm_type"},
            audit={"type": "_cfg_test_auditor"},
        )
        with pytest.raises(ConfigurationError):
            build(config)

    def test_unknown_auditor_type_raises(self):
        from carapex.core.config import CarapexConfig
        from carapex.carapex import build

        config = CarapexConfig(
            main_llm={"type": "_cfg_test_llm", "url": "http://test"},
            safety={"input_guard_temperature": 0.1, "output_guard_temperature": 0.1},
            audit={"type": "nonexistent_auditor_type"},
        )
        with pytest.raises(ConfigurationError):
            build(config)

    def test_unknown_server_type_raises(self):
        from carapex.core.config import CarapexConfig
        from carapex.carapex import build

        config = CarapexConfig(
            main_llm={"type": "_cfg_test_llm", "url": "http://test"},
            safety={"input_guard_temperature": 0.1, "output_guard_temperature": 0.1},
            audit={"type": "_cfg_test_auditor"},
            server={"type": "nonexistent_server_type"},
        )
        with pytest.raises(ConfigurationError):
            build(config)


# ---------------------------------------------------------------------------
# Build failure cleanup
# ---------------------------------------------------------------------------

class TestBuildFailureCleanup:
    def test_llm_closed_when_auditor_build_fails(self):
        """On build failure, already-constructed components must have close() called."""
        from carapex.core.config import CarapexConfig
        from carapex.carapex import build

        closed = []

        class TrackingLLM(_CfgTestLLM):
            name = "_tracking_llm"

            def close(self):
                closed.append("closed")

        class FailingAuditor(Auditor):
            name = "_always_fails_auditor"

            def log(self, event, data): pass
            def close(self): pass

            @classmethod
            def from_config(cls, raw):
                raise ValueError("auditor construction always fails")

            def __repr__(self): return "FailingAuditor()"

        # Register temporarily
        if "_tracking_llm" not in _llm_registry:
            register_llm(TrackingLLM)
        if "_always_fails_auditor" not in _auditor_registry:
            register_auditor(FailingAuditor)

        try:
            config = CarapexConfig(
                main_llm={"type": "_tracking_llm", "url": "http://test"},
                safety={"input_guard_temperature": 0.1, "output_guard_temperature": 0.1},
                audit={"type": "_always_fails_auditor"},
            )
            with pytest.raises(ConfigurationError):
                build(config)

            assert closed, "LLM close() must be called on build failure"
        finally:
            _llm_registry.pop("_tracking_llm", None)
            _auditor_registry.pop("_always_fails_auditor", None)
