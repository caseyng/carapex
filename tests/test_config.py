"""
tests/test_config.py
---------------------
Unit tests for config loading, validation, and generation.

No external dependencies required.
"""

import sys, os, json, tempfile, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from carapex import config
from carapex.config     import CarapexConfig
from carapex.exceptions import ConfigurationError


# ── from_dict ──────────────────────────────────────────────────────────────

class TestFromDict(unittest.TestCase):

    def test_default_config(self):
        cfg = config.default()
        self.assertEqual(cfg.system_prompt, "You are a helpful assistant.")
        self.assertIsInstance(cfg.backend, dict)
        self.assertIsNone(cfg.guard_backend)

    def test_minimal_dict(self):
        cfg = config.from_dict({})
        self.assertEqual(cfg.system_prompt, "You are a helpful assistant.")

    def test_custom_system_prompt(self):
        cfg = config.from_dict({"system_prompt": "You are a pirate."})
        self.assertEqual(cfg.system_prompt, "You are a pirate.")

    def test_backend_is_raw_dict(self):
        cfg = config.from_dict({
            "backend": {"temperature": 0.5, "max_tokens": 500}
        })
        self.assertIsInstance(cfg.backend, dict)
        self.assertEqual(cfg.backend["temperature"], 0.5)

    def test_audit_is_raw_dict(self):
        cfg = config.from_dict({"audit": {"backend": "null"}})
        self.assertIsInstance(cfg.audit, dict)
        self.assertEqual(cfg.audit["backend"], "null")

    def test_guard_backend_none(self):
        cfg = config.from_dict({"guard_backend": None})
        self.assertIsNone(cfg.guard_backend)

    def test_guard_backend_set(self):
        cfg = config.from_dict({
            "guard_backend": {"base_url": "http://localhost:8081"}
        })
        self.assertIsInstance(cfg.guard_backend, dict)
        self.assertEqual(cfg.guard_backend["base_url"], "http://localhost:8081")

    def test_normaliser_config(self):
        cfg = config.from_dict({
            "normaliser": {"max_passes": 3, "decoders": ["base64", "url"]}
        })
        self.assertEqual(cfg.normaliser.max_passes, 3)

    def test_empty_system_prompt_raises_configuration_error(self):
        with self.assertRaises(ConfigurationError):
            config.from_dict({"system_prompt": ""})

    def test_whitespace_system_prompt_raises_configuration_error(self):
        with self.assertRaises(ConfigurationError):
            config.from_dict({"system_prompt": "   "})


# ── File loading ───────────────────────────────────────────────────────────

class TestLoadFile(unittest.TestCase):

    def test_load_valid_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump({"system_prompt": "Test assistant."}, f)
            path = f.name
        try:
            cfg = config.load(path)
            self.assertEqual(cfg.system_prompt, "Test assistant.")
        finally:
            os.unlink(path)

    def test_load_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            config.load("/nonexistent/config.json")

    def test_load_invalid_json_raises(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            f.write("{{{ not json")
            path = f.name
        try:
            with self.assertRaises(Exception):
                config.load(path)
        finally:
            os.unlink(path)


# ── write_default ──────────────────────────────────────────────────────────

class TestWriteDefault(unittest.TestCase):

    def _write_and_load(self):
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False
        ) as f:
            path = f.name
        config.write_default(path)
        return path

    def test_writes_valid_json(self):
        path = self._write_and_load()
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            self.assertIn("system_prompt", data)
            self.assertIn("backend", data)
            self.assertIn("audit", data)
        finally:
            os.unlink(path)

    def test_written_config_is_loadable(self):
        path = self._write_and_load()
        try:
            cfg = config.load(path)
            self.assertEqual(cfg.system_prompt, "You are a helpful assistant.")
        finally:
            os.unlink(path)

    def test_backend_defaults_from_registry(self):
        """write_default() queries registered components — not hardcoded."""
        path = self._write_and_load()
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            self.assertIn("base_url", data["backend"])
        finally:
            os.unlink(path)

    def test_safety_config_present(self):
        path = self._write_and_load()
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            self.assertIn("safety", data)
        finally:
            os.unlink(path)


# ── explain ────────────────────────────────────────────────────────────────

class TestExplain(unittest.TestCase):

    def test_explain_runs_without_error(self):
        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        try:
            with redirect_stdout(f):
                config.explain()
        except Exception as e:
            self.fail(f"explain() raised: {e}")

    def test_explain_covers_key_sections(self):
        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            config.explain()
        output = f.getvalue().lower()
        self.assertIn("backend", output)
        self.assertIn("audit", output)


# ── Backend extension protocol ─────────────────────────────────────────────

class TestBackendProtocol(unittest.TestCase):

    def test_openai_default_config(self):
        from carapex.backends.openai_compatible import OpenAICompatibleBackend
        cfg = OpenAICompatibleBackend.default_config()
        self.assertIn("base_url", cfg)
        self.assertIn("temperature", cfg)

    def test_openai_default_config_includes_type_key(self):
        from carapex.backends.openai_compatible import OpenAICompatibleBackend
        cfg = OpenAICompatibleBackend.default_config()
        self.assertEqual(cfg["type"], "openai_compatible")

    def test_llama_default_config(self):
        from carapex.backends.llama_cpp import LlamaCppBackend
        cfg = LlamaCppBackend.default_config()
        self.assertIn("model_path", cfg)
        self.assertIn("n_threads", cfg)
        self.assertIn("n_gpu_layers", cfg)

    def test_llama_default_config_includes_type_key(self):
        from carapex.backends.llama_cpp import LlamaCppBackend
        cfg = LlamaCppBackend.default_config()
        self.assertEqual(cfg["type"], "llama_cpp")

    def test_openai_from_config_returns_instance(self):
        from carapex.backends.openai_compatible import OpenAICompatibleBackend
        b = OpenAICompatibleBackend.from_config({
            "type": "openai_compatible",
            "base_url": "http://localhost:9999",
            "temperature": 0.3,
        })
        self.assertIsInstance(b, OpenAICompatibleBackend)

    def test_openai_from_config_ignores_type_key(self):
        """'type' is a routing key — must not be passed to BackendConfig constructor."""
        from carapex.backends.openai_compatible import OpenAICompatibleBackend
        try:
            b = OpenAICompatibleBackend.from_config({
                "type": "openai_compatible",
                "base_url": "http://localhost:9999",
            })
            self.assertIsInstance(b, OpenAICompatibleBackend)
        except TypeError as e:
            self.fail(f"from_config() passed 'type' to constructor: {e}")

    def test_openai_repr_is_informative(self):
        from carapex.backends.openai_compatible import OpenAICompatibleBackend
        b = OpenAICompatibleBackend.from_config({
            "base_url": "http://localhost:9999",
        })
        r = repr(b)
        self.assertNotEqual(r, "")
        self.assertIn("localhost", r)


class TestProvideBackend(unittest.TestCase):

    def test_missing_type_raises_configuration_error(self):
        """provide_backend must raise ConfigurationError if 'type' is absent."""
        from carapex.providers import provide_backend
        from carapex.exceptions import ConfigurationError
        with self.assertRaises(ConfigurationError) as ctx:
            provide_backend({"base_url": "http://localhost:8080"})
        self.assertIn("type", str(ctx.exception))

    def test_empty_type_raises_configuration_error(self):
        from carapex.providers import provide_backend
        from carapex.exceptions import ConfigurationError
        with self.assertRaises(ConfigurationError):
            provide_backend({"type": ""})

    def test_unknown_type_raises_configuration_error(self):
        from carapex.providers import provide_backend
        from carapex.exceptions import ConfigurationError
        with self.assertRaises(ConfigurationError) as ctx:
            provide_backend({"type": "nonexistent_backend"})
        self.assertIn("nonexistent_backend", str(ctx.exception))

    def test_openai_compatible_resolves_by_type(self):
        from carapex.providers import provide_backend
        from carapex.backends.openai_compatible import OpenAICompatibleBackend
        b = provide_backend({
            "type": "openai_compatible",
            "base_url": "http://localhost:9999",
        })
        self.assertIsInstance(b, OpenAICompatibleBackend)


# ── Audit backend extension protocol ──────────────────────────────────────

class TestAuditBackendProtocol(unittest.TestCase):

    def test_file_default_config(self):
        from carapex.audit.file_backend import FileAuditBackend
        cfg = FileAuditBackend.default_config()
        self.assertIn("log_path", cfg)

    def test_null_default_config(self):
        from carapex.audit.null_backend import NullAuditBackend
        cfg = NullAuditBackend.default_config()
        self.assertEqual(cfg, {})

    def test_null_from_config(self):
        from carapex.audit.null_backend import NullAuditBackend
        b = NullAuditBackend.from_config({})
        self.assertIsInstance(b, NullAuditBackend)

    def test_file_from_config(self):
        from carapex.audit.file_backend import FileAuditBackend
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            path = f.name
        try:
            b = FileAuditBackend.from_config({"log_path": path})
            self.assertIsInstance(b, FileAuditBackend)
            b.close()
        finally:
            if os.path.exists(path):
                os.unlink(path)


class TestRegistryDuplicateProtection(unittest.TestCase):

    def test_normaliser_duplicate_raises(self):
        from carapex.normaliser import _register
        from carapex.normaliser.base import Decoder

        class _Dup(Decoder):
            name = "url"  # already registered
            def decode(self, text): return text

        with self.assertRaises(RuntimeError) as ctx:
            _register("url", _Dup)
        self.assertIn("url", str(ctx.exception))
        self.assertIn("Duplicate", str(ctx.exception))

    def test_backends_duplicate_raises(self):
        from carapex.backends import _register
        from carapex.backends.base import LLMBackend

        class _Dup(LLMBackend):
            name = "openai_compatible"  # already registered
            def chat(self, s, u, temperature=None): return None
            def health_check(self): return True
            def last_usage(self): return {}
            def close(self): pass

        with self.assertRaises(RuntimeError) as ctx:
            _register("openai_compatible", _Dup)
        self.assertIn("openai_compatible", str(ctx.exception))

    def test_audit_duplicate_raises(self):
        from carapex.audit import _register
        from carapex.audit.base import AuditBackend

        class _Dup(AuditBackend):
            name = "null"  # already registered
            def log(self, e): pass
            def close(self): pass

        with self.assertRaises(RuntimeError) as ctx:
            _register("null", _Dup)
        self.assertIn("null", str(ctx.exception))



if __name__ == '__main__':
    unittest.main(verbosity=2)


# ── OpenAI backend session lifecycle ──────────────────────────────────────

class TestOpenAIBackendLifecycle(unittest.TestCase):

    def test_close_does_not_raise(self):
        from carapex.backends.openai_compatible import OpenAICompatibleBackend
        b = OpenAICompatibleBackend.from_config({
            "type": "openai_compatible",
            "base_url": "http://localhost:9999",
        })
        try:
            b.close()
        except Exception as e:
            self.fail(f"close() raised unexpectedly: {e}")

    def test_session_exists_at_construction(self):
        from carapex.backends.openai_compatible import OpenAICompatibleBackend
        b = OpenAICompatibleBackend.from_config({
            "type": "openai_compatible",
            "base_url": "http://localhost:9999",
        })
        self.assertTrue(hasattr(b, "_session"))
        b.close()


# ── LlamaCpp health_check ──────────────────────────────────────────────────

class TestLlamaCppHealthCheck(unittest.TestCase):

    def _make_backend(self):
        """Construct LlamaCppBackend with a stub model, no real llama_cpp needed."""
        import sys
        import types
        import unittest.mock as mock

        # llama_cpp is imported inside __init__ — inject a stub module
        fake_llama_cpp = types.ModuleType("llama_cpp")
        fake_llama_cpp.Llama = mock.MagicMock(return_value=object())
        sys.modules.setdefault("llama_cpp", fake_llama_cpp)

        from carapex.backends.llama_cpp import LlamaCppBackend, LlamaCppConfig
        return LlamaCppBackend(LlamaCppConfig(model_path="/fake/model.gguf"))

    def test_health_check_true_when_model_loaded(self):
        b = self._make_backend()
        self.assertTrue(b.health_check())

    def test_health_check_false_after_close(self):
        b = self._make_backend()
        b.close()
        self.assertFalse(b.health_check())
