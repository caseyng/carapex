"""
carapex — a transparent safety proxy for LLM applications.

Quick start (library):
    from carapex import build
    from carapex.core.config import CarapexConfig

    config = CarapexConfig.load("carapex.yaml")
    cx = build(config)
    result = cx.evaluate([{"role": "user", "content": "Hello"}])
    cx.close()

Quick start (HTTP proxy):
    cx = build(config)
    cx.serve()  # blocks — point your OpenAI client at http://host:8000/v1
"""

from carapex.carapex import Carapex, build
from carapex.core.config import CarapexConfig
from carapex.core.exceptions import (
    CarapexViolation,
    ConfigurationError,
    IntegrityFailure,
    NormalisationError,
    PipelineInternalError,
)
from carapex.core.types import EvaluationResult

__all__ = [
    "Carapex",
    "build",
    "CarapexConfig",
    "EvaluationResult",
    "ConfigurationError",
    "PipelineInternalError",
    "CarapexViolation",
    "IntegrityFailure",
    "NormalisationError",
]

__version__ = "0.13.0"
