# NEON AI (TM) SOFTWARE, Software Development Kit & Application Framework
# All trademark and other rights reserved by their respective owners
# Copyright 2008-2025 Neongecko.com Inc.
"""BrainForge inference pipeline utilities."""

from neon_data_models.models.api.llm import (
    LLMPersona,
    LLMRequest,
    LLMResponse,
)

from brainforge_inference_utils.version import __version__

from brainforge_inference_utils.client import BrainForgeClient
from brainforge_inference_utils.config import PipelineConfig, StepConfig
from brainforge_inference_utils.pipeline import (
    InputPipeline,
    ResponsePipeline,
    build_pipelines,
)
from brainforge_inference_utils.steps import steps_by_id
from brainforge_inference_utils.steps.base import (
    InputStep,
    ResponseStep,
    Step,
)
from brainforge_inference_utils.transaction import UserTransaction

__all__ = [
    "__version__",
    "BrainForgeClient",
    # Request/response models (re-exported from neon-data-models)
    "LLMRequest",
    "LLMResponse",
    "LLMPersona",
    # Transaction
    "UserTransaction",
    # Steps
    "Step",
    "InputStep",
    "ResponseStep",
    "steps_by_id",
    # Pipelines
    "InputPipeline",
    "ResponsePipeline",
    "build_pipelines",
    # Config
    "PipelineConfig",
    "StepConfig",
]
