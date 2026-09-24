# Copyright (C) 2008-2026 Neongecko.com Inc.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
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
    RequestContextOption,
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
    "RequestContextOption",
    "steps_by_id",
    # Pipelines
    "InputPipeline",
    "ResponsePipeline",
    "build_pipelines",
    # Config
    "PipelineConfig",
    "StepConfig",
]
