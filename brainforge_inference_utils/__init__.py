# NEON AI (TM) SOFTWARE, Software Development Kit & Application Framework
# All trademark and other rights reserved by their respective owners
# Copyright 2008-2026 Neongecko.com Inc.
# Contributors: Daniel McKnight, Richard Leeds, Bohdan Mykhailenko, 
# Dmitry Honcharenko, Mike Gray, Clary Gasper, Charlie Bailey, Ryan Venturi,
# Roger Wiebe, Pierce Seigne, Sathvika Nadipalli, Guy Daniels, Elon Gasper,
# Regina Bloomstine, Casimiro Ferreira, Andrii Pernatii, Kirill Hrymailo
# BSD-3 License
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from this
#    software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS  BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS;  OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE,  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
    RequestPipeline,
    ResponsePipeline,
    build_pipelines,
)
from brainforge_inference_utils.registry import (
    REQUEST_STEP_ENTRY_POINT,
    RESPONSE_STEP_ENTRY_POINT,
)
from brainforge_inference_utils.steps import steps_by_id
from brainforge_inference_utils.steps.base import (
    RequestStep,
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
    "RequestStep",
    "ResponseStep",
    "RequestContextOption",
    "steps_by_id",
    # Pipelines
    "RequestPipeline",
    "ResponsePipeline",
    "build_pipelines",
    "REQUEST_STEP_ENTRY_POINT",
    "RESPONSE_STEP_ENTRY_POINT",
    # Config
    "PipelineConfig",
    "StepConfig",
]
