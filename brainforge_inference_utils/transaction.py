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
"""
The object a pipeline operates on.

`UserTransaction` represents one user request/response cycle. Request steps shape
its `request` and write `request_step_context`; generation (performed elsewhere)
populates its `response`; response steps annotate that `response` and write
`response_step_context`.

The request and response are the existing `neon-data-models` models used across
the BrainForge stack (HANA, `brainforge-llm-service`), so this package does not
redefine their shape.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from neon_data_models.models.api.llm import LLMRequest, LLMResponse


class UserTransaction(BaseModel):
    """
    One user request/response cycle as it flows through the pipelines.

    :param request: The request being shaped by the request pipeline.
    :param response: The generated response, populated after generation and
        shaped by the response pipeline. `None` until generation completes.
    :param request_step_context: Transient data written by request steps, keyed by
        request step id.
    :param response_step_context: Transient data written by response steps, keyed by
        response step id.
    """

    request: LLMRequest = Field(
        description="The LLM request shaped by the request pipeline")
    response: Optional[LLMResponse] = Field(
        default=None,
        description="The generated response shaped by the response pipeline; "
                    "None until generation completes")
    request_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Client-provided context read by request steps")
    request_step_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Request-step context keyed by request step id")
    response_step_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Response-step context keyed by response step id")
