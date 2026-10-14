# NEON AI (TM) SOFTWARE, Software Development Kit & Application Framework
# All trademark and other rights reserved by their respective owners
# Copyright 2008-2025 Neongecko.com Inc.
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
"""Configured runner for the request and response pipelines.

`BrainForgeClient` does not call an LLM. The host runs inference between
`run_request_pipeline` and `run_response_pipeline`, and is responsible for keeping
the returned `UserTransaction` until the response is ready.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Type

from neon_data_models.models.api.llm import LLMRequest, LLMResponse

from brainforge_inference_utils.config import PipelineConfig
from brainforge_inference_utils.pipeline import build_pipelines
from brainforge_inference_utils.steps.base import RequestContextOption, Step
from brainforge_inference_utils.transaction import UserTransaction


class BrainForgeClient:
    """Build pipelines from config and run them around host-owned inference."""

    def __init__(
        self,
        config: PipelineConfig,
        steps: Optional[Mapping[str, Type[Step]]] = None,
    ):
        self.config = config
        self.request_pipeline, self.response_pipeline = build_pipelines(
            config, steps)

    async def supported_request_context(
        self, timeout: float = 10.0
    ) -> Dict[str, RequestContextOption]:
        """
        Collect `request_context` keys and allowed values from the pipeline.

        The host uses this to populate UI; it does not call MCP itself.
        Duplicate keys across steps are merged: enumerated values are unioned
        in first-seen order, and if any declaration is unconstrained
        (`values is None`) the merged option is unconstrained.
        """
        merged: Dict[str, RequestContextOption] = {}
        for step in (*self.request_pipeline.steps, *self.response_pipeline.steps):
            for option in await step.supported_request_context(timeout=timeout):
                _merge_request_context_option(merged, option)
        return merged

    async def run_request_pipeline(
        self, request: LLMRequest
    ) -> UserTransaction:
        """
        Shape `request` and collect step context on a new transaction.

        The caller must hold the returned transaction until
        `run_response_pipeline` (or drop it if the response pipeline is unused).
        """
        transaction = UserTransaction(request=request)
        return await self.request_pipeline.run(transaction)

    async def run_response_pipeline(
        self, transaction: UserTransaction, response: LLMResponse
    ) -> UserTransaction:
        """
        Attach `response` to `transaction` and run the response pipeline.

        Request-step context is on `transaction.request_step_context`; response steps
        write to `transaction.response_step_context`. The caller must pass through
        the transaction from `run_request_pipeline`.
        """
        transaction.response = response
        return await self.response_pipeline.run(transaction)


def _merge_request_context_option(
    merged: Dict[str, RequestContextOption],
    option: RequestContextOption,
) -> None:
    prior = merged.get(option.key)
    if prior is None:
        merged[option.key] = option
        return
    description = prior.description or option.description
    if prior.values is None or option.values is None:
        merged[option.key] = RequestContextOption(
            option.key, None, description)
        return
    values = list(prior.values)
    for value in option.values:
        if value not in values:
            values.append(value)
    merged[option.key] = RequestContextOption(
        option.key, values, description)
