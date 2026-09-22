# NEON AI (TM) SOFTWARE, Software Development Kit & Application Framework
# All trademark and other rights reserved by their respective owners
# Copyright 2008-2025 Neongecko.com Inc.
"""Configured runner for the input and response pipelines.

`BrainForgeClient` does not call an LLM. The host runs inference between
`run_input_pipeline` and `run_output_pipeline`, and is responsible for keeping
the returned `UserTransaction` until the response is ready.
"""

from __future__ import annotations

from typing import Mapping, Optional, Type

from neon_data_models.models.api.llm import LLMRequest, LLMResponse

from brainforge_inference_utils.config import PipelineConfig
from brainforge_inference_utils.pipeline import build_pipelines
from brainforge_inference_utils.steps.base import Step
from brainforge_inference_utils.transaction import UserTransaction


class BrainForgeClient:
    """Build pipelines from config and run them around host-owned inference."""

    def __init__(
        self,
        config: PipelineConfig,
        steps: Optional[Mapping[str, Type[Step]]] = None,
    ):
        self.config = config
        self.input_pipeline, self.response_pipeline = build_pipelines(
            config, steps)

    async def run_input_pipeline(
        self, request: LLMRequest
    ) -> UserTransaction:
        """
        Shape `request` and collect step context on a new transaction.

        The caller must hold the returned transaction until
        `run_output_pipeline` (or drop it if the response pipeline is unused).
        """
        transaction = UserTransaction(request=request)
        return await self.input_pipeline.run(transaction)

    async def run_output_pipeline(
        self, transaction: UserTransaction, response: LLMResponse
    ) -> UserTransaction:
        """
        Attach `response` to `transaction` and run the response pipeline.

        Input-step context is on `transaction.input_context`; response steps
        write to `transaction.output_context`. The caller must pass through
        the transaction from `run_input_pipeline`.
        """
        transaction.response = response
        return await self.response_pipeline.run(transaction)
