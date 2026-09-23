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
