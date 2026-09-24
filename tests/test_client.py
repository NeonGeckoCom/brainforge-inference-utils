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

import unittest
from typing import Optional

from neon_data_models.models.api.llm import LLMResponse

from brainforge_inference_utils.client import BrainForgeClient
from brainforge_inference_utils.config import PipelineConfig
from brainforge_inference_utils.steps.base import (
    InputStep,
    RequestContextOption,
    ResponseStep,
)
from brainforge_inference_utils.transaction import UserTransaction
from helpers import (
    PrefixResponseStep,
    make_request,
    make_response,
)


class RecordInputStep(InputStep):
    id = "record_input"

    async def process(self, transaction: UserTransaction) -> None:
        transaction.input_context[self.id] = self.config["value"]


class CatalogStep(InputStep):
    id = "catalog"

    async def process(self, transaction: UserTransaction) -> None:
        return None

    async def supported_request_context(self, timeout: float = 10.0):
        return [
            RequestContextOption(
                "rag_providers",
                list(self.config.get("providers") or []),
            ),
            RequestContextOption("location", ["Seattle"]),
        ]


class FlagStep(ResponseStep):
    id = "flag"
    request_context_options = (
        RequestContextOption("citations_enabled", (True, False)),
        RequestContextOption("location"),
    )

    async def process(
        self, transaction: UserTransaction
    ) -> Optional[LLMResponse]:
        return None


class RecordOutputStep(ResponseStep):
    id = "record_output"

    async def process(
        self, transaction: UserTransaction
    ) -> Optional[LLMResponse]:
        transaction.output_context[self.id] = self.config["value"]
        return None


class TestBrainForgeClient(unittest.IsolatedAsyncioTestCase):
    def make_client(self):
        config = PipelineConfig.model_validate({
            "input": [{
                "id": "record_input",
                "config": {"value": {"documents": ["one"]}},
            }],
            "response": [
                {
                    "id": "prefix_response",
                    "config": {"prefix": "processed: "},
                },
                {
                    "id": "record_output",
                    "config": {"value": {"citations": [1]}},
                },
            ],
        })
        return BrainForgeClient(
            config,
            {
                RecordInputStep.id: RecordInputStep,
                RecordOutputStep.id: RecordOutputStep,
                PrefixResponseStep.id: PrefixResponseStep,
            },
        )

    async def test_run_input_pipeline_returns_transaction_with_context(self):
        client = self.make_client()
        request = make_request()
        transaction = await client.run_input_pipeline(request)

        self.assertIs(transaction.request, request)
        self.assertIsNone(transaction.response)
        self.assertEqual(
            transaction.input_context,
            {"record_input": {"documents": ["one"]}},
        )
        self.assertEqual(transaction.output_context, {})

    async def test_run_output_pipeline_keeps_separate_step_namespaces(self):
        client = self.make_client()
        transaction = await client.run_input_pipeline(make_request())
        response = make_response("answer", transaction.request)

        result = await client.run_output_pipeline(transaction, response)

        self.assertIs(result, transaction)
        self.assertEqual(transaction.response.response, "processed: answer")
        self.assertEqual(
            transaction.input_context,
            {"record_input": {"documents": ["one"]}},
        )
        self.assertEqual(
            transaction.output_context,
            {"record_output": {"citations": [1]}},
        )

    async def test_supported_request_context_merges_pipeline_steps(self):
        config = PipelineConfig.model_validate({
            "input": [
                {"id": "catalog", "config": {"providers": ["news", "docs"]}},
                {"id": "record_input", "config": {"value": {}}},
                {"id": "catalog", "config": {"providers": ["docs", "web"]}},
            ],
            "response": [{"id": "flag"}],
        })
        client = BrainForgeClient(
            config,
            {
                CatalogStep.id: CatalogStep,
                RecordInputStep.id: RecordInputStep,
                FlagStep.id: FlagStep,
            },
        )

        options = await client.supported_request_context()
        self.assertEqual(
            list(options["rag_providers"].values), ["news", "docs", "web"])
        self.assertEqual(
            list(options["citations_enabled"].values), [True, False])
        self.assertIsNone(options["location"].values)
        self.assertNotIn("record_input", options)
