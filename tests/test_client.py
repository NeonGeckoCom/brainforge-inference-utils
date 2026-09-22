import unittest
from typing import Optional

from neon_data_models.models.api.llm import LLMResponse

from brainforge_inference_utils.client import BrainForgeClient
from brainforge_inference_utils.config import PipelineConfig
from brainforge_inference_utils.steps.base import InputStep, ResponseStep
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
