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

import unittest
from unittest.mock import patch

from brainforge_inference_utils.client import BrainForgeClient
from brainforge_inference_utils.config import PipelineConfig
from brainforge_inference_utils.steps.base import (
    RequestStep,
    RequestContextOption,
    ResponseStep,
)
from brainforge_inference_utils.transaction import UserTransaction
from helpers import (
    PrefixResponseStep,
    entry_points_for_steps,
    make_request,
    make_response,
)


class RecordRequestStep(RequestStep):
    id = "test.record_request"

    async def process(self, transaction: UserTransaction) -> None:
        transaction.request_step_context[self.id] = self.config["value"]


class CatalogStep(RequestStep):
    id = "test.catalog"

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
    id = "test.flag"
    request_context_options = (
        RequestContextOption("citations_enabled", (True, False)),
        RequestContextOption("location"),
    )

    async def process(self, transaction: UserTransaction) -> None:
        return None


class RecordResponseStep(ResponseStep):
    id = "test.record_response"

    async def process(self, transaction: UserTransaction) -> None:
        transaction.response_step_context[self.id] = self.config["value"]


class TestBrainForgeClient(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._entry_points = patch(
            "brainforge_inference_utils.registry.entry_points",
            entry_points_for_steps(
                RecordRequestStep,
                CatalogStep,
                PrefixResponseStep,
                RecordResponseStep,
                FlagStep,
            ),
        )
        self._entry_points.start()

    def tearDown(self):
        self._entry_points.stop()

    def make_client(self):
        config = PipelineConfig.model_validate({
            "request": [{
                "id": "test.record_request",
                "config": {"value": {"documents": ["one"]}},
            }],
            "response": [
                {
                    "id": "test.prefix_response",
                    "config": {"prefix": "processed: "},
                },
                {
                    "id": "test.record_response",
                    "config": {"value": {"citations": [1]}},
                },
            ],
        })
        return BrainForgeClient(config)

    async def test_run_request_pipeline_returns_transaction_with_context(self):
        client = self.make_client()
        request = make_request()
        transaction = await client.run_request_pipeline(request)

        self.assertIs(transaction.request, request)
        self.assertIsNone(transaction.response)
        self.assertEqual(
            transaction.request_step_context,
            {"test.record_request": {"documents": ["one"]}},
        )
        self.assertEqual(transaction.response_step_context, {})

    async def test_run_response_pipeline_keeps_separate_step_namespaces(self):
        client = self.make_client()
        transaction = await client.run_request_pipeline(make_request())
        response = make_response("answer", transaction.request)

        result = await client.run_response_pipeline(transaction, response)

        self.assertIs(result, transaction)
        self.assertEqual(transaction.response.response, "processed: answer")
        self.assertEqual(
            transaction.request_step_context,
            {"test.record_request": {"documents": ["one"]}},
        )
        self.assertEqual(
            transaction.response_step_context,
            {"test.record_response": {"citations": [1]}},
        )

    async def test_supported_request_context_merges_pipeline_steps(self):
        config = PipelineConfig.model_validate({
            "request": [
                {"id": "test.catalog", "config": {"providers": ["news", "docs"]}},
                {"id": "test.record_request", "config": {"value": {}}},
                {"id": "test.catalog", "config": {"providers": ["docs", "web"]}},
            ],
            "response": [{"id": "test.flag"}],
        })
        client = BrainForgeClient(config)

        options = await client.supported_request_context()
        self.assertEqual(
            list(options["rag_providers"].values), ["news", "docs", "web"])
        self.assertEqual(
            list(options["citations_enabled"].values), [True, False])
        self.assertIsNone(options["location"].values)
        self.assertNotIn("test.record_request", options)
