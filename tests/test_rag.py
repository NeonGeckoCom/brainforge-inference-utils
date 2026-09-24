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
from unittest.mock import AsyncMock, patch

from brainforge_inference_utils.steps.rag import RagStep
from brainforge_inference_utils.transaction import UserTransaction
from helpers import make_request


STEP_CONFIG = {
    "mcp_url": "http://localhost:8000/mcp",
    "top_k": 10,
    "timeout": 10.0,
}

DOCUMENTS = [
    {
        "text": "chunk one",
        "metadata": {"file_name": "doc-a.md", "citation_uri": "https://a"},
    },
    {
        "text": "chunk two",
        "metadata": {"citation_uri": "https://b"},
    },
]


class TestRagStep(unittest.IsolatedAsyncioTestCase):
    async def test_injects_information_and_stores_documents(self):
        step = RagStep(STEP_CONFIG)
        transaction = UserTransaction(
            request=make_request(query="what?", history=[("user", "hi")]),
            request_context={"rag_providers": ["news"]},
        )

        with patch.object(
            step, "_lookup_rag", new=AsyncMock(return_value=DOCUMENTS)
        ) as lookup:
            await step.process(transaction)

        lookup.assert_awaited_once_with(
            "what?", ["news"], 10, STEP_CONFIG["timeout"])
        self.assertEqual(transaction.request.history[0][0], "information")
        self.assertIn("doc-a.md: chunk one", transaction.request.history[0][1])
        self.assertIn("https://b: chunk two", transaction.request.history[0][1])
        self.assertEqual(transaction.request.history[1], ("user", "hi"))
        self.assertEqual(
            transaction.input_context["rag"]["documents"], DOCUMENTS)
        self.assertEqual(
            transaction.input_context["rag"]["providers"], ["news"])

    async def test_uses_request_top_k_override(self):
        step = RagStep(STEP_CONFIG)
        transaction = UserTransaction(
            request=make_request(),
            request_context={"rag_providers": ["docs"], "rag_top_k": 3},
        )

        with patch.object(
            step, "_lookup_rag", new=AsyncMock(return_value=[])
        ) as lookup:
            await step.process(transaction)

        lookup.assert_awaited_once_with(
            transaction.request.query, ["docs"], 3, STEP_CONFIG["timeout"])
        self.assertEqual(transaction.request.history, make_request().history)
        self.assertEqual(transaction.input_context["rag"]["documents"], [])

    async def test_lookup_failure_leaves_history_unchanged(self):
        step = RagStep(STEP_CONFIG)
        original = [("user", "hi"), ("llm", "hey")]
        transaction = UserTransaction(
            request=make_request(history=list(original)),
            request_context={"rag_providers": ["news"]},
        )

        with patch.object(
            step,
            "_lookup_rag",
            new=AsyncMock(side_effect=TimeoutError("timed out")),
        ):
            await step.process(transaction)

        self.assertEqual(transaction.request.history, original)
        self.assertEqual(
            transaction.input_context["rag"]["error"], "timed out")

    def test_skips_when_no_providers_selected(self):
        step = RagStep(STEP_CONFIG)
        reason = step.applicable(UserTransaction(request=make_request()))
        self.assertEqual(reason, "no rag providers selected")

    async def test_supported_request_context_uses_live_catalog(self):
        step = RagStep(STEP_CONFIG)
        with patch.object(
            step,
            "_available_providers",
            new=AsyncMock(return_value=["news", "docs"]),
        ) as catalog:
            options = {
                option.key: option
                for option in await step.supported_request_context(timeout=2.0)
            }

        catalog.assert_awaited_once_with(timeout=2.0)
        self.assertEqual(list(options["rag_providers"].values), ["news", "docs"])
        self.assertIsNone(options["rag_top_k"].values)

    def test_skips_when_mcp_url_missing(self):
        step = RagStep({"mcp_url": "", "top_k": 10})
        transaction = UserTransaction(
            request=make_request(),
            request_context={"rag_providers": ["news"]},
        )
        self.assertEqual(step.applicable(transaction), "mcp_url not configured")
