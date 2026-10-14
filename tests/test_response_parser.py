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

from brainforge_inference_utils.steps.rag import (
    ResponseParser,
    apply_citation_markup,
)
from brainforge_inference_utils.transaction import UserTransaction
from helpers import make_request, make_response


STEP_CONFIG = {
    "mcp_url": "http://localhost:8000/mcp",
    "openai_url": "https://hana.example/brainforge/openai/chat/completions",
    "openai_api_key": "token",
    "timeout": 30.0,
}

DOCUMENTS = [
    {"text": "one", "metadata": {"citation": "Doc A", "citation_uri": "https://a"}},
    {"text": "two", "metadata": {"citation": "Doc A", "citation_uri": "https://a"}},
    {"text": "three", "metadata": {"citation": "Doc B", "citation_uri": "https://b"}},
]


class TestApplyCitationMarkup(unittest.TestCase):
    def test_rewrites_raw_indexes_to_unique_links(self):
        rewritten, unique, credits, formatted = apply_citation_markup(
            "See [1] and [3] and [0].", DOCUMENTS)
        self.assertNotIn("[0]", rewritten)
        self.assertIn("](https://", rewritten)
        self.assertEqual(len(unique), 2)
        self.assertIn("Doc A", credits)
        self.assertIn("\n", formatted)


class TestResponseParser(unittest.IsolatedAsyncioTestCase):
    def _transaction(self, text="See the docs.", **request_context):
        request = make_request()
        transaction = UserTransaction(
            request=request,
            response=make_response(text, request),
            request_context=request_context,
            request_step_context={"rag": {"documents": DOCUMENTS}},
        )
        return transaction

    async def test_rewrites_response_via_openai_endpoint(self):
        step = ResponseParser(STEP_CONFIG)
        transaction = self._transaction()

        with patch.object(
            step, "_filter_sources", new=AsyncMock(return_value=DOCUMENTS)
        ), patch.object(
            step, "_build_citation_request",
            new=AsyncMock(return_value={"messages": []}),
        ) as build, patch.object(
            step, "_complete",
            new=AsyncMock(return_value="See [1] and [3]."),
        ) as complete:
            revised = await step.process(transaction)

        build.assert_awaited_once()
        complete.assert_awaited_once_with(
            {"messages": []}, transaction.request.model, 30.0)
        self.assertIsNotNone(revised)
        self.assertNotEqual(revised.response, "See the docs.")
        self.assertEqual(revised.history[-1][0], "llm")
        self.assertEqual(revised.history[-1][1], revised.response)
        self.assertIn("Doc A", transaction.response_step_context["citations"]["credits"])

    async def test_skips_without_rag_documents(self):
        step = ResponseParser(STEP_CONFIG)
        transaction = UserTransaction(
            request=make_request(),
            response=make_response(),
        )
        self.assertEqual(step.applicable(transaction), "no rag documents")

    async def test_skips_when_openai_url_missing(self):
        step = ResponseParser({"mcp_url": "http://localhost:8000/mcp"})
        transaction = self._transaction()
        self.assertEqual(step.applicable(transaction), "openai_url not configured")

    async def test_skips_when_citations_disabled(self):
        step = ResponseParser(STEP_CONFIG)
        transaction = self._transaction(citations_enabled=False)
        self.assertEqual(step.applicable(transaction), "citations disabled")

    async def test_completion_failure_leaves_response(self):
        step = ResponseParser(STEP_CONFIG)
        transaction = self._transaction()
        original = transaction.response.response

        with patch.object(
            step, "_filter_sources", new=AsyncMock(return_value=DOCUMENTS)
        ), patch.object(
            step, "_build_citation_request",
            new=AsyncMock(return_value={"messages": []}),
        ), patch.object(
            step, "_complete",
            new=AsyncMock(side_effect=RuntimeError("nope")),
        ):
            revised = await step.process(transaction)

        self.assertIsNone(revised)
        self.assertEqual(transaction.response.response, original)
        self.assertEqual(
            transaction.response_step_context["citations"]["error"], "nope")

    def test_appends_chat_completions_to_base_url(self):
        step = ResponseParser({
            **STEP_CONFIG,
            "openai_url": "https://api.openai.com/v1",
        })
        self.assertEqual(
            step._completions_url(),
            "https://api.openai.com/v1/chat/completions",
        )

    async def test_supported_request_context_lists_citations_enabled(self):
        step = ResponseParser(STEP_CONFIG)
        options = await step.supported_request_context()
        self.assertEqual(len(options), 1)
        self.assertEqual(options[0].key, "citations_enabled")
        self.assertEqual(list(options[0].values), [True, False])
