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

from brainforge_inference_utils.steps.request_history import RequestHistoryStep
from brainforge_inference_utils.transaction import UserTransaction
from helpers import make_request


def _turns(*labels: str):
    history = []
    for label in labels:
        role, _, text = label.partition(":")
        history.append((role, text))
    return history


class TestRequestHistoryStep(unittest.IsolatedAsyncioTestCase):
    async def test_keeps_last_complete_pairs(self):
        step = RequestHistoryStep({"max_messages": 5})
        history = _turns(
            "user:u1", "llm:a1", "user:u2", "llm:a2", "user:u3", "llm:a3")
        transaction = UserTransaction(request=make_request(history=history))

        await step.process(transaction)

        self.assertEqual(transaction.request.history, [
            ("user", "u2"),
            ("llm", "a2"),
            ("user", "u3"),
            ("llm", "a3"),
        ])
        self.assertEqual(transaction.request.max_history, 2)
        self.assertEqual(
            transaction.request_step_context["request_history"]["dropped_turns"], 2)
        self.assertEqual(
            transaction.request_step_context["request_history"]["kept_turns"], 4)

    async def test_preserves_context_messages_outside_the_window(self):
        step = RequestHistoryStep({"max_history": 2})
        history = _turns(
            "information:time",
            "information:weather",
            "user:u1", "llm:a1",
            "user:u2", "llm:a2",
            "user:u3", "llm:a3",
        )
        transaction = UserTransaction(request=make_request(history=history))

        await step.process(transaction)

        self.assertEqual(transaction.request.history[0], ("information", "time"))
        self.assertEqual(
            transaction.request.history[1], ("information", "weather"))
        self.assertEqual(transaction.request.history[2:], [
            ("user", "u2"),
            ("llm", "a2"),
            ("user", "u3"),
            ("llm", "a3"),
        ])

    async def test_non_conversational_drops_prior_turns(self):
        step = RequestHistoryStep({"max_history": 2})
        history = _turns(
            "information:rag",
            "user:u1", "llm:a1", "user:u2", "llm:a2",
        )
        transaction = UserTransaction(
            request=make_request(history=history),
            request_context={"conversational": False},
        )

        await step.process(transaction)

        self.assertEqual(transaction.request.history, [("information", "rag")])
        self.assertEqual(transaction.request.max_history, 0)
        ctx = transaction.request_step_context["request_history"]
        self.assertFalse(ctx["conversational"])
        self.assertEqual(ctx["dropped_turns"], 4)

    async def test_short_history_is_unchanged(self):
        step = RequestHistoryStep({"max_history": 2})
        history = _turns("user:u1", "llm:a1")
        transaction = UserTransaction(request=make_request(history=history))

        await step.process(transaction)

        self.assertEqual(transaction.request.history, history)
        self.assertEqual(
            transaction.request_step_context["request_history"]["dropped_turns"], 0)

    async def test_drops_system_empty_and_duplicate_query(self):
        step = RequestHistoryStep({"max_history": 2})
        transaction = UserTransaction(
            request=make_request(
                query="hello",
                history=[
                    ("information", "ctx"),
                    ("user", ""),
                    ("user", "u1"),
                    ("llm", "a1"),
                    ("user", "hello"),
                ],
            ),
        )
        # `system` is not an LLMRequest history role; earlier steps or hosts
        # may still append it by mutating the list.
        transaction.request.history.insert(0, ("system", "you are a bot"))

        await step.process(transaction)

        self.assertEqual(transaction.request.history, [
            ("information", "ctx"),
            ("user", "u1"),
            ("llm", "a1"),
        ])

    async def test_drops_unpaired_turns_to_keep_complete_pairs(self):
        step = RequestHistoryStep({"max_history": 2})
        history = _turns("llm:orphan", "user:u1", "llm:a1", "user:dangling")
        transaction = UserTransaction(request=make_request(history=history))

        await step.process(transaction)

        self.assertEqual(transaction.request.history, [
            ("user", "u1"),
            ("llm", "a1"),
        ])

    async def test_uses_request_max_history_by_default(self):
        step = RequestHistoryStep()
        history = _turns(
            "user:u1", "llm:a1", "user:u2", "llm:a2", "user:u3", "llm:a3")
        transaction = UserTransaction(
            request=make_request(history=history, max_history=1),
        )

        await step.process(transaction)

        self.assertEqual(transaction.request.history, [
            ("user", "u3"),
            ("llm", "a3"),
        ])
        self.assertEqual(transaction.request.max_history, 1)

    def test_always_applicable(self):
        step = RequestHistoryStep()
        self.assertIsNone(
            step.applicable(UserTransaction(request=make_request())))

    async def test_strips_citation_markup_from_history(self):
        step = RequestHistoryStep({"max_history": 2})
        transaction = UserTransaction(
            request=make_request(history=[
                ("user", "what about that?"),
                ("llm", "See [[1](https://example.com/a)] for details."),
            ]),
        )

        await step.process(transaction)

        self.assertEqual(transaction.request.history, [
            ("user", "what about that?"),
            ("llm", "See  for details."),
        ])

    async def test_supported_request_context_lists_conversational(self):
        step = RequestHistoryStep()
        options = await step.supported_request_context()
        self.assertEqual(len(options), 1)
        self.assertEqual(options[0].key, "conversational")
        self.assertEqual(list(options[0].values), [True, False])
