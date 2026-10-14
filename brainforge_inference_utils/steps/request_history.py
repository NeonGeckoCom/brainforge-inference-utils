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
"""
Guarantee `LLMRequest.history` is valid, including truncation.

`LLMRequest` contract: history excludes the system prompt (on `persona`) and
the incoming query (on `request.query`). Roles are `information` /
`instruction` / `tool` context, then complete `user`/`llm` pairs.

This step is meant to run last on the request pipeline so earlier steps can
prepend context freely. It repairs rather than skipping: unknown or empty
messages are dropped, prior citation markup is stripped, `assistant` becomes
`llm`, a trailing user turn that duplicates `query` is removed, then
conversational turns are truncated to `max_history` complete pairs.
"""

from __future__ import annotations

import re
from typing import Any, List, Optional, Sequence, Tuple

from neon_data_models.types import LlmMessageRole

from brainforge_inference_utils.steps.base import RequestStep, RequestContextOption
from brainforge_inference_utils.transaction import UserTransaction

# Matches `[[n](url)]` written by `ResponseParser.apply_citation_markup`.
_CITATION_MARKUP = re.compile(r"\[\[\d+\]\([^\)]*\)\]")

_CONTEXT_ROLES = {"information", "instruction", "tool"}
_TURN_ROLES = {"user", "llm", "assistant"}
HistoryItem = Tuple[LlmMessageRole, str]


def remove_citation_strings(text: str) -> str:
    """Remove citation strings like `[[1](https://example)]` from text."""
    return _CITATION_MARKUP.sub("", text)


class RequestHistoryStep(RequestStep):
    """
    Normalize, truncate, and order `request.history` into a valid LLM history.

    Optional configuration:

    - `max_history`: User/llm pairs to keep, excluding `request.query`.
      Defaults to `transaction.request.max_history`.
    - `max_messages`: Alternate cap that includes `request.query`
      (webapp `history[-5:]` after appending the query). If set, pair budget
      is `(max_messages - 1) // 2`.

    `transaction.request_context` may include:

    - `conversational`: When false, drop all prior turns (query only).
      Defaults to true.
    """

    id = "request_history"
    request_context_options = (
        RequestContextOption(
            "conversational",
            (True, False),
            "When false, drop all prior turns (query only). Defaults to true.",
        ),
    )

    def applicable(self, transaction: UserTransaction) -> Optional[str]:
        return None

    async def process(self, transaction: UserTransaction) -> None:
        conversational = bool(
            transaction.request_context.get("conversational", True))
        max_history = self._pair_budget(transaction, conversational)

        cleaned = self._clean_messages(
            transaction.request.history, transaction.request.query)
        context_messages, turn_messages = self._split_history(cleaned)
        windowed_turns = self._last_complete_pairs(turn_messages, max_history)
        dropped_turns = len(turn_messages) - len(windowed_turns)

        transaction.request.history = list(context_messages) + list(windowed_turns)
        transaction.request.max_history = max_history
        transaction.request_step_context[self.id] = {
            "conversational": conversational,
            "max_history": max_history,
            "kept_turns": len(windowed_turns),
            "dropped_turns": dropped_turns,
        }

    def _pair_budget(self, transaction: UserTransaction,
                     conversational: bool) -> int:
        if not conversational:
            return 0
        if self.config.get("max_messages") not in (None, ""):
            max_messages = int(self.config["max_messages"])
            return max(0, (max_messages - 1) // 2)
        if self.config.get("max_history") not in (None, ""):
            return max(0, int(self.config["max_history"]))
        return max(0, int(transaction.request.max_history))

    @staticmethod
    def _clean_messages(
        history: Sequence[Tuple[Any, Any]],
        query: str,
    ) -> List[HistoryItem]:
        cleaned: List[HistoryItem] = []
        for role, content in history:
            if not isinstance(content, str):
                continue
            content = remove_citation_strings(content).strip()
            if not content:
                continue
            if role == "assistant":
                role = "llm"
            if role == "system":
                # System prompt belongs on `request.persona`, not history.
                continue
            if role not in _CONTEXT_ROLES and role not in _TURN_ROLES:
                continue
            cleaned.append((role, content))
        query_text = remove_citation_strings(query).strip()
        if (
            cleaned
            and cleaned[-1][0] == "user"
            and cleaned[-1][1] == query_text
        ):
            cleaned.pop()
        return cleaned

    @staticmethod
    def _split_history(
        history: Sequence[HistoryItem],
    ) -> Tuple[List[HistoryItem], List[HistoryItem]]:
        context_messages: List[HistoryItem] = []
        turn_messages: List[HistoryItem] = []
        for role, content in history:
            if role in _TURN_ROLES:
                turn_messages.append((role, content))
            else:
                context_messages.append((role, content))
        return context_messages, turn_messages

    @staticmethod
    def _last_complete_pairs(
        turns: Sequence[HistoryItem],
        max_history: int,
    ) -> List[HistoryItem]:
        if max_history <= 0 or not turns:
            return []
        pairs: List[Tuple[HistoryItem, HistoryItem]] = []
        idx = len(turns) - 1
        while idx >= 1 and len(pairs) < max_history:
            later = turns[idx]
            earlier = turns[idx - 1]
            if later[0] == "llm" and earlier[0] == "user":
                pairs.append((earlier, later))
                idx -= 2
            else:
                idx -= 1
        pairs.reverse()
        windowed: List[HistoryItem] = []
        for earlier, later in pairs:
            windowed.append(earlier)
            windowed.append(later)
        return windowed
