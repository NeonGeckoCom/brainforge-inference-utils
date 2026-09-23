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

import json
import unittest
from types import SimpleNamespace

from brainforge_inference_utils.mcp import (
    mcp_tool_content_parts,
    mcp_tool_inner_payload,
)


class TestMcpToolHelpers(unittest.TestCase):
    def test_inner_payload_from_structured_content(self):
        result = SimpleNamespace(
            structuredContent={"result": {"result": {"now": "Monday"}}},
            content=[],
        )
        self.assertEqual(mcp_tool_inner_payload(result), {"now": "Monday"})

    def test_inner_payload_from_text_content(self):
        payload = {"result": {"result": {"temp": 60}}}
        result = SimpleNamespace(
            structuredContent=None,
            content=[SimpleNamespace(text=json.dumps(payload))],
        )
        self.assertEqual(mcp_tool_inner_payload(result), {"temp": 60})

    def test_content_parts_accepts_list(self):
        parts = [{"type": "text", "text": "hello"}]
        self.assertEqual(mcp_tool_content_parts(parts), parts)

    def test_content_parts_accepts_result_object(self):
        parts = [{"type": "text", "text": "hello"}]
        result = SimpleNamespace(content=parts)
        self.assertEqual(mcp_tool_content_parts(result), parts)
