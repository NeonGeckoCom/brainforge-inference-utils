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
