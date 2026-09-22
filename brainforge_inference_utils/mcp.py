# NEON AI (TM) SOFTWARE, Software Development Kit & Application Framework
# All trademark and other rights reserved by their respective owners
# Copyright 2008-2025 Neongecko.com Inc.
"""Generic MCP client helpers shared by pipeline steps."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastmcp import Client as MCPClient
from fastmcp.client.transports import StreamableHttpTransport


@asynccontextmanager
async def mcp_client(mcp_url: str) -> AsyncIterator[MCPClient]:
    """
    Open a short-lived MCP client session for one or more tool calls.

    :param mcp_url: MCP server URL from step configuration.
    """
    transport = StreamableHttpTransport(mcp_url)
    async with MCPClient(transport) as client:
        yield client


def mcp_tool_inner_payload(result: Any) -> dict:
    """
    Server tool payload from tools/call structuredContent envelope.

    Newer mcp CallToolResult stores structuredContent as a model field; older
    stacks put it only in model_extra. Fall back to JSON in text content.
    """
    envelope = getattr(result, "structuredContent", None)
    if not isinstance(envelope, dict):
        envelope = (getattr(result, "model_extra", None) or {}).get(
            "structuredContent") or {}
    inner = envelope.get("result") or {}
    payload = inner.get("result")
    if isinstance(payload, dict):
        return payload
    content = getattr(result, "content", None) or []
    if not content:
        return {}
    try:
        parsed = json.loads(content[0].text)
    except (json.JSONDecodeError, TypeError, AttributeError, IndexError):
        return {}
    inner = parsed.get("result") or {}
    payload = inner.get("result")
    return payload if isinstance(payload, dict) else {}


def mcp_tool_content_parts(tool_result: Any) -> list:
    """
    Normalize tools/call handling across fastmcp and raw protocol objects.
    """
    if isinstance(tool_result, list):
        return tool_result
    content = getattr(tool_result, "content", None)
    if isinstance(content, list):
        return content
    raise TypeError(
        f"Unexpected MCP tool result {type(tool_result)!r}; "
        "expected list or .content list")
