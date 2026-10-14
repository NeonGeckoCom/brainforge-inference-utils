# NEON AI (TM) SOFTWARE, Software Development Kit & Application Framework
# All trademark and other rights reserved by their respective owners
# Copyright 2008-2025 Neongecko.com Inc.
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
