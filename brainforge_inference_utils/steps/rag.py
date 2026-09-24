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
RAG retrieval and citation rewrite.

`RagStep` lists providers and injects retrieved documents on the input
pipeline. `ResponseParser` (id `citations`) runs on the response pipeline:
it filters those documents, builds a citation completion via MCP, POSTs it
to a configured OpenAI-compatible endpoint, and writes the annotated text
back onto the same `UserTransaction`.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import requests
from neon_data_models.models.api.llm import LLMResponse

from brainforge_inference_utils.mcp import mcp_client, mcp_tool_content_parts
from brainforge_inference_utils.steps.base import (
    InputStep,
    RequestContextOption,
    ResponseStep,
)
from brainforge_inference_utils.transaction import UserTransaction

LOG = logging.getLogger(__name__)


class RagStep(InputStep):
    """
    Look up selected RAG providers and prepend retrieved context.

    Required configuration:

    - `mcp_url`: MCP server URL.
    - `top_k`: Default document cap for `lookup_rag`.
    - `timeout`: Seconds to wait for MCP calls.

    `transaction.request_context` may include:

    - `rag_providers`: Provider ids selected by the host (required to run).
    - `rag_top_k`: Optional per-request override of configured `top_k`.
    """

    id = "rag"

    def __init__(self, config: Optional[dict] = None) -> None:
        resolved = dict(config or {})
        resolved.setdefault("mcp_url", "https://mcp.neonaiservices2.com/mcp")
        resolved.setdefault("top_k", 10)
        resolved.setdefault("timeout", 10.0)
        super().__init__(resolved)

    async def supported_request_context(
        self, timeout: float = 10.0
    ) -> List[RequestContextOption]:
        """Advertise selected RAG providers (live catalog) and optional top_k."""
        providers = await self._available_providers(timeout=timeout)
        return [
            RequestContextOption(
                "rag_providers",
                providers,
                "Provider ids selected by the host (required to run).",
            ),
            RequestContextOption(
                "rag_top_k",
                description="Optional per-request override of configured top_k.",
            ),
        ]

    def applicable(self, transaction: UserTransaction) -> Optional[str]:
        if self.config.get("mcp_url") in (None, ""):
            return "mcp_url not configured"
        if self.config.get("top_k") in (None, ""):
            return "top_k not configured"
        providers = transaction.request_context.get("rag_providers") or []
        if not providers:
            return "no rag providers selected"
        return None

    async def _available_providers(self, timeout: float = 10.0) -> List[str]:
        """Return the live provider catalog from the configured MCP server."""
        timeout = float(timeout if timeout is not None else self.config["timeout"])

        async def _fetch() -> List[str]:
            async with mcp_client(self.config["mcp_url"]) as client:
                result = await client.call_tool_mcp("available_contexts", {})
                if getattr(result, "isError", False):
                    err_text = (
                        result.content[0].text
                        if getattr(result, "content", None)
                        else "available_contexts failed"
                    )
                    raise RuntimeError(err_text)
                return [
                    part.text
                    for part in mcp_tool_content_parts(result)
                    if getattr(part, "text", None)
                ]

        return await asyncio.wait_for(_fetch(), timeout=timeout)

    async def process(self, transaction: UserTransaction) -> None:
        providers = [str(p) for p in (
            transaction.request_context.get("rag_providers") or [])]
        top_k = int(
            transaction.request_context.get("rag_top_k")
            if transaction.request_context.get("rag_top_k") not in (None, "")
            else self.config["top_k"]
        )
        timeout = float(self.config["timeout"])
        step_context: Dict[str, Any] = {
            "providers": providers,
            "top_k": top_k,
            "documents": [],
        }
        try:
            documents = await self._lookup_rag(
                transaction.request.query, providers, top_k, timeout)
        except Exception as exc:
            LOG.warning("Skipping RAG lookup for %s: %s", providers, exc)
            step_context["error"] = str(exc)
            transaction.input_context[self.id] = step_context
            return

        step_context["documents"] = documents
        if documents:
            transaction.request.history = (
                [self._information_message(documents)]
                + list(transaction.request.history)
            )
        transaction.input_context[self.id] = step_context

    async def _lookup_rag(
        self,
        query: str,
        providers: List[str],
        top_k: int,
        timeout: float,
    ) -> List[dict]:
        async def _fetch() -> List[dict]:
            async with mcp_client(self.config["mcp_url"]) as client:
                result = await client.call_tool(
                    "lookup_rag",
                    {
                        "message": query,
                        "top_k": top_k,
                        "context_providers": providers,
                    },
                )
                documents = []
                for part in mcp_tool_content_parts(result):
                    documents.append(json.loads(part.text))
                return documents

        return await asyncio.wait_for(_fetch(), timeout=timeout)

    @staticmethod
    def _information_message(documents: List[dict]) -> tuple:
        context_str = "\n".join(
            f"{doc.get('metadata', {}).get('file_name', doc.get('metadata', {}).get('citation_uri'))}: "
            f"{doc.get('text', '')}\n---"
            for doc in documents
        )
        return ("information", context_str)


def _citation_key(document: dict) -> str:
    meta = document.get("metadata") or {}
    return str(
        meta.get("citation")
        or meta.get("citation_uri")
        or meta.get("file_name")
        or ""
    )


def apply_citation_markup(
    response: str, documents: List[dict]
) -> Tuple[str, List[str], str, str]:
    """
    Map raw `[n]` citation tags onto unique documents and markdown links.

    :returns: rewritten text, unique citation labels, compact credits,
        formatted credits for a UI box.
    """
    unique_contexts = list({_citation_key(doc) for doc in documents})
    raw_to_unique_idx = {
        documents.index(doc) + 1: unique_contexts.index(_citation_key(doc)) + 1
        for doc in documents
    }
    rewritten = response.replace("[0]", "")
    for raw_idx, unique_idx in raw_to_unique_idx.items():
        citation_uri = (documents[raw_idx - 1].get("metadata") or {}).get(
            "citation_uri", "")
        citation_string = (
            f"[[0{unique_idx}]({citation_uri})]" if citation_uri
            else f"[{unique_idx}]"
        )
        if f"`[{raw_idx}]`" in rewritten:
            rewritten = rewritten.replace(f"`[{raw_idx}]`", citation_string)
        else:
            rewritten = rewritten.replace(f"[{raw_idx}]", citation_string)
    rewritten = rewritten.replace("[[0", "[[")
    credits = ", ".join(
        f"[{unique_contexts.index(label) + 1}] {label}"
        for label in unique_contexts
    )
    formatted_credits = "\n".join(
        f"[{unique_contexts.index(label) + 1}] {label}"
        for label in unique_contexts
    )
    return rewritten, unique_contexts, credits, formatted_credits


class ResponseParser(ResponseStep):
    """
    Rewrite the generated response with citations using RAG input-context.

    The whole MCP filter → citation completion → markup path runs here so the
    host does not handle a second request/response pair.

    Required configuration:

    - `mcp_url`: MCP server URL (same server as `RagStep`).
    - `openai_url`: OpenAI-compatible chat completions URL (or API base).
    - `timeout`: Seconds to wait for MCP and the completion call.

    Optional configuration:

    - `openai_api_key`: Bearer token for `openai_url`.

    `transaction.request_context` may include:

    - `citations_enabled`: When false, the step is skipped.
    """

    id = "citations"
    request_context_options = (
        RequestContextOption(
            "citations_enabled",
            (True, False),
            "When false, the step is skipped.",
        ),
    )

    def __init__(self, config: Optional[dict] = None) -> None:
        resolved = dict(config or {})
        resolved.setdefault("mcp_url", "https://mcp.neonaiservices2.com/mcp")
        resolved.setdefault("timeout", 30.0)
        super().__init__(resolved)

    def applicable(self, transaction: UserTransaction) -> Optional[str]:
        if transaction.response is None:
            return "no response"
        if self.config.get("mcp_url") in (None, ""):
            return "mcp_url not configured"
        if self.config.get("openai_url") in (None, ""):
            return "openai_url not configured"
        if transaction.request_context.get("citations_enabled") is False:
            return "citations disabled"
        documents = (transaction.input_context.get("rag") or {}).get(
            "documents") or []
        if not documents:
            return "no rag documents"
        return None

    async def process(
        self, transaction: UserTransaction
    ) -> Optional[LLMResponse]:
        documents = list(
            (transaction.input_context.get("rag") or {}).get("documents") or [])
        original = transaction.response.response
        timeout = float(self.config["timeout"])
        try:
            filtered = await self._filter_sources(original, documents, timeout)
            if not filtered:
                transaction.output_context[self.id] = {
                    "documents": [],
                    "credits": "",
                    "formatted_credits": "",
                }
                return None
            payload = await self._build_citation_request(
                original, filtered, timeout)
            raw_cited = await self._complete(
                payload, transaction.request.model, timeout)
            rewritten, unique, credits, formatted = apply_citation_markup(
                raw_cited, filtered)
        except Exception as exc:
            LOG.warning("Skipping citation rewrite: %s", exc)
            transaction.output_context[self.id] = {
                "documents": documents,
                "error": str(exc),
            }
            return None

        transaction.output_context[self.id] = {
            "documents": filtered,
            "unique_citations": unique,
            "credits": credits,
            "formatted_credits": formatted,
        }
        history = list(transaction.response.history)
        if history and history[-1][0] == "llm":
            history[-1] = ("llm", rewritten)
        return transaction.response.model_copy(
            update={"response": rewritten, "history": history})

    async def _filter_sources(
        self, response: str, documents: List[dict], timeout: float
    ) -> List[dict]:
        async def _fetch() -> List[dict]:
            async with mcp_client(self.config["mcp_url"]) as client:
                result = await client.call_tool(
                    "filter_citation_sources",
                    {
                        "generated_response": response,
                        "rag_documents": documents,
                    },
                )
                return [
                    json.loads(part.text)
                    for part in mcp_tool_content_parts(result)
                ]

        return await asyncio.wait_for(_fetch(), timeout=timeout)

    async def _build_citation_request(
        self, response: str, documents: List[dict], timeout: float
    ) -> dict:
        async def _fetch() -> dict:
            async with mcp_client(self.config["mcp_url"]) as client:
                result = await client.call_tool(
                    "build_citation_request",
                    {
                        "original_response": response,
                        "rag_documents": documents,
                    },
                )
                parts = mcp_tool_content_parts(result)
                return json.loads(parts[0].text)

        return await asyncio.wait_for(_fetch(), timeout=timeout)

    async def _complete(
        self, payload: dict, model: str, timeout: float
    ) -> str:
        request = dict(payload)
        request.setdefault("model", model)
        request["stream"] = False
        url = self._completions_url()
        headers = {"Content-Type": "application/json"}
        api_key = self.config.get("openai_api_key")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        def _post() -> dict:
            response = requests.post(
                url, json=request, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.json()

        completion = await asyncio.to_thread(_post)
        return completion["choices"][0]["message"]["content"]

    def _completions_url(self) -> str:
        url = str(self.config["openai_url"]).rstrip("/")
        if url.endswith("/chat/completions") or url.endswith("/completions"):
            return url
        return f"{url}/chat/completions"
