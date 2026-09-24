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
Inject ambient time and weather context as `information` messages.

Mirrors the time/weather injection in `brainforge-webapp`: resolve the user's
location (from `transaction.request_context` or step config), call MCP tools,
and prepend two `information` tuples to `request.history`.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import requests
from neon_data_models.types import LlmMessageRole

from brainforge_inference_utils.mcp import mcp_client, mcp_tool_inner_payload
from brainforge_inference_utils.steps.base import InputStep, RequestContextOption
from brainforge_inference_utils.transaction import UserTransaction

LOG = logging.getLogger(__name__)


class AmbientContextStep(InputStep):
    """
    Prepend time and weather `information` messages to `request.history`.

    Required configuration:

    - `mcp_url`: MCP server URL.
    - `default_location`: Fallback when no location can be resolved.
    - `timeout`: Seconds to wait for MCP calls.

    `transaction.request_context` may include:

    - `location`: Pre-resolved location description (preferred when set).
    - `client_ip`: Client IP used to resolve location via ip-api.com.
    """

    id = "ambient_context"
    request_context_options = (
        RequestContextOption(
            "location",
            description="Pre-resolved location description (preferred when set).",
        ),
        RequestContextOption(
            "client_ip",
            description="Client IP used to resolve location via ip-api.com.",
        ),
    )

    def __init__(self, config: Optional[dict] = None) -> None:
        resolved = dict(config or {})
        resolved.setdefault("mcp_url", "https://mcp.neonaiservices2.com/mcp")
        resolved.setdefault("default_location", "Renton, WA, USA")
        resolved.setdefault("timeout", 5.0)
        super().__init__(resolved)

    def applicable(self, transaction: UserTransaction) -> Optional[str]:
        for key in ("mcp_url", "timeout"):
            if self.config.get(key) in (None, ""):
                return f"{key} not configured"
        return None

    async def process(self, transaction: UserTransaction) -> None:
        location = self._resolve_location(transaction)
        mcp_url = self.config["mcp_url"]
        timeout = float(self.config["timeout"])

        step_context: Dict[str, Any] = {"location": location}
        try:
            time_str, weather_str = await self._fetch_time_and_weather(
                mcp_url, location, timeout=timeout)
        except Exception as exc:
            LOG.warning("Skipping ambient context for %r: %s", location, exc)
            step_context["error"] = str(exc)
            transaction.input_context[self.id] = step_context
            return

        messages = self._build_information_messages(
            location, time_str, weather_str)
        transaction.request.history = (
            list(messages) + list(transaction.request.history))

        step_context.update({
            "time": time_str,
            "weather": weather_str,
            "messages": [content for _, content in messages],
        })
        transaction.input_context[self.id] = step_context

    def _resolve_location(self, transaction: UserTransaction) -> str:
        ctx = transaction.request_context
        location = ctx.get("location")
        if location:
            return str(location)

        client_ip = ctx.get("client_ip")
        if client_ip:
            return self._get_location_from_ip(
                str(client_ip), self.config["default_location"])

        return self.config["default_location"]

    @staticmethod
    def _get_location_from_ip(ip_addr: str, default_location: str) -> str:
        """Resolve a human-readable location string from a client IP address."""
        location = None
        if ip_addr:
            try:
                response = requests.get(
                    f"http://ip-api.com/json/{ip_addr}",
                    timeout=5,
                ).json()
                location = ", ".join(
                    part
                    for part in (
                        response.get("city"),
                        response.get("region"),
                        response.get("country"),
                    )
                    if part
                )
            except Exception:
                location = None
        return location or default_location

    async def _fetch_time_and_weather(
        self,
        mcp_url: str,
        location: str,
        timeout: float,
    ) -> tuple[str, str]:
        async def _fetch() -> tuple[str, str]:
            async with mcp_client(mcp_url) as client:
                time_result = await client.call_tool(
                    "get_formatted_time",
                    {"location": location},
                )
                time_str = json.dumps(mcp_tool_inner_payload(time_result))

                units = "imperial" if "USA" in location else "metric"
                weather_result = await client.call_tool_mcp(
                    "get_weather_forecast",
                    {"location": location, "unit": units},
                )
                weather_str = self._format_weather_payload(
                    mcp_tool_inner_payload(weather_result))
                return time_str, weather_str

        return await asyncio.wait_for(_fetch(), timeout=timeout)

    @staticmethod
    def _format_weather_payload(weather_dict: dict) -> str:
        """Trim weather forecast fields the same way brainforge-webapp does."""
        weather_dict = dict(weather_dict)
        weather_dict.pop("minutely", None)
        if "hourly" in weather_dict:
            weather_dict["hourly"] = weather_dict["hourly"][:24]
        if "daily" in weather_dict:
            weather_dict["daily"] = weather_dict["daily"][:4]
        return json.dumps(weather_dict)

    @staticmethod
    def _build_information_messages(
        location: str,
        time_str: str,
        weather_str: str,
    ) -> List[Tuple[LlmMessageRole, str]]:
        return [
            ("information",
             f"The current time and date is: {time_str}.\n"
             f"The user's location is {location}."),
            ("information",
             "Weather forcast reported in Imperial units for USA locations, "
             f"otherwise Metric:\n{weather_str}\n---"),
        ]

