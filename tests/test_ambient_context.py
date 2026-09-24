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
from unittest.mock import AsyncMock, patch

from brainforge_inference_utils.steps.ambient_context import AmbientContextStep
from brainforge_inference_utils.transaction import UserTransaction
from helpers import make_request


STEP_CONFIG = {
    "mcp_url": "http://localhost:8000/mcp",
    "default_location": "Renton, WA, USA",
    "timeout": 5.0,
}


class TestFormatWeatherPayload(unittest.TestCase):
    def test_trims_forecast_fields(self):
        payload = {
            "minutely": [{"time": 1}],
            "hourly": list(range(30)),
            "daily": list(range(10)),
        }
        formatted = AmbientContextStep._format_weather_payload(payload)
        trimmed = json.loads(formatted)
        self.assertNotIn("minutely", trimmed)
        self.assertEqual(len(trimmed["hourly"]), 24)
        self.assertEqual(len(trimmed["daily"]), 4)


class TestAmbientContextStep(unittest.IsolatedAsyncioTestCase):
    async def test_prepends_information_to_request_history(self):
        step = AmbientContextStep(STEP_CONFIG)
        transaction = UserTransaction(
            request=make_request(history=[("user", "hi"), ("llm", "hey")]),
            request_context={"location": "Seattle, WA, USA"},
        )

        with patch.object(
            step,
            "_fetch_time_and_weather",
            new=AsyncMock(return_value=('{"now": "Monday"}', '{"temp": 60}')),
        ) as fetch:
            await step.process(transaction)

        fetch.assert_awaited_once_with(
            STEP_CONFIG["mcp_url"], "Seattle, WA, USA", timeout=5.0)
        self.assertEqual(len(transaction.request.history), 4)
        self.assertEqual(transaction.request.history[0][0], "information")
        self.assertIn("Monday", transaction.request.history[0][1])
        self.assertIn("Seattle, WA, USA", transaction.request.history[0][1])
        self.assertEqual(transaction.request.history[1][0], "information")
        self.assertIn("Weather forcast", transaction.request.history[1][1])
        self.assertIn('{"temp": 60}', transaction.request.history[1][1])
        self.assertEqual(transaction.request.history[2:], [
            ("user", "hi"),
            ("llm", "hey"),
        ])
        self.assertEqual(
            transaction.input_context["ambient_context"]["location"],
            "Seattle, WA, USA",
        )

    async def test_resolves_location_from_client_ip(self):
        step = AmbientContextStep(STEP_CONFIG)
        transaction = UserTransaction(
            request=make_request(),
            request_context={"client_ip": "203.0.113.10"},
        )

        with patch.object(
            step,
            "_get_location_from_ip",
            return_value="Portland, OR, USA",
        ) as get_location, patch.object(
            step,
            "_fetch_time_and_weather",
            new=AsyncMock(return_value=('{"now": "Tuesday"}', '{"temp": 55}')),
        ) as fetch:
            await step.process(transaction)

        get_location.assert_called_once_with(
            "203.0.113.10", STEP_CONFIG["default_location"])
        fetch.assert_awaited_once_with(
            STEP_CONFIG["mcp_url"], "Portland, OR, USA", timeout=5.0)

    async def test_uses_default_location_when_request_context_empty(self):
        step = AmbientContextStep(STEP_CONFIG)
        transaction = UserTransaction(request=make_request())

        with patch.object(
            step,
            "_fetch_time_and_weather",
            new=AsyncMock(return_value=('{"now": "Wednesday"}', '{"temp": 50}')),
        ) as fetch:
            await step.process(transaction)

        fetch.assert_awaited_once_with(
            STEP_CONFIG["mcp_url"],
            STEP_CONFIG["default_location"],
            timeout=5.0,
        )

    async def test_uses_configured_mcp_url(self):
        config = {
            **STEP_CONFIG,
            "mcp_url": "https://mcp.example.com/mcp",
            "timeout": 2.5,
        }
        step = AmbientContextStep(config)
        transaction = UserTransaction(
            request=make_request(),
            request_context={"location": "Austin, TX, USA"},
        )

        with patch.object(
            step,
            "_fetch_time_and_weather",
            new=AsyncMock(return_value=('{"now": "Thursday"}', '{"temp": 70}')),
        ) as fetch:
            await step.process(transaction)

        fetch.assert_awaited_once_with(
            "https://mcp.example.com/mcp", "Austin, TX, USA", timeout=2.5)

    async def test_mcp_failure_leaves_history_unchanged(self):
        step = AmbientContextStep(STEP_CONFIG)
        original_history = [("user", "hi"), ("llm", "hey")]
        transaction = UserTransaction(
            request=make_request(history=list(original_history)),
            request_context={"location": "Denver, CO, USA"},
        )

        with patch.object(
            step,
            "_fetch_time_and_weather",
            new=AsyncMock(side_effect=TimeoutError("timed out")),
        ):
            await step.process(transaction)

        self.assertEqual(transaction.request.history, original_history)
        self.assertEqual(
            transaction.input_context["ambient_context"],
            {"location": "Denver, CO, USA", "error": "timed out"},
        )

    async def test_prefers_request_context_location_over_client_ip(self):
        step = AmbientContextStep(STEP_CONFIG)
        transaction = UserTransaction(
            request=make_request(),
            request_context={
                "location": "Chicago, IL, USA",
                "client_ip": "203.0.113.10",
            },
        )

        with patch.object(
            step,
            "_get_location_from_ip",
        ) as get_location, patch.object(
            step,
            "_fetch_time_and_weather",
            new=AsyncMock(return_value=('{"now": "Friday"}', '{"temp": 40}')),
        ) as fetch:
            await step.process(transaction)

        get_location.assert_not_called()
        fetch.assert_awaited_once_with(
            STEP_CONFIG["mcp_url"], "Chicago, IL, USA", timeout=5.0)

    def test_skips_when_timeout_missing(self):
        step = AmbientContextStep({
            "mcp_url": "http://localhost:8000/mcp",
            "default_location": "Renton, WA, USA",
            "timeout": "",
        })
        reason = step.applicable(UserTransaction(request=make_request()))
        self.assertEqual(reason, "timeout not configured")

    def test_uses_default_location_from_step_config(self):
        step = AmbientContextStep({"mcp_url": "http://localhost:8000/mcp"})
        transaction = UserTransaction(request=make_request())
        self.assertEqual(
            step._resolve_location(transaction),
            "Renton, WA, USA",
        )

    async def test_supported_request_context_is_unconstrained(self):
        step = AmbientContextStep(STEP_CONFIG)
        options = await step.supported_request_context()
        self.assertEqual([option.key for option in options], ["location", "client_ip"])
        self.assertTrue(all(option.values is None for option in options))
