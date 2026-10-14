# NEON AI (TM) SOFTWARE, Software Development Kit & Application Framework
# All trademark and other rights reserved by their respective owners
# Copyright 2008-2026 Neongecko.com Inc.
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
from types import SimpleNamespace
from typing import Optional, Type

from neon_data_models.models.api.llm import (
    LLMPersona,
    LLMRequest,
    LLMResponse,
)

from brainforge_inference_utils.registry import (
    REQUEST_STEP_ENTRY_POINT,
    RESPONSE_STEP_ENTRY_POINT,
)
from brainforge_inference_utils.steps.base import RequestStep, ResponseStep, Step
from brainforge_inference_utils.transaction import UserTransaction


def make_request(**overrides) -> LLMRequest:
    values = dict(
        query="hello",
        history=[("user", "hi"), ("llm", "hey")],
        persona=LLMPersona(name="vanilla"),
        model="demo@1",
    )
    values.update(overrides)
    return LLMRequest(**values)


def make_response(text: str = "ok", request: Optional[LLMRequest] = None,
                  **overrides) -> LLMResponse:
    request = request or make_request()
    values = dict(
        response=text,
        history=list(request.history) + [("user", request.query), ("llm", text)],
    )
    values.update(overrides)
    return LLMResponse(**values)


def make_transaction(response: Optional[LLMResponse] = None,
                     **request_overrides) -> UserTransaction:
    request = make_request(**request_overrides)
    return UserTransaction(request=request, response=response)


class SetMaxHistoryStep(RequestStep):
    """Request step that writes `config["max_history"]` onto the request."""

    id = "test.set_max_history"

    async def process(self, transaction: UserTransaction) -> None:
        transaction.request.max_history = self.config["max_history"]


class SkipUnlessQueryStep(RequestStep):
    """Request step skipped unless `request.query` matches `config["query"]`."""

    id = "test.skip_unless_query"

    def applicable(self, transaction: UserTransaction) -> Optional[str]:
        expected = self.config.get("query")
        if transaction.request.query != expected:
            return f"query is not {expected!r}"
        return None

    async def process(self, transaction: UserTransaction) -> None:
        transaction.request.max_tokens = self.config.get("max_tokens", 256)


class AppendQueryStep(RequestStep):
    """Request step that appends `config["suffix"]` to `request.query`."""

    id = "test.append_query"

    async def process(self, transaction: UserTransaction) -> None:
        transaction.request.query += self.config.get("suffix", "")


class PrefixResponseStep(ResponseStep):
    """Response step that prepends `config["prefix"]` to the response."""

    id = "test.prefix_response"

    def applicable(self, transaction: UserTransaction) -> Optional[str]:
        if transaction.response is None:
            return "no response"
        return None

    async def process(self, transaction: UserTransaction) -> None:
        prefix = self.config.get("prefix", "")
        transaction.response.response = prefix + transaction.response.response


class MutateResponseInPlaceStep(ResponseStep):
    """Response step that appends `config["suffix"]` to the response."""

    id = "test.mutate_in_place"

    async def process(self, transaction: UserTransaction) -> None:
        transaction.response.response += self.config.get("suffix", "!")


class FakeEntryPoint:
    """Stand-in for an importlib entry point that loads a step class."""

    def __init__(self, name, cls, dist="test-plugin"):
        self.name = name
        self._cls = cls
        self.dist = SimpleNamespace(name=dist)

    def load(self):
        return self._cls


def entry_points_for(request_points=(), response_points=()):
    """Return an `entry_points(group=...)` replacement for the two step groups."""

    def fake_entry_points(group):
        if group == REQUEST_STEP_ENTRY_POINT:
            return list(request_points)
        if group == RESPONSE_STEP_ENTRY_POINT:
            return list(response_points)
        return []

    return fake_entry_points


def entry_points_for_steps(
    *step_classes: Type[Step],
    dist: str = "test-plugin",
):
    """
    Advertise `step_classes` on the entry point group matching each base class.
    """
    request_points = []
    response_points = []
    for cls in step_classes:
        point = FakeEntryPoint(cls.id, cls, dist=dist)
        if issubclass(cls, RequestStep):
            request_points.append(point)
        elif issubclass(cls, ResponseStep):
            response_points.append(point)
        else:
            raise TypeError(f"{cls} is not a request or response step")
    return entry_points_for(request_points, response_points)
