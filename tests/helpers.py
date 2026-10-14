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
from typing import Optional

from neon_data_models.models.api.llm import (
    LLMPersona,
    LLMRequest,
    LLMResponse,
)

from brainforge_inference_utils.steps.base import RequestStep, ResponseStep
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

    id = "set_max_history"

    async def process(self, transaction: UserTransaction) -> None:
        transaction.request.max_history = self.config["max_history"]


class SkipUnlessQueryStep(RequestStep):
    """Request step skipped unless `request.query` matches `config["query"]`."""

    id = "skip_unless_query"

    def applicable(self, transaction: UserTransaction) -> Optional[str]:
        expected = self.config.get("query")
        if transaction.request.query != expected:
            return f"query is not {expected!r}"
        return None

    async def process(self, transaction: UserTransaction) -> None:
        transaction.request.max_tokens = self.config.get("max_tokens", 256)


class AppendQueryStep(RequestStep):
    """Request step that appends `config["suffix"]` to `request.query`."""

    id = "append_query"

    async def process(self, transaction: UserTransaction) -> None:
        transaction.request.query += self.config.get("suffix", "")


class PrefixResponseStep(ResponseStep):
    """Response step that prepends `config["prefix"]` to the response."""

    id = "prefix_response"

    def applicable(self, transaction: UserTransaction) -> Optional[str]:
        if transaction.response is None:
            return "no response"
        return None

    async def process(self, transaction: UserTransaction) -> None:
        prefix = self.config.get("prefix", "")
        transaction.response.response = prefix + transaction.response.response


class MutateResponseInPlaceStep(ResponseStep):
    """Response step that appends `config["suffix"]` to the response."""

    id = "mutate_in_place"

    async def process(self, transaction: UserTransaction) -> None:
        transaction.response.response += self.config.get("suffix", "!")


TEST_STEPS = {
    SetMaxHistoryStep.id: SetMaxHistoryStep,
    SkipUnlessQueryStep.id: SkipUnlessQueryStep,
    AppendQueryStep.id: AppendQueryStep,
    PrefixResponseStep.id: PrefixResponseStep,
    MutateResponseInPlaceStep.id: MutateResponseInPlaceStep,
}
