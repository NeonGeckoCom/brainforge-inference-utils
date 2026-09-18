from typing import Optional

from neon_data_models.models.api.llm import (
    LLMPersona,
    LLMRequest,
    LLMResponse,
)

from brainforge_inference_utils.steps.base import InputStep, ResponseStep
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


class SetMaxHistoryStep(InputStep):
    """Input step that writes `config["max_history"]` onto the request."""

    id = "set_max_history"

    async def process(self, transaction: UserTransaction) -> None:
        transaction.request.max_history = self.config["max_history"]


class SkipUnlessQueryStep(InputStep):
    """Input step skipped unless `request.query` matches `config["query"]`."""

    id = "skip_unless_query"

    def applicable(self, transaction: UserTransaction) -> Optional[str]:
        expected = self.config.get("query")
        if transaction.request.query != expected:
            return f"query is not {expected!r}"
        return None

    async def process(self, transaction: UserTransaction) -> None:
        transaction.request.max_tokens = self.config.get("max_tokens", 256)


class AppendQueryStep(InputStep):
    """Input step that appends `config["suffix"]` to `request.query`."""

    id = "append_query"

    async def process(self, transaction: UserTransaction) -> None:
        transaction.request.query += self.config.get("suffix", "")


class PrefixResponseStep(ResponseStep):
    """Response step that returns a copy with `config["prefix"]` prepended."""

    id = "prefix_response"

    def applicable(self, transaction: UserTransaction) -> Optional[str]:
        if transaction.response is None:
            return "no response"
        return None

    async def process(self, transaction: UserTransaction) -> Optional[LLMResponse]:
        current = transaction.response
        prefix = self.config.get("prefix", "")
        return current.model_copy(update={"response": prefix + current.response})


class MutateResponseInPlaceStep(ResponseStep):
    """Response step that mutates `response.response` and returns None."""

    id = "mutate_in_place"

    async def process(self, transaction: UserTransaction) -> Optional[LLMResponse]:
        transaction.response.response += self.config.get("suffix", "!")
        return None


TEST_STEPS = {
    SetMaxHistoryStep.id: SetMaxHistoryStep,
    SkipUnlessQueryStep.id: SkipUnlessQueryStep,
    AppendQueryStep.id: AppendQueryStep,
    PrefixResponseStep.id: PrefixResponseStep,
    MutateResponseInPlaceStep.id: MutateResponseInPlaceStep,
}
