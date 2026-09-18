from abc import ABC

import pytest

from brainforge_inference_utils.steps.base import InputStep, ResponseStep
from helpers import (
    AppendQueryStep,
    make_transaction,
)


class TestStepBase:
    def test_config_defaults_to_empty_dict(self):
        step = AppendQueryStep()
        assert step.config == {}

    def test_none_config_defaults_to_empty_dict(self):
        step = AppendQueryStep(None)
        assert step.config == {}

    def test_constructor_copies_config(self):
        payload = {"suffix": "!"}
        step = AppendQueryStep(payload)
        payload["suffix"] = "?"
        assert step.config["suffix"] == "!"

    def test_repr_includes_class_and_id(self):
        step = AppendQueryStep()
        assert repr(step) == "<AppendQueryStep id='append_query'>"

    def test_default_applicable_runs_unconditionally(self):
        class DefaultResponse(ResponseStep):
            id = "default_response"

            async def process(self, transaction):
                return None

        assert AppendQueryStep().applicable(make_transaction()) is None
        assert DefaultResponse().applicable(make_transaction()) is None

    def test_abstract_bases_cannot_be_instantiated(self):
        assert issubclass(InputStep, ABC)
        assert issubclass(ResponseStep, ABC)
        with pytest.raises(TypeError):
            InputStep()
        with pytest.raises(TypeError):
            ResponseStep()
