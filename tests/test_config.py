import pytest
from pydantic import ValidationError

from brainforge_inference_utils.config import PipelineConfig, StepConfig


class TestStepConfig:
    def test_shorthand_string_becomes_id(self):
        parsed = StepConfig.model_validate("persona")
        assert parsed.id == "persona"
        assert parsed.enabled is True
        assert parsed.config == {}

    def test_explicit_mapping(self):
        parsed = StepConfig.model_validate({
            "id": "rag",
            "enabled": False,
            "config": {"top_k": 10},
        })
        assert parsed.id == "rag"
        assert parsed.enabled is False
        assert parsed.config == {"top_k": 10}

    def test_missing_id_is_invalid(self):
        with pytest.raises(ValidationError):
            StepConfig.model_validate({"config": {"top_k": 1}})

    def test_extra_fields_are_forbidden(self):
        with pytest.raises(ValidationError):
            StepConfig.model_validate({"id": "persona", "unknown": True})

    def test_constructor_copies_config(self):
        payload = {"id": "rag", "config": {"top_k": 3}}
        parsed = StepConfig.model_validate(payload)
        payload["config"]["top_k"] = 99
        assert parsed.config["top_k"] == 3


class TestPipelineConfig:
    def test_defaults_to_empty_lists(self):
        parsed = PipelineConfig()
        assert parsed.input == []
        assert parsed.response == []

    def test_mixed_shorthand_and_mapping_entries(self):
        parsed = PipelineConfig.model_validate({
            "input": [
                "persona",
                {"id": "rag", "config": {"top_k": 10}},
            ],
            "response": [
                {"id": "citations"},
            ],
        })
        assert [entry.id for entry in parsed.input] == ["persona", "rag"]
        assert parsed.input[1].config == {"top_k": 10}
        assert parsed.response[0].id == "citations"

    def test_duplicate_step_ids_are_allowed_with_distinct_configs(self):
        parsed = PipelineConfig.model_validate({
            "input": [
                {"id": "append_query", "config": {"suffix": " A"}},
                {"id": "append_query", "config": {"suffix": " B"}},
            ],
            "response": [
                {"id": "prefix_response", "config": {"prefix": "["}},
                {"id": "prefix_response", "config": {"prefix": "<"}},
            ],
        })
        assert [entry.id for entry in parsed.input] == [
            "append_query",
            "append_query",
        ]
        assert [entry.config for entry in parsed.input] == [
            {"suffix": " A"},
            {"suffix": " B"},
        ]
        assert [entry.config["prefix"] for entry in parsed.response] == [
            "[",
            "<",
        ]

    def test_duplicate_identical_entries_are_allowed(self):
        parsed = PipelineConfig.model_validate({
            "input": ["sanitize", "sanitize"],
        })
        assert [entry.id for entry in parsed.input] == ["sanitize", "sanitize"]

    def test_empty_lists_are_valid(self):
        parsed = PipelineConfig.model_validate({"input": [], "response": []})
        assert parsed.input == []
        assert parsed.response == []

    def test_extra_fields_are_forbidden(self):
        with pytest.raises(ValidationError):
            PipelineConfig.model_validate({"input": [], "chunk": []})

    def test_invalid_entry_type_is_rejected(self):
        with pytest.raises(ValidationError):
            PipelineConfig.model_validate({"input": [123]})
