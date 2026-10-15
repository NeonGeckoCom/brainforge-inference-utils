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
        assert parsed.request == []
        assert parsed.response == []

    def test_mixed_shorthand_and_mapping_entries(self):
        parsed = PipelineConfig.model_validate({
            "request": [
                "persona",
                {"id": "rag", "config": {"top_k": 10}},
            ],
            "response": [
                {"id": "citations"},
            ],
        })
        assert [entry.id for entry in parsed.request] == ["persona", "rag"]
        assert parsed.request[1].config == {"top_k": 10}
        assert parsed.response[0].id == "citations"

    def test_duplicate_step_ids_are_allowed_with_distinct_configs(self):
        parsed = PipelineConfig.model_validate({
            "request": [
                {"id": "append_query", "config": {"suffix": " A"}},
                {"id": "append_query", "config": {"suffix": " B"}},
            ],
            "response": [
                {"id": "prefix_response", "config": {"prefix": "["}},
                {"id": "prefix_response", "config": {"prefix": "<"}},
            ],
        })
        assert [entry.id for entry in parsed.request] == [
            "append_query",
            "append_query",
        ]
        assert [entry.config for entry in parsed.request] == [
            {"suffix": " A"},
            {"suffix": " B"},
        ]
        assert [entry.config["prefix"] for entry in parsed.response] == [
            "[",
            "<",
        ]

    def test_duplicate_identical_entries_are_allowed(self):
        parsed = PipelineConfig.model_validate({
            "request": ["sanitize", "sanitize"],
        })
        assert [entry.id for entry in parsed.request] == ["sanitize", "sanitize"]

    def test_empty_lists_are_valid(self):
        parsed = PipelineConfig.model_validate({"request": [], "response": []})
        assert parsed.request == []
        assert parsed.response == []

    def test_extra_fields_are_forbidden(self):
        with pytest.raises(ValidationError):
            PipelineConfig.model_validate({"request": [], "chunk": []})

    def test_invalid_entry_type_is_rejected(self):
        with pytest.raises(ValidationError):
            PipelineConfig.model_validate({"request": [123]})
