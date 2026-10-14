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

import unittest
from unittest.mock import patch

import pytest

from brainforge_inference_utils.config import PipelineConfig
from brainforge_inference_utils.pipeline import (
    RequestPipeline,
    ResponsePipeline,
    build_pipelines,
)
from helpers import (
    AppendQueryStep,
    MutateResponseInPlaceStep,
    PrefixResponseStep,
    SetMaxHistoryStep,
    SkipUnlessQueryStep,
    entry_points_for_steps,
    make_response,
    make_transaction,
)


def _patch_test_steps(monkeypatch):
    monkeypatch.setattr(
        "brainforge_inference_utils.registry.entry_points",
        entry_points_for_steps(
            AppendQueryStep,
            SkipUnlessQueryStep,
            PrefixResponseStep,
            MutateResponseInPlaceStep,
        ),
    )


class TestBuildPipelines:
    def test_empty_config_builds_empty_pipelines(self, monkeypatch):
        _patch_test_steps(monkeypatch)
        request_pipeline, response_pipeline = build_pipelines(PipelineConfig())
        assert request_pipeline.steps == []
        assert response_pipeline.steps == []

    def test_unregistered_id_raises(self, monkeypatch):
        _patch_test_steps(monkeypatch)
        with pytest.raises(KeyError, match="test.missing"):
            build_pipelines(PipelineConfig.model_validate({
                "request": ["test.missing"],
            }))

    def test_unknown_id_lists_known_steps(self, monkeypatch):
        _patch_test_steps(monkeypatch)
        with pytest.raises(KeyError, match="test.append_query"):
            build_pipelines(
                PipelineConfig.model_validate({"request": ["test.missing"]}),
            )

    def test_disabled_steps_are_omitted(self, monkeypatch):
        _patch_test_steps(monkeypatch)
        request_pipeline, response_pipeline = build_pipelines(
            PipelineConfig.model_validate({
                "request": [
                    {"id": "test.append_query", "enabled": False,
                     "config": {"suffix": " skipped"}},
                    {"id": "test.append_query", "config": {"suffix": " kept"}},
                ],
                "response": [
                    {"id": "test.prefix_response", "enabled": False,
                     "config": {"prefix": "X"}},
                ],
            }),
        )
        assert len(request_pipeline.steps) == 1
        assert request_pipeline.steps[0].config == {"suffix": " kept"}
        assert response_pipeline.steps == []

    def test_response_step_id_is_unknown_on_the_request_pipeline(
            self, monkeypatch):
        _patch_test_steps(monkeypatch)
        with pytest.raises(KeyError, match="test.prefix_response"):
            build_pipelines(
                PipelineConfig.model_validate({
                    "request": ["test.prefix_response"]}),
            )

    def test_request_step_id_is_unknown_on_the_response_pipeline(
            self, monkeypatch):
        _patch_test_steps(monkeypatch)
        with pytest.raises(KeyError, match="test.append_query"):
            build_pipelines(
                PipelineConfig.model_validate(
                    {"response": ["test.append_query"]}),
            )

    def test_duplicate_ids_instantiate_independent_instances(self, monkeypatch):
        _patch_test_steps(monkeypatch)
        request_pipeline, response_pipeline = build_pipelines(
            PipelineConfig.model_validate({
                "request": [
                    {"id": "test.append_query", "config": {"suffix": " A"}},
                    {"id": "test.append_query", "config": {"suffix": " B"}},
                ],
                "response": [
                    {"id": "test.prefix_response", "config": {"prefix": "["}},
                    {"id": "test.prefix_response", "config": {"prefix": "<"}},
                ],
            }),
        )
        assert len(request_pipeline.steps) == 2
        assert request_pipeline.steps[0] is not request_pipeline.steps[1]
        assert [type(step) for step in request_pipeline.steps] == [
            AppendQueryStep,
            AppendQueryStep,
        ]
        assert [step.config for step in request_pipeline.steps] == [
            {"suffix": " A"},
            {"suffix": " B"},
        ]
        assert [step.config["prefix"] for step in response_pipeline.steps] == [
            "[",
            "<",
        ]


class TestPipelineRun(unittest.IsolatedAsyncioTestCase):
    async def test_request_steps_run_in_declared_order(self):
        pipeline = RequestPipeline([
            AppendQueryStep({"suffix": " A"}),
            AppendQueryStep({"suffix": " B"}),
        ])
        transaction = make_transaction(query="hello")
        result = await pipeline.run(transaction)
        assert result is transaction
        assert transaction.request.query == "hello A B"

    async def test_inapplicable_request_step_is_skipped(self):
        pipeline = RequestPipeline([
            SkipUnlessQueryStep({"query": "other", "max_tokens": 64}),
            SetMaxHistoryStep({"max_history": 7}),
        ])
        transaction = make_transaction(query="hello", max_history=2)
        await pipeline.run(transaction)
        assert transaction.request.max_tokens == 512
        assert transaction.request.max_history == 7

    async def test_response_steps_run_in_declared_order(self):
        pipeline = ResponsePipeline([
            PrefixResponseStep({"prefix": "["}),
            PrefixResponseStep({"prefix": "<"}),
        ])
        transaction = make_transaction()
        transaction.response = make_response("ok", transaction.request)
        await pipeline.run(transaction)
        assert transaction.response.response == "<[ok"

    async def test_response_steps_mutate_the_same_response(self):
        pipeline = ResponsePipeline([
            MutateResponseInPlaceStep({"suffix": "!"}),
            PrefixResponseStep({"prefix": ">"}),
        ])
        transaction = make_transaction()
        transaction.response = make_response("ok", transaction.request)
        await pipeline.run(transaction)
        assert transaction.response.response == ">ok!"

    async def test_inapplicable_response_step_is_skipped(self):
        pipeline = ResponsePipeline([
            PrefixResponseStep({"prefix": "["}),
        ])
        transaction = make_transaction()
        assert transaction.response is None
        await pipeline.run(transaction)
        assert transaction.response is None

    async def test_build_then_run_duplicate_configured_steps(self):
        with patch(
            "brainforge_inference_utils.registry.entry_points",
            entry_points_for_steps(
                AppendQueryStep,
                SkipUnlessQueryStep,
                PrefixResponseStep,
                MutateResponseInPlaceStep,
            ),
        ):
            request_pipeline, response_pipeline = build_pipelines(
                PipelineConfig.model_validate({
                    "request": [
                        {"id": "test.append_query", "config": {"suffix": " A"}},
                        {"id": "test.skip_unless_query",
                         "config": {"query": "nope", "max_tokens": 64}},
                        {"id": "test.append_query", "config": {"suffix": " B"}},
                    ],
                    "response": [
                        {"id": "test.prefix_response", "config": {"prefix": "["}},
                        {"id": "test.mutate_in_place", "config": {"suffix": "!"}},
                        {"id": "test.prefix_response", "config": {"prefix": "<"}},
                    ],
                }),
            )
        transaction = make_transaction(query="hello")
        await request_pipeline.run(transaction)
        assert transaction.request.query == "hello A B"
        assert transaction.request.max_tokens == 512

        transaction.response = make_response("ok", transaction.request)
        await response_pipeline.run(transaction)
        assert transaction.response.response == "<[ok!"
