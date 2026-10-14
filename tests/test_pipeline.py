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

import unittest

import pytest

from brainforge_inference_utils.config import PipelineConfig
from brainforge_inference_utils.pipeline import (
    RequestPipeline,
    ResponsePipeline,
    build_pipelines,
)
from brainforge_inference_utils.steps import steps_by_id
from helpers import (
    AppendQueryStep,
    MutateResponseInPlaceStep,
    PrefixResponseStep,
    SetMaxHistoryStep,
    SkipUnlessQueryStep,
    TEST_STEPS,
    make_response,
    make_transaction,
)


class TestBuildPipelines:
    def test_empty_config_builds_empty_pipelines(self):
        request_pipeline, response_pipeline = build_pipelines(
            PipelineConfig(), TEST_STEPS)
        assert request_pipeline.steps == []
        assert response_pipeline.steps == []

    def test_default_registry_has_no_bundled_steps(self):
        assert steps_by_id == {}
        with pytest.raises(KeyError, match="append_query"):
            build_pipelines(PipelineConfig.model_validate({
                "request": ["append_query"],
            }))

    def test_unknown_id_lists_known_steps(self):
        with pytest.raises(KeyError, match="append_query"):
            build_pipelines(
                PipelineConfig.model_validate({"request": ["nope"]}),
                TEST_STEPS,
            )

    def test_disabled_steps_are_omitted(self):
        request_pipeline, response_pipeline = build_pipelines(
            PipelineConfig.model_validate({
                "request": [
                    {"id": "append_query", "enabled": False,
                     "config": {"suffix": " skipped"}},
                    {"id": "append_query", "config": {"suffix": " kept"}},
                ],
                "response": [
                    {"id": "prefix_response", "enabled": False,
                     "config": {"prefix": "X"}},
                ],
            }),
            TEST_STEPS,
        )
        assert len(request_pipeline.steps) == 1
        assert request_pipeline.steps[0].config == {"suffix": " kept"}
        assert response_pipeline.steps == []

    def test_response_step_in_request_list_raises(self):
        with pytest.raises(TypeError, match="not a RequestStep"):
            build_pipelines(
                PipelineConfig.model_validate({
                    "request": ["prefix_response"]}),
                TEST_STEPS,
            )

    def test_request_step_in_response_list_raises(self):
        with pytest.raises(TypeError, match="not a ResponseStep"):
            build_pipelines(
                PipelineConfig.model_validate(
                    {"response": ["append_query"]}),
                TEST_STEPS,
            )

    def test_duplicate_ids_instantiate_independent_instances(self):
        request_pipeline, response_pipeline = build_pipelines(
            PipelineConfig.model_validate({
                "request": [
                    {"id": "append_query", "config": {"suffix": " A"}},
                    {"id": "append_query", "config": {"suffix": " B"}},
                ],
                "response": [
                    {"id": "prefix_response", "config": {"prefix": "["}},
                    {"id": "prefix_response", "config": {"prefix": "<"}},
                ],
            }),
            TEST_STEPS,
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

    async def test_response_steps_thread_returned_revisions(self):
        pipeline = ResponsePipeline([
            PrefixResponseStep({"prefix": "["}),
            PrefixResponseStep({"prefix": "<"}),
        ])
        transaction = make_transaction()
        transaction.response = make_response("ok", transaction.request)
        await pipeline.run(transaction)
        assert transaction.response.response == "<[ok"

    async def test_response_step_none_keeps_in_place_mutation(self):
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
        request_pipeline, response_pipeline = build_pipelines(
            PipelineConfig.model_validate({
                "request": [
                    {"id": "append_query", "config": {"suffix": " A"}},
                    {"id": "skip_unless_query",
                     "config": {"query": "nope", "max_tokens": 64}},
                    {"id": "append_query", "config": {"suffix": " B"}},
                ],
                "response": [
                    {"id": "prefix_response", "config": {"prefix": "["}},
                    {"id": "mutate_in_place", "config": {"suffix": "!"}},
                    {"id": "prefix_response", "config": {"prefix": "<"}},
                ],
            }),
            TEST_STEPS,
        )
        transaction = make_transaction(query="hello")
        await request_pipeline.run(transaction)
        assert transaction.request.query == "hello A B"
        assert transaction.request.max_tokens == 512

        transaction.response = make_response("ok", transaction.request)
        await response_pipeline.run(transaction)
        assert transaction.response.response == "<[ok!"
