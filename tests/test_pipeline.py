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
