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

from types import SimpleNamespace

import pytest

from brainforge_inference_utils.config import PipelineConfig
from brainforge_inference_utils.pipeline import build_pipelines
from brainforge_inference_utils.registry import (
    REQUEST_STEP_ENTRY_POINT,
    RESPONSE_STEP_ENTRY_POINT,
    load_entry_point_steps,
)
from brainforge_inference_utils.steps.base import RequestStep, ResponseStep
from helpers import TEST_STEPS


class _EntryPoint:
    def __init__(self, name, cls, dist="acme-plugin"):
        self.name = name
        self._cls = cls
        self.dist = SimpleNamespace(name=dist)

    def load(self):
        return self._cls


class AcmeRequestGuard(RequestStep):
    id = "acme.guard"

    async def process(self, transaction):
        return None


class AcmeResponseGuard(ResponseStep):
    id = "acme.guard"

    async def process(self, transaction):
        return None


class MismatchedId(RequestStep):
    id = "acme.other"

    async def process(self, transaction):
        return None


def _entry_points_for(request_points=(), response_points=()):
    def fake_entry_points(group):
        if group == REQUEST_STEP_ENTRY_POINT:
            return list(request_points)
        if group == RESPONSE_STEP_ENTRY_POINT:
            return list(response_points)
        return []

    return fake_entry_points


def test_same_author_step_name_loads_from_both_groups(monkeypatch):
    monkeypatch.setattr(
        "brainforge_inference_utils.registry.entry_points",
        _entry_points_for(
            [_EntryPoint("acme.guard", AcmeRequestGuard)],
            [_EntryPoint("acme.guard", AcmeResponseGuard, dist="acme-plugin")],
        ),
    )

    request_steps = load_entry_point_steps(
        REQUEST_STEP_ENTRY_POINT, RequestStep)
    response_steps = load_entry_point_steps(
        RESPONSE_STEP_ENTRY_POINT, ResponseStep)

    assert request_steps["acme.guard"] is AcmeRequestGuard
    assert response_steps["acme.guard"] is AcmeResponseGuard


def test_build_pipelines_resolves_plugins_and_builtin_steps(monkeypatch):
    monkeypatch.setattr(
        "brainforge_inference_utils.registry.entry_points",
        _entry_points_for(
            [_EntryPoint("acme.guard", AcmeRequestGuard)],
            [_EntryPoint("acme.guard", AcmeResponseGuard)],
        ),
    )

    request_pipeline, response_pipeline = build_pipelines(
        PipelineConfig.model_validate({
            "request": ["ambient_context", "acme.guard"],
            "response": ["citations", "acme.guard"],
        }))

    assert [step.id for step in request_pipeline.steps] == [
        "ambient_context",
        "acme.guard",
    ]
    assert [step.id for step in response_pipeline.steps] == [
        "citations",
        "acme.guard",
    ]
    assert type(request_pipeline.steps[1]) is AcmeRequestGuard
    assert type(response_pipeline.steps[1]) is AcmeResponseGuard


def test_explicit_step_map_skips_entry_points(monkeypatch):
    def fail_entry_points(group):
        raise AssertionError(f"entry points consulted for {group}")

    monkeypatch.setattr(
        "brainforge_inference_utils.registry.entry_points",
        fail_entry_points,
    )
    request_pipeline, response_pipeline = build_pipelines(
        PipelineConfig.model_validate({
            "request": ["append_query"],
            "response": ["prefix_response"],
        }),
        TEST_STEPS,
    )
    assert [step.id for step in request_pipeline.steps] == ["append_query"]
    assert [step.id for step in response_pipeline.steps] == ["prefix_response"]


def test_entry_point_name_must_be_author_step_name(monkeypatch):
    monkeypatch.setattr(
        "brainforge_inference_utils.registry.entry_points",
        _entry_points_for([_EntryPoint("guardrails", AcmeRequestGuard)]),
    )
    with pytest.raises(ValueError, match="author.step_name"):
        load_entry_point_steps(REQUEST_STEP_ENTRY_POINT, RequestStep)


def test_duplicate_name_in_one_group_names_both_packages(monkeypatch):
    monkeypatch.setattr(
        "brainforge_inference_utils.registry.entry_points",
        _entry_points_for([
            _EntryPoint("acme.guard", AcmeRequestGuard, dist="acme-a"),
            _EntryPoint("acme.guard", AcmeRequestGuard, dist="acme-b"),
        ]),
    )
    with pytest.raises(ValueError, match="acme-a and acme-b"):
        load_entry_point_steps(REQUEST_STEP_ENTRY_POINT, RequestStep)


def test_class_id_must_match_entry_point_name(monkeypatch):
    monkeypatch.setattr(
        "brainforge_inference_utils.registry.entry_points",
        _entry_points_for([_EntryPoint("acme.guard", MismatchedId)]),
    )
    with pytest.raises(ValueError, match="does not match entry point"):
        load_entry_point_steps(REQUEST_STEP_ENTRY_POINT, RequestStep)


def test_response_class_in_request_group_raises(monkeypatch):
    monkeypatch.setattr(
        "brainforge_inference_utils.registry.entry_points",
        _entry_points_for([_EntryPoint("acme.guard", AcmeResponseGuard)]),
    )
    with pytest.raises(TypeError, match="not a RequestStep"):
        load_entry_point_steps(REQUEST_STEP_ENTRY_POINT, RequestStep)


def test_plugin_id_that_collides_with_builtin_raises(monkeypatch):
    monkeypatch.setattr(
        "brainforge_inference_utils.pipeline.steps_by_id",
        {"acme.guard": AcmeRequestGuard},
    )
    monkeypatch.setattr(
        "brainforge_inference_utils.registry.entry_points",
        _entry_points_for([
            _EntryPoint("acme.guard", AcmeRequestGuard, dist="acme-plugin"),
        ]),
    )
    with pytest.raises(ValueError, match="conflicts with a built-in step"):
        build_pipelines(PipelineConfig.model_validate({
            "request": ["acme.guard"],
        }))
