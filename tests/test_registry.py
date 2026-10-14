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

from brainforge_inference_utils.config import PipelineConfig
from brainforge_inference_utils.pipeline import build_pipelines
from brainforge_inference_utils.registry import (
    REQUEST_STEP_ENTRY_POINT,
    RESPONSE_STEP_ENTRY_POINT,
    load_entry_point_steps,
)
from brainforge_inference_utils.steps.base import RequestStep, ResponseStep
from helpers import FakeEntryPoint, entry_points_for


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


def test_same_author_step_name_loads_from_both_groups(monkeypatch):
    monkeypatch.setattr(
        "brainforge_inference_utils.registry.entry_points",
        entry_points_for(
            [FakeEntryPoint("acme.guard", AcmeRequestGuard)],
            [FakeEntryPoint("acme.guard", AcmeResponseGuard, dist="acme-plugin")],
        ),
    )

    request_steps = load_entry_point_steps(
        REQUEST_STEP_ENTRY_POINT, RequestStep)
    response_steps = load_entry_point_steps(
        RESPONSE_STEP_ENTRY_POINT, ResponseStep)

    assert request_steps["acme.guard"] is AcmeRequestGuard
    assert response_steps["acme.guard"] is AcmeResponseGuard


def test_build_pipelines_resolves_entry_point_steps(monkeypatch):
    monkeypatch.setattr(
        "brainforge_inference_utils.registry.entry_points",
        entry_points_for(
            [FakeEntryPoint("acme.guard", AcmeRequestGuard)],
            [FakeEntryPoint("acme.guard", AcmeResponseGuard)],
        ),
    )

    request_pipeline, response_pipeline = build_pipelines(
        PipelineConfig.model_validate({
            "request": ["acme.guard"],
            "response": ["acme.guard"],
        }))

    assert [step.id for step in request_pipeline.steps] == ["acme.guard"]
    assert [step.id for step in response_pipeline.steps] == ["acme.guard"]
    assert type(request_pipeline.steps[0]) is AcmeRequestGuard
    assert type(response_pipeline.steps[0]) is AcmeResponseGuard


def test_entry_point_name_must_be_author_step_name(monkeypatch):
    monkeypatch.setattr(
        "brainforge_inference_utils.registry.entry_points",
        entry_points_for([FakeEntryPoint("guardrails", AcmeRequestGuard)]),
    )
    with pytest.raises(ValueError, match="author.step_name"):
        load_entry_point_steps(REQUEST_STEP_ENTRY_POINT, RequestStep)


def test_duplicate_name_in_one_group_names_both_packages(monkeypatch):
    monkeypatch.setattr(
        "brainforge_inference_utils.registry.entry_points",
        entry_points_for([
            FakeEntryPoint("acme.guard", AcmeRequestGuard, dist="acme-a"),
            FakeEntryPoint("acme.guard", AcmeRequestGuard, dist="acme-b"),
        ]),
    )
    with pytest.raises(ValueError, match="acme-a and acme-b"):
        load_entry_point_steps(REQUEST_STEP_ENTRY_POINT, RequestStep)


def test_class_id_must_match_entry_point_name(monkeypatch):
    monkeypatch.setattr(
        "brainforge_inference_utils.registry.entry_points",
        entry_points_for([FakeEntryPoint("acme.guard", MismatchedId)]),
    )
    with pytest.raises(ValueError, match="does not match entry point"):
        load_entry_point_steps(REQUEST_STEP_ENTRY_POINT, RequestStep)


def test_response_class_in_request_group_raises(monkeypatch):
    monkeypatch.setattr(
        "brainforge_inference_utils.registry.entry_points",
        entry_points_for([FakeEntryPoint("acme.guard", AcmeResponseGuard)]),
    )
    with pytest.raises(TypeError, match="not a RequestStep"):
        load_entry_point_steps(REQUEST_STEP_ENTRY_POINT, RequestStep)

