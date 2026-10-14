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
"""
The request and response pipelines.

- `RequestPipeline` runs ordered `RequestStep`s over a `UserTransaction` to shape
  its request before generation.
- `ResponsePipeline` runs ordered `ResponseStep`s over a `UserTransaction` to
  annotate or rewrite its response after generation.

Generation itself is out of scope here: a caller runs the request pipeline,
performs generation separately to populate `transaction.response`, then runs the
response pipeline. Steps that are inapplicable to a given transaction are
skipped (with a logged reason) rather than failing the pipeline.
"""

from __future__ import annotations

import logging
from typing import List, Mapping, Sequence, Tuple, Type

from brainforge_inference_utils.config import PipelineConfig
from brainforge_inference_utils.registry import (
    REQUEST_STEP_ENTRY_POINT,
    RESPONSE_STEP_ENTRY_POINT,
    load_entry_point_steps,
)
from brainforge_inference_utils.steps.base import (
    RequestStep,
    ResponseStep,
    Step,
)
from brainforge_inference_utils.transaction import UserTransaction

LOG = logging.getLogger(__name__)


class RequestPipeline:
    """
    Runs `RequestStep`s in order over a transaction's request.

    :param steps: Ordered request steps.
    """

    def __init__(self, steps: Sequence[RequestStep]):
        self.steps: List[RequestStep] = list(steps)

    async def run(self, transaction: UserTransaction) -> UserTransaction:
        """
        Run every applicable request step, mutating the transaction in place.

        :param transaction: The transaction whose request to shape.
        :returns: The same `transaction`.
        """
        for step in self.steps:
            reason = step.applicable(transaction)
            if reason is not None:
                LOG.debug("Skipping request step %r: %s", step.id, reason)
                continue
            await step.process(transaction)
        return transaction


class ResponsePipeline:
    """
    Runs `ResponseStep`s in order over a transaction's response.

    :param steps: Ordered response steps.
    """

    def __init__(self, steps: Sequence[ResponseStep]):
        self.steps: List[ResponseStep] = list(steps)

    async def run(self, transaction: UserTransaction) -> UserTransaction:
        """
        Run every applicable response step, mutating the transaction in place.

        :param transaction: The transaction whose response to shape.
        :returns: The same `transaction`.
        """
        for step in self.steps:
            reason = step.applicable(transaction)
            if reason is not None:
                LOG.debug("Skipping response step %r: %s", step.id, reason)
                continue
            await step.process(transaction)
        return transaction


def _resolve_step(stage_id: str,
                  steps: Mapping[str, Type[Step]]) -> Type[Step]:
    try:
        return steps[stage_id]
    except KeyError:
        raise KeyError(
            f"No step registered with id {stage_id!r}. "
            f"Known steps: {sorted(steps)}")


def build_pipelines(
    config: PipelineConfig,
) -> Tuple[RequestPipeline, ResponsePipeline]:
    """
    Instantiate both pipelines from configuration.

    Step classes come from the entry point group for that pipeline
    (`brainforge.inference.request_steps` or
    `brainforge.inference.response_steps`). A response step placed in the
    request list (or vice versa) is absent from that group and fails at
    build time.

    :param config: Parsed pipeline configuration.
    :returns: `(request_pipeline, response_pipeline)`.
    :raises KeyError: If a configured step id is not registered in that
        pipeline's entry point group.
    :raises TypeError: If an installed entry point is not the step type for
        its group.
    """
    request_step_classes = load_entry_point_steps(
        REQUEST_STEP_ENTRY_POINT, RequestStep)
    response_step_classes = load_entry_point_steps(
        RESPONSE_STEP_ENTRY_POINT, ResponseStep)

    request_steps: List[RequestStep] = []
    for entry in config.request:
        if not entry.enabled:
            LOG.info("Request step %r disabled by config", entry.id)
            continue
        cls = _resolve_step(entry.id, request_step_classes)
        request_steps.append(cls(entry.config))

    response_steps: List[ResponseStep] = []
    for entry in config.response:
        if not entry.enabled:
            LOG.info("Response step %r disabled by config", entry.id)
            continue
        cls = _resolve_step(entry.id, response_step_classes)
        response_steps.append(cls(entry.config))

    LOG.info("Built request pipeline: %s",
             [s.id for s in request_steps] or "(empty)")
    LOG.info("Built response pipeline: %s",
             [s.id for s in response_steps] or "(empty)")
    return RequestPipeline(request_steps), ResponsePipeline(response_steps)
