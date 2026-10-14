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
from typing import List, Mapping, Optional, Sequence, Tuple, Type

from brainforge_inference_utils.config import PipelineConfig
from brainforge_inference_utils.steps import steps_by_id
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
        Run every applicable response step, threading the (possibly replaced)
        response back onto the transaction.

        :param transaction: The transaction whose response to shape.
        :returns: The same `transaction`.
        """
        for step in self.steps:
            reason = step.applicable(transaction)
            if reason is not None:
                LOG.debug("Skipping response step %r: %s", step.id, reason)
                continue
            revised = await step.process(transaction)
            if revised is not None:
                transaction.response = revised
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
    steps: Optional[Mapping[str, Type[Step]]] = None,
) -> Tuple[RequestPipeline, ResponsePipeline]:
    """
    Instantiate both pipelines from configuration.

    Step classes are looked up by id in `steps` (defaulting to the static
    `steps_by_id` mapping) and validated against the pipeline they are declared
    in, so a response step placed in the request list (or vice versa) fails fast
    at build time.

    :param config: Parsed pipeline configuration.
    :param steps: Optional id -> `Step` class mapping. Defaults to
        `brainforge_inference_utils.steps.steps_by_id`.
    :returns: `(request_pipeline, response_pipeline)`.
    :raises KeyError: If a configured step id is not present in `steps`.
    :raises TypeError: If a configured step resolves to the wrong step type.
    """
    steps = steps_by_id if steps is None else steps

    request_steps: List[RequestStep] = []
    for entry in config.request:
        if not entry.enabled:
            LOG.info("Request step %r disabled by config", entry.id)
            continue
        cls = _resolve_step(entry.id, steps)
        if not (isinstance(cls, type) and issubclass(cls, RequestStep)):
            raise TypeError(
                f"Step {entry.id!r} is configured as a request step but "
                f"{getattr(cls, '__name__', cls)!r} is not a RequestStep")
        request_steps.append(cls(entry.config))

    response_steps: List[ResponseStep] = []
    for entry in config.response:
        if not entry.enabled:
            LOG.info("Response step %r disabled by config", entry.id)
            continue
        cls = _resolve_step(entry.id, steps)
        if not (isinstance(cls, type) and issubclass(cls, ResponseStep)):
            raise TypeError(
                f"Step {entry.id!r} is configured as a response step but "
                f"{getattr(cls, '__name__', cls)!r} is not a ResponseStep")
        response_steps.append(cls(entry.config))

    LOG.info("Built request pipeline: %s",
             [s.id for s in request_steps] or "(empty)")
    LOG.info("Built response pipeline: %s",
             [s.id for s in response_steps] or "(empty)")
    return RequestPipeline(request_steps), ResponsePipeline(response_steps)
