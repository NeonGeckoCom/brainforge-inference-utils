# NEON AI (TM) SOFTWARE, Software Development Kit & Application Framework
# All trademark and other rights reserved by their respective owners
# Copyright 2008-2025 Neongecko.com Inc.
"""
The input and response pipelines.

- `InputPipeline` runs ordered `InputStep`s over a `UserTransaction` to shape
  its request before generation.
- `ResponsePipeline` runs ordered `ResponseStep`s over a `UserTransaction` to
  annotate or rewrite its response after generation.

Generation itself is out of scope here: a caller runs the input pipeline,
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
    InputStep,
    ResponseStep,
    Step,
)
from brainforge_inference_utils.transaction import UserTransaction

LOG = logging.getLogger(__name__)


class InputPipeline:
    """
    Runs `InputStep`s in order over a transaction's request.

    :param steps: Ordered input steps.
    """

    def __init__(self, steps: Sequence[InputStep]):
        self.steps: List[InputStep] = list(steps)

    async def run(self, transaction: UserTransaction) -> UserTransaction:
        """
        Run every applicable input step, mutating the transaction in place.

        :param transaction: The transaction whose request to shape.
        :returns: The same `transaction`.
        """
        for step in self.steps:
            reason = step.applicable(transaction)
            if reason is not None:
                LOG.debug("Skipping input step %r: %s", step.id, reason)
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
) -> Tuple[InputPipeline, ResponsePipeline]:
    """
    Instantiate both pipelines from configuration.

    Step classes are looked up by id in `steps` (defaulting to the static
    `steps_by_id` mapping) and validated against the pipeline they are declared
    in, so a response step placed in the input list (or vice versa) fails fast
    at build time.

    :param config: Parsed pipeline configuration.
    :param steps: Optional id -> `Step` class mapping. Defaults to
        `brainforge_inference_utils.steps.steps_by_id`.
    :returns: `(input_pipeline, response_pipeline)`.
    :raises KeyError: If a configured step id is not present in `steps`.
    :raises TypeError: If a configured step resolves to the wrong step type.
    """
    steps = steps_by_id if steps is None else steps

    input_steps: List[InputStep] = []
    for entry in config.input:
        if not entry.enabled:
            LOG.info("Input step %r disabled by config", entry.id)
            continue
        cls = _resolve_step(entry.id, steps)
        if not (isinstance(cls, type) and issubclass(cls, InputStep)):
            raise TypeError(
                f"Step {entry.id!r} is configured as an input step but "
                f"{getattr(cls, '__name__', cls)!r} is not an InputStep")
        input_steps.append(cls(entry.config))

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

    LOG.info("Built input pipeline: %s",
             [s.id for s in input_steps] or "(empty)")
    LOG.info("Built response pipeline: %s",
             [s.id for s in response_steps] or "(empty)")
    return InputPipeline(input_steps), ResponsePipeline(response_steps)
