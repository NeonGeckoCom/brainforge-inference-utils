# NEON AI (TM) SOFTWARE, Software Development Kit & Application Framework
# All trademark and other rights reserved by their respective owners
# Copyright 2008-2025 Neongecko.com Inc.
"""
Step base classes, first-party step implementations, and their id mapping.

Concrete steps are intentionally not shipped yet. New steps subclass
`InputStep` or `ResponseStep`, set a unique `id`, and are added to
`steps_by_id` below (mirroring `collectors_by_id` in `brainforge_dataset_utils`)
so pipelines can resolve them by id.
"""

from __future__ import annotations

from typing import Dict, Type

from brainforge_inference_utils.steps.ambient_context import AmbientContextStep
from brainforge_inference_utils.steps.base import (
    InputStep,
    ResponseStep,
    Step,
)

steps_by_id: Dict[str, Type[Step]] = {
    AmbientContextStep.id: AmbientContextStep,
}

__all__ = [
    "Step",
    "InputStep",
    "ResponseStep",
    "AmbientContextStep",
    "steps_by_id",
]
