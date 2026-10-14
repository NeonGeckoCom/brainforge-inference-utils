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
Step base classes, first-party step implementations, and their id mapping.

Bundled steps subclass `RequestStep` or `ResponseStep`, set a unique `id`, and
are added to `steps_by_id` below so pipelines can resolve them by id. Plugin
steps are discovered from entry points instead of this mapping. Their ids use
`author.step_name`.
"""

from __future__ import annotations

from typing import Dict, Type

from brainforge_inference_utils.steps.ambient_context import AmbientContextStep
from brainforge_inference_utils.steps.base import (
    RequestStep,
    RequestContextOption,
    ResponseStep,
    Step,
)
from brainforge_inference_utils.steps.rag import RagStep, ResponseParser
from brainforge_inference_utils.steps.request_history import RequestHistoryStep

steps_by_id: Dict[str, Type[Step]] = {
    AmbientContextStep.id: AmbientContextStep,
    RagStep.id: RagStep,
    ResponseParser.id: ResponseParser,
    RequestHistoryStep.id: RequestHistoryStep,
}

__all__ = [
    "Step",
    "RequestStep",
    "ResponseStep",
    "RequestContextOption",
    "AmbientContextStep",
    "RagStep",
    "ResponseParser",
    "RequestHistoryStep",
    "steps_by_id",
]
