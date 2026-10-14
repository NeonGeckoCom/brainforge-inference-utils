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
Step base classes.

Concrete steps ship in separate plugins and are discovered from entry points.
`steps_by_id` remains for any steps still bundled with this package. Plugin
step ids use `author.step_name`.
"""

from __future__ import annotations

from typing import Dict, Type

from brainforge_inference_utils.steps.base import (
    RequestStep,
    RequestContextOption,
    ResponseStep,
    Step,
)

steps_by_id: Dict[str, Type[Step]] = {}

__all__ = [
    "Step",
    "RequestStep",
    "ResponseStep",
    "RequestContextOption",
    "steps_by_id",
]
