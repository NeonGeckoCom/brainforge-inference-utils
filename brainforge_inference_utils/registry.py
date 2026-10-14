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
Discovery of plugin steps via setuptools entry points.

Request and response steps are registered in separate groups, so the same
`author.step_name` id can exist on both sides. Every step class comes from
the entry point group for its pipeline.
"""

from __future__ import annotations

import logging
import re
from importlib.metadata import entry_points
from typing import Dict, Type

from brainforge_inference_utils.steps.base import Step

LOG = logging.getLogger(__name__)

REQUEST_STEP_ENTRY_POINT = "brainforge.inference.request_steps"
RESPONSE_STEP_ENTRY_POINT = "brainforge.inference.response_steps"

# author.step_name — one developer slug, one step name.
STEP_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")


def _distribution_name(entry_point) -> str:
    dist = getattr(entry_point, "dist", None)
    return getattr(dist, "name", None) or "unknown"


def load_entry_point_steps(
    group: str,
    expected_type: Type[Step],
) -> Dict[str, Type[Step]]:
    """
    Load step classes advertised under `group`.

    Each entry point name must be `author.step_name` and must equal the
    loaded class's `id`. Names must be unique within the group. The same name
    may still be registered under the other pipeline's group.

    :param group: Entry point group to read.
    :param expected_type: `RequestStep` or `ResponseStep`. Loaded classes must
        subclass it.
    :returns: Entry point name -> step class.
    :raises ValueError: If a name is malformed, duplicated, or differs from
        the class `id`.
    :raises TypeError: If a loaded object is not a subclass of `expected_type`.
    """
    seen: Dict[str, str] = {}
    loaded: Dict[str, Type[Step]] = {}
    for entry_point in entry_points(group=group):
        dist_name = _distribution_name(entry_point)
        if not STEP_ID_PATTERN.fullmatch(entry_point.name):
            raise ValueError(
                f"Entry point {entry_point.name!r} in {group} from "
                f"{dist_name} must use the form author.step_name")
        prior = seen.get(entry_point.name)
        if prior is not None:
            raise ValueError(
                f"Multiple packages register {group} step "
                f"{entry_point.name!r}: {prior} and {dist_name}")
        seen[entry_point.name] = dist_name

        try:
            step_cls = entry_point.load()
        except Exception as exc:
            raise ImportError(
                f"Failed to load step {entry_point.name!r} from {dist_name}"
            ) from exc
        if not (isinstance(step_cls, type) and issubclass(step_cls, expected_type)):
            raise TypeError(
                f"Entry point {entry_point.name!r} in {group} from {dist_name} "
                f"loaded {getattr(step_cls, '__name__', step_cls)!r}, which is "
                f"not a {expected_type.__name__}")
        if step_cls.id != entry_point.name:
            raise ValueError(
                f"Step {step_cls.__name__} from {dist_name} has id "
                f"{step_cls.id!r}, which does not match entry point "
                f"{entry_point.name!r}")
        loaded[entry_point.name] = step_cls
        LOG.debug("Loaded %s step %r from %s",
                  expected_type.__name__, entry_point.name, dist_name)
    return loaded
