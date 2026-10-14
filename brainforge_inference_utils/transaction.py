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
The object a pipeline operates on.

`UserTransaction` represents one user request/response cycle. Request steps shape
its `request` and write `request_step_context`; generation (performed elsewhere)
populates its `response`; response steps annotate that `response` and write
`response_step_context`.

The request and response are the existing `neon-data-models` models used across
the BrainForge stack (HANA, `brainforge-llm-service`), so this package does not
redefine their shape.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from neon_data_models.models.api.llm import LLMRequest, LLMResponse


class UserTransaction(BaseModel):
    """
    One user request/response cycle as it flows through the pipelines.

    :param request: The request being shaped by the request pipeline.
    :param response: The generated response, populated after generation and
        shaped by the response pipeline. `None` until generation completes.
    :param request_step_context: Transient data written by request steps, keyed by
        request step id.
    :param response_step_context: Transient data written by response steps, keyed by
        response step id.
    """

    request: LLMRequest = Field(
        description="The LLM request shaped by the request pipeline")
    response: Optional[LLMResponse] = Field(
        default=None,
        description="The generated response shaped by the response pipeline; "
                    "None until generation completes")
    request_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Client-provided context read by request steps")
    request_step_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Request-step context keyed by request step id")
    response_step_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Response-step context keyed by response step id")
