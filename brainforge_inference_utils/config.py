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
Pipeline configuration schema.

Configuration is two explicitly ordered lists of step entries: `request` and
`response`. Order is authoritative; the pipeline runs steps in exactly the
order given and does not reorder. Each entry names a step `id` (a bundled id
from `steps_by_id`, or a plugin id `author.step_name`) and optionally carries
step-specific `config`.

Example (YAML)::

    pipeline:
      request:
        - id: neon.locale
          config:
            mcp_url: https://mcp.neonaiservices2.com/mcp
            default_location: Renton, WA, USA
            timeout: 5.0
        - {id: neon.rag, config: {mcp_url: https://mcp.neonaiservices2.com/mcp, top_k: 10}}
        - neon.history                             # last: valid, truncated history
      response:
        - id: neon.citations
          config:
            mcp_url: https://mcp.neonaiservices2.com/mcp
            openai_url: https://hana.example/brainforge/openai/chat/completions

A host typically loads its own config and hands the `pipeline` section here::

    pipeline_config = PipelineConfig.model_validate(config["pipeline"])
"""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StepConfig(BaseModel):
    """
    A single step entry within a pipeline.

    :param id: Step id to resolve and instantiate. Bundled steps use their
        current ids. Plugin steps use `author.step_name`.
    :param enabled: When false, the entry is skipped at build time (a
        convenient way to turn a step off without deleting its config).
    :param config: Step-specific configuration passed to the step constructor.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    enabled: bool = True
    config: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _coerce_shorthand(cls, data: Any) -> Any:
        """Allow a bare string entry as shorthand for `{id: <string>}`."""
        if isinstance(data, str):
            return {"id": data}
        return data


class PipelineConfig(BaseModel):
    """
    Ordered step lists for the request and response pipelines.

    :param request: Ordered `RequestStep` entries, run before generation.
    :param response: Ordered `ResponseStep` entries, run after generation.
    """

    model_config = ConfigDict(extra="forbid")

    request: List[StepConfig] = Field(default_factory=list)
    response: List[StepConfig] = Field(default_factory=list)
