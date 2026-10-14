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
Pipeline configuration schema.

Configuration is two explicitly ordered lists of step entries: `request` and
`response`. Order is authoritative; the pipeline runs steps in exactly the
order given and does not reorder. Each entry names a step `id`
(`author.step_name`, the entry point name) and optionally carries
step-specific `config`.

Example (YAML)::

    pipeline:
      request:
        - id: brainforge.locale
          config:
            mcp_url: https://mcp.neonaiservices2.com/mcp
            default_location: Renton, WA, USA
            timeout: 5.0
        - {id: brainforge.rag, config: {mcp_url: https://mcp.neonaiservices2.com/mcp, top_k: 10}}
        - brainforge.history                       # last: valid, truncated history
      response:
        - id: brainforge.citations
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

    :param id: Step id to resolve and instantiate. This is the entry point
        name, `author.step_name`.
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
