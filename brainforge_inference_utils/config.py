# NEON AI (TM) SOFTWARE, Software Development Kit & Application Framework
# All trademark and other rights reserved by their respective owners
# Copyright 2008-2025 Neongecko.com Inc.
"""
Pipeline configuration schema.

Configuration is two explicitly ordered lists of step entries: `input` and
`response`. Order is authoritative; the pipeline runs steps in exactly the
order given and does not reorder. Each entry names a step `id` (resolved via the
`steps_by_id` mapping) and optionally carries step-specific `config`.

Example (YAML)::

    pipeline:
      input:
        - persona                                   # shorthand for {id: persona}
        - id: ambient_context
          config:
            mcp_url: https://mcp.neonaiservices2.com/mcp
            default_location: Renton, WA, USA
            timeout: 5.0
        - {id: rag, config: {top_k: 10}}
        - {id: history_window, config: {max_messages: 5}}
        - {id: sanitize}
      response:
        - {id: reasoning_normalize}
        - {id: citations}

A host typically loads its own config and hands the `pipeline` section here::

    pipeline_config = PipelineConfig.model_validate(config["pipeline"])
"""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StepConfig(BaseModel):
    """
    A single step entry within a pipeline.

    :param id: Step id to look up in `steps_by_id` and instantiate.
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
    Ordered step lists for the input and response pipelines.

    :param input: Ordered `InputStep` entries, run before generation.
    :param response: Ordered `ResponseStep` entries, run after generation.
    """

    model_config = ConfigDict(extra="forbid")

    input: List[StepConfig] = Field(default_factory=list)
    response: List[StepConfig] = Field(default_factory=list)
