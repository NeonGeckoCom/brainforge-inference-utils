# NEON AI (TM) SOFTWARE, Software Development Kit & Application Framework
# All trademark and other rights reserved by their respective owners
# Copyright 2008-2025 Neongecko.com Inc.
"""
The object a pipeline operates on.

`UserTransaction` represents one user request/response cycle. Input steps shape
its `request` and write `input_context`; generation (performed elsewhere)
populates its `response`; response steps annotate that `response` and write
`output_context`.

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

    :param request: The request being shaped by the input pipeline.
    :param response: The generated response, populated after generation and
        shaped by the response pipeline. `None` until generation completes.
    :param input_context: Transient data written by input steps, keyed by
        input step id.
    :param output_context: Transient data written by response steps, keyed by
        response step id.
    """

    request: LLMRequest = Field(
        description="The LLM request shaped by the input pipeline")
    response: Optional[LLMResponse] = Field(
        default=None,
        description="The generated response shaped by the response pipeline; "
                    "None until generation completes")
    input_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Input-step context keyed by input step id")
    output_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Response-step context keyed by response step id")
