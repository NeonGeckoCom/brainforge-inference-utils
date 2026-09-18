# NEON AI (TM) SOFTWARE, Software Development Kit & Application Framework
# All trademark and other rights reserved by their respective owners
# Copyright 2008-2025 Neongecko.com Inc.
"""
Step base classes.

Two step types plug into the two pipelines:

- `InputStep` runs before generation. It shapes the request on the
  `UserTransaction` (persona selection, history windowing/compaction, RAG/MCP
  context, ...), mutating it in place.
- `ResponseStep` runs after generation. It has the complete response on the
  `UserTransaction` and may return a revised response (or mutate it in place) to
  annotate or rewrite it (citations, sanitizing, guardrails, ...).

Both share two conventions:

1. **Defensive, not dependent.** Steps declare no relationships to one another.
   A step that requires something it does not find reports itself inapplicable
   from `applicable()` and is skipped (with a logged reason) rather than failing
   the pipeline. This keeps configuration order the only ordering contract and
   lets steps be added, removed, or substituted freely.

2. **Shared object, narrow surface.** Data that must pass between steps travels
   on the `UserTransaction`, not through a step-to-step interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Optional

from neon_data_models.models.api.llm import LLMResponse

from brainforge_inference_utils.transaction import UserTransaction


class Step(ABC):
    """
    Common base for all pipeline steps.

    A step is constructed once, at pipeline build time, with its resolved
    configuration. The same instance is reused across transactions, so steps
    should keep per-request state on the `UserTransaction`, not on `self`.

    :cvar id: Unique step identifier used in configuration. Subclasses must set
        this to a stable, non-empty string.
    """

    id: ClassVar[str] = ""

    def __init__(self, config: Optional[dict] = None):
        """
        :param config: Step-specific configuration from the pipeline config
            entry's `config` mapping. Stored as `self.config`.
        """
        self.config: dict = dict(config or {})

    def __repr__(self) -> str:
        return f"<{type(self).__name__} id={self.id!r}>"


class InputStep(Step, ABC):
    """
    A step that shapes the request before generation.

    Implementations mutate `transaction.request` in place (selecting a persona,
    trimming or compacting history, injecting retrieved context, adjusting
    generation params, ...).
    """

    def applicable(self, transaction: UserTransaction) -> Optional[str]:
        """
        Decide whether this step should run for the given transaction.

        :param transaction: The current transaction.
        :returns: `None` to run the step, or a short human-readable reason to
            skip it (logged). The default runs unconditionally; override to add
            per-request gating.
        """
        return None

    @abstractmethod
    async def process(self, transaction: UserTransaction) -> None:
        """
        Apply this step's effect to the request.

        Only called when `applicable(transaction)` returned `None`.
        Implementations mutate `transaction` in place and return nothing.

        :param transaction: The transaction whose `request` to shape.
        """
        raise NotImplementedError


class ResponseStep(Step, ABC):
    """
    A step that shapes the generated response.

    Implementations read the response from `transaction.response` and either
    mutate it in place or return a new `LLMResponse` to replace it. Returning
    `None` leaves the current response unchanged.
    """

    def applicable(self, transaction: UserTransaction) -> Optional[str]:
        """
        Decide whether this step should run for the given transaction.

        :param transaction: The current transaction (with `response` populated).
        :returns: `None` to run the step, or a short human-readable reason to
            skip it (logged). The default runs unconditionally; override to add
            per-request gating (e.g. skip when there is nothing to annotate).
        """
        return None

    @abstractmethod
    async def process(self,
                      transaction: UserTransaction) -> Optional[LLMResponse]:
        """
        Annotate or rewrite the response.

        Only called when `applicable(transaction)` returned `None`.

        :param transaction: The transaction whose `response` to shape.
        :returns: A revised `LLMResponse` to replace the current one, or `None`
            to keep it (including when the step mutated the response in place).
        """
        raise NotImplementedError
