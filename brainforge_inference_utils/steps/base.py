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

Two step types plug into the two pipelines:

- `InputStep` runs before generation. It shapes the request on the
  `UserTransaction` (RAG/MCP context, ambient context, request history
  validation, ...), mutating it in place.
- `ResponseStep` runs after generation. It has the complete response on the
  `UserTransaction` and may return a revised response (or mutate it in place) to
  annotate or rewrite it (citations, sanitizing, guardrails, ...).

Both share two conventions:

1. **Defensive, not dependent.** Steps declare no relationships to one another.
   A step that requires something it does not find reports itself inapplicable
   from `applicable()` and is skipped (with a logged reason) rather than failing
   the pipeline. This keeps configuration order the only ordering contract and
   lets steps be added, removed, or substituted freely.

2. **Shared object, separate namespaces.** Data that must pass between steps
   travels on the `UserTransaction`, not through a step-to-step interface.
   Input steps use `transaction.input_context`; response steps use
   `transaction.output_context`. Step ids are unique only within one side.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, List, Optional, Sequence

from neon_data_models.models.api.llm import LLMResponse

from brainforge_inference_utils.transaction import UserTransaction


@dataclass(frozen=True)
class RequestContextOption:
    """
    One `UserTransaction.request_context` key a step reads.

    Hosts collect these from the built pipeline to present the keys and, when
    known, the values a client may send. `values is None` means the key is
    supported but unconstrained (any value of the implied type). An empty
    `values` list means the key is supported but no choices are available
    right now (for example an empty live catalog).
    """

    key: str
    values: Optional[Sequence[Any]] = None
    description: str = ""


class Step(ABC):
    """
    Common base for all pipeline steps.

    A step is constructed once, at pipeline build time, with its resolved
    configuration. The same instance is reused across transactions, so steps
    should keep per-request state on the `UserTransaction`, not on `self`.

    :cvar id: Unique step identifier used in configuration. Subclasses must set
        this to a stable, non-empty string.
    :cvar request_context_options: Static `request_context` keys this step
        reads. Override `supported_request_context` when allowed values must be
        fetched live (for example a provider catalog).
    """

    id: ClassVar[str] = ""
    request_context_options: ClassVar[Sequence[RequestContextOption]] = ()

    def __init__(self, config: Optional[dict] = None):
        """
        :param config: Step-specific configuration from the pipeline config
            entry's `config` mapping. Stored as `self.config`.
        """
        self.config: dict = dict(config or {})

    def __repr__(self) -> str:
        return f"<{type(self).__name__} id={self.id!r}>"

    async def supported_request_context(
        self, timeout: float = 10.0
    ) -> List[RequestContextOption]:
        """
        Return the `request_context` keys this step reads, with allowed values.

        The default copies `request_context_options`. Override when values must
        be fetched live (for example a provider catalog). `timeout` is unused
        for static declarations.

        :param timeout: Seconds to wait for a live catalog, when overridden.
        """
        return list(self.request_context_options)


class InputStep(Step, ABC):
    """
    A step that shapes the request before generation.

    Implementations mutate `transaction.request` in place (selecting a persona,
    trimming or compacting history, injecting retrieved context, adjusting
    generation params, ...).

    Steps that read `transaction.request_context` should declare those keys on
    `request_context_options` (static choices) or override
    `supported_request_context` (live catalogs). Hosts collect the result via
    `BrainForgeClient.supported_request_context`.
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
