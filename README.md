# BrainForge Inference Utilities

Pluggable pipelines for shaping BrainForge LLM requests and responses.

This package defines request/response orchestration that attaches to an
existing inference workflow. Concrete behavior (ambient context, RAG/MCP
injection, request history validation, citations, response sanitizing,
safety guardrails, ...) can be implemented as **steps** that plug into one of
two pipelines:

- **Request pipeline** — runs before generation and shapes the request:
  injecting ambient time/weather, retrieving RAG for host-selected
  providers, then validating and truncating history last.
- **Response pipeline** — runs after generation and shapes the response:
  citation rewrite via `ResponseParser`, sanitizing, guardrails, etc.

Primary generation stays with the host, between the two pipelines. The host
keeps the `UserTransaction` from the request pipeline until the response is
ready. A response step may call a configured OpenAI-compatible endpoint
(for example citation rewrite) without the host owning that second request.

Both pipelines operate on a `UserTransaction`, which carries the request,
response, and two per-side context dicts (`request_step_context` and
`response_step_context`). Requests and responses use the existing
`neon-data-models` models (`LLMRequest` / `LLMResponse`).

## Concepts

| Concept | Purpose |
| --- | --- |
| `UserTransaction` | The request/response pair plus `request_step_context` / `response_step_context` dicts. The host holds this object across generation. |
| `BrainForgeClient` | Builds pipelines from config and exposes `run_request_pipeline` / `run_response_pipeline` / `supported_request_context`. |
| `RequestStep` | Shapes `transaction.request` before generation. Declares supported `request_context` keys/values. |
| `ResponseStep` | Annotates/rewrites `transaction.response` after generation. |
| `RequestContextOption` | One `request_context` key a step reads, with optional allowed values for hosts to present. |
| `RequestPipeline` / `ResponsePipeline` | Run an ordered list of steps over a transaction. |
| `PipelineConfig` / `StepConfig` | Declarative, ordered step lists (see below). |
| `build_pipelines(config)` | Build both pipelines from config, resolving step ids via `steps_by_id`. |

## Usage sketch

```python
client = BrainForgeClient(pipeline_config)

options = await client.supported_request_context()
# e.g. options["rag_providers"].values, options["conversational"].values

transaction = await client.run_request_pipeline(llm_request)
# host performs generation using transaction.request
transaction = await client.run_response_pipeline(transaction, llm_response)
```

Request steps write `transaction.request_step_context[step.id]`. Response steps write
`transaction.response_step_context[step.id]` and may *read* the request-step context. Those
namespaces are independent, so the same step id may exist on both sides
without colliding. Context is available across generation only because the
same `UserTransaction` is passed back in; this package does not store it
between calls.

## Design notes

- **Steps are independently defensive.** There is no dependency graph between
  steps. A step that needs something it does not find (e.g. a citation step with
  no retrieved context) reports itself inapplicable via `applicable()` and is
  skipped with a logged reason, rather than failing the pipeline.
- **Ordering is explicit.** Step order is exactly the order declared in
  configuration; the pipeline does not reorder.
- **Steps may be repeated.** The same step id may appear more than once in a
  pipeline, including with different `config`. Each appearance is a separate
  instance and runs in declared order. The pipeline does not merge or
  de-duplicate entries.
- **Steps own their context.** Each side has its own dict keyed by step id.
  A repeated step on the *same* side shares that key; a step id on the request
  side does not clash with the same id on the response side. Each step is
  responsible for how it stores and reads its own data across one or more
  calls (for example appending to a list instead of overwriting a scalar).
  Response steps that consume request-step data should read
  `transaction.request_step_context` and skip when nothing usable is present.
- **Format contracts live together.** Steps that share a parse format
  (e.g. a citation formatter and the sanitizer that strips it) belong in the
  same module and should derive from a shared constant so they cannot drift.

## Deferred

- A step **registry** (in-process registration + entry-point discovery under
  `brainforge.inference.steps`). For now, steps are resolved from the static
  `steps_by_id` mapping.
- A **services** adapter for streaming, tool calls, and token counting.
- Richer persistent state or execution tracing beyond the transient
  `request_step_context` / `response_step_context` dicts on `UserTransaction`.
