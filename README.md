# BrainForge Inference Utilities

Pluggable pipelines for shaping BrainForge LLM requests and responses.

This package defines request/response orchestration that attaches to an
existing inference workflow. Concrete behavior (conversation history handling,
RAG/MCP context injection, persona validation, citations, response sanitizing,
safety guardrails, ...) can be implemented as **steps** that plug into one of
two pipelines:

- **Input pipeline** — runs before generation and shapes the request: selecting
  a persona, windowing/compacting history, retrieving RAG or MCP context, etc.
- **Response pipeline** — runs after generation and shapes the response:
  inserting citations, stripping artifacts, applying guardrails, etc.

This package does not call an LLM. The host performs generation between the
two pipelines and keeps the `UserTransaction` returned by the input pipeline
until the response is ready.

Both pipelines operate on a `UserTransaction`, which carries the request,
response, and two per-side context dicts (`input_context` and
`output_context`). Requests and responses use the existing
`neon-data-models` models (`LLMRequest` / `LLMResponse`).

## Concepts

| Concept | Purpose |
| --- | --- |
| `UserTransaction` | The request/response pair plus `input_context` / `output_context` dicts. The host holds this object across generation. |
| `BrainForgeClient` | Builds pipelines from config and exposes `run_input_pipeline` / `run_output_pipeline`. |
| `InputStep` | Shapes `transaction.request` before generation. |
| `ResponseStep` | Annotates/rewrites `transaction.response` after generation. |
| `InputPipeline` / `ResponsePipeline` | Run an ordered list of steps over a transaction. |
| `PipelineConfig` / `StepConfig` | Declarative, ordered step lists (see below). |
| `build_pipelines(config)` | Build both pipelines from config, resolving step ids via `steps_by_id`. |

## Usage sketch

```python
client = BrainForgeClient(pipeline_config)

transaction = await client.run_input_pipeline(llm_request)
# host performs generation using transaction.request
transaction = await client.run_output_pipeline(transaction, llm_response)
```

Input steps write `transaction.input_context[step.id]`. Response steps write
`transaction.output_context[step.id]` and may *read* the input dict. Those
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
  A repeated step on the *same* side shares that key; a step id on the input
  side does not clash with the same id on the output side. Each step is
  responsible for how it stores and reads its own data across one or more
  calls (for example appending to a list instead of overwriting a scalar).
  Response steps that consume input-step data should read
  `transaction.input_context` and skip when nothing usable is present.
- **Format contracts live together.** Steps that share an output/parse format
  (e.g. a citation formatter and the sanitizer that strips it) belong in the
  same module and should derive from a shared constant so they cannot drift.

## Deferred

- A step **registry** (in-process registration + entry-point discovery under
  `brainforge.inference.steps`). For now, steps are resolved from the static
  `steps_by_id` mapping.
- A **services** adapter for streaming, tool calls, and token counting.
- Richer persistent state or execution tracing beyond the transient
  `input_context` / `output_context` dicts on `UserTransaction`.
