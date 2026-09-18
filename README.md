# BrainForge Inference Utilities

Pluggable pipelines for shaping BrainForge LLM requests and responses.

This package defines LLM request/response orchestration. Concrete
behavior (conversation history handling, RAG/MCP context injection, persona
validation, citations, response sanitizing, safety guardrails, ...) can be
implemented as **steps** that plug into one of two pipelines:

- **Input pipeline** — runs before generation and shapes the request: selecting
  a persona, windowing/compacting history, retrieving RAG or MCP context, etc.
- **Response pipeline** — runs after generation and shapes the response:
  inserting citations, stripping artifacts, applying guardrails, etc.

Both pipelines operate on a `UserTransaction`, which carries the request and
response as the existing `neon-data-models` models (`LLMRequest` / `LLMResponse`)
used across the BrainForge stack.

## Concepts

| Concept | Purpose |
| --- | --- |
| `UserTransaction` | The request/response pair a pipeline operates on (`request: LLMRequest`, `response: LLMResponse`). |
| `InputStep` | Shapes `transaction.request` before generation. |
| `ResponseStep` | Annotates/rewrites `transaction.response` after generation. |
| `InputPipeline` / `ResponsePipeline` | Run an ordered list of steps over a transaction. |
| `PipelineConfig` / `StepConfig` | Declarative, ordered step lists (see below). |
| `build_pipelines(config)` | Build both pipelines from config, resolving step ids via `steps_by_id`. |

## Usage sketch

```python
input_pipeline, response_pipeline = build_pipelines(pipeline_config)

await input_pipeline.run(transaction)     # shape request
# ... caller performs generation, sets transaction.response ...
await response_pipeline.run(transaction)  # shape response
```

Generation sits between the two pipelines and is the caller's responsibility;
this package does not perform or depend on any particular LLM transport.

## Design notes

- **Steps are independently defensive.** There is no dependency graph between
  steps. A step that needs something it does not find (e.g. a citation step with
  no retrieved context) reports itself inapplicable via `applicable()` and is
  skipped with a logged reason, rather than failing the pipeline.
- **Ordering is explicit.** Step order is exactly the order declared in
  configuration; the pipeline does not reorder.
- **Format contracts live together.** Steps that share an output/parse format
  (e.g. a citation formatter and the sanitizer that strips it) belong in the
  same module and should derive from a shared constant so they cannot drift.

## Deferred

- A step **registry** (in-process registration + entry-point discovery under
  `brainforge.inference.steps`). For now, steps are resolved from the static
  `steps_by_id` mapping.
- A **services** adapter for LLM completion/streaming/tool calls and token
  counting.
- Richer per-request **state** (context blocks, retrieved documents, execution
  trace) beyond the request/response on `UserTransaction`.
