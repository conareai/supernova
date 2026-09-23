# supernova

`supernova` is a toolkit for building large-scale vector search benchmarks. It does five things:

1. **Embedding generation** — take a dataset, embed it with any model (dense, sparse, or multivector), and produce parquet files.
2. **Ground truth computation** — perform brute-force, exact, nearest neighbour search over a corpus.
3. **Vector store loading** — take pre-embedded parquet files and load them into a vector database.
4. **Query testing** — fire a query load at a vector store to measure its latency and recall.
5. **Parameter sweeping** — orchestrate loading + load testing across a matrix of index/search configs, in one combined report

Each is its own, self-contained tool, and each tool partitions work to allow parallelization via `--num-jobs` / `--job-rank`.
## Mental model

The pipelines are independent and meant to be reproducible. Each tool doesn't know that the others exist. You can run them separately, on different machines, and at different times. The intermediate parquet files connect each step.

## One CLI, many tools

`nova` is a **cli-style dispatcher**: `nova <cmd>` finds an executable named `nova-<cmd>` on your `PATH` and execs it. A command can be implemented in any language — a Rust binary (`nova-load`, `nova-storm`) or a Python console script (`nova-embed`) look identical from the outside. You install the dispatcher once and add only the sub-tools you need.

`nova --help` lists every `nova-*` tool found on your `PATH`.

| Command | Purpose | Language |
|---------|---------|----------|
| `nova embed` | Embed a dataset into parquet | Python |
| `nova bf`    | Brute-force exact k-NN ground truth (GPU) | Python |
| `nova load`  | Load pre-embedded parquet into a vector store | Rust |
| `nova storm` | Load-test a vector store | Rust |
| `nova sweep` | Sweep index/search configs across `nova load` + `nova storm` | Python |

Distributed runs are just N copies of a tool, each with its own `--job-rank`. Orchestration (SkyPilot, Slurm, a shell loop) is **external** — it provisions nodes and invokes `nova <tool>` on each, passing that node's rank. The tools themselves know nothing about the fleet. (`nova sweep` is the one exception today — single machine, sequential; see [Sweep](sweep/overview.md#distribution).)

## Key design principles

- **Streaming** — no pipeline loads the full dataset into memory; data is processed in chunks/batches throughout.
- **Self-sharding tools** — each tool is designed to be stateless and partitions its own work from `--num-jobs` / `--job-rank` (rank defaults to `$SKYPILOT_JOB_RANK`). No central coordinator.
- **YAML-driven** — every run is defined by a YAML config; `${VAR}` / `${VAR:-default}` references are expanded from the environment.
- **Flat parquet output** — embedding output is flat columnar data (no nested JSON). Payload composition happens at load time.

See the [CLI reference](reference/cli.md) for every flag, and the [Embedding](embedding/overview.md), [Loading](loading/overview.md), [Brute Force](brute-force/overview.md), and [Sweep](sweep/overview.md) sections for each tool's config.
