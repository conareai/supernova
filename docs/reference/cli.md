# CLI

`nova <cmd> [args...]` runs the matching `nova-<cmd>` executable from your `PATH`.

```bash
nova --help
nova --version
nova <cmd> --help
```

## `nova embed`

```bash
nova embed <config> [--num-jobs N] [--job-rank R] [--dry-run]
nova embed predict <config> [flags]
```

Common flags:

| Flag | Meaning |
|---|---|
| `--num-jobs` | Total workers |
| `--job-rank` | This worker's rank |
| `--dry-run` | Preview config and work split |
| `--gpu` | GPU model for `predict` |
| `--num-gpus` | GPUs used for estimate |
| `--rate` | Hourly GPU cost override |
| `--output PATH` | Write prediction results as JSON |

`predict` estimates embedding throughput, runtime, and cost without running the workload.

Config: [Embed](../embedding.md).

## `nova bf`

```bash
nova bf compute <config> [--num-jobs N] [--job-rank R] [flags]
nova bf merge <config> [-j N] [--search NAME]...
```

Common `compute` flags:

| Flag | Meaning |
|---|---|
| `--num-jobs` | Total workers |
| `--job-rank` | This worker's rank |
| `--io-workers` | Override I/O workers |
| `--io-thread-count` | Override I/O threads |
| `--cpu-thread-count` | Override CPU threads |
| `--max-files` | Limit files for benchmarking; output is not valid full ground truth |

`merge` combines distributed partial results.

Config: [Brute Force](../brute-force/overview.md).

## `nova load`

```bash
nova load run <config>
nova load prepare <config>
nova load load <config> --num-jobs N --job-rank R [--continue]
nova load finalize <config>
nova load reindex <config>
nova load delete <config>
nova load inspect <config>
```

| Flag | Meaning |
|---|---|
| `--num-jobs` | Total workers |
| `--job-rank` | This worker's rank |
| `--continue` | Resume an interrupted load |

Optional backends can be built with:

```bash
make load LOAD_FEATURES=elastic,opensearch,milvus
```

Config: [Load](../loading/overview.md).

## `nova storm`

```bash
nova storm <config> [--json]
```

`--json` emits one machine-readable summary line. Every distributed worker runs the same workload.

Optional backends:

```bash
make storm STORM_FEATURES=elastic,opensearch,milvus
```

Config: [Storm](../storm/overview.md).

## `nova sweep`

```bash
nova sweep <config> [--skip-insert] [--cleanup] [--dry-run]
```

| Flag | Meaning |
|---|---|
| `--skip-insert` | Reuse existing collections |
| `--cleanup` | Delete collections created by the run |
| `--dry-run` | Preview the sweep without executing it |

Config: [Sweep](../sweep/overview.md).

## `nova dist`

Launches distributed runs with SkyPilot.

```bash
nova dist embed <config> --num-jobs N
nova dist load <config> --num-jobs N [--continue]
nova dist load <config> --finalize
nova dist bf compute <config> --num-jobs N
nova dist bf merge <config>
nova dist storm <config> --num-jobs N
```

Common flags:

| Flag | Meaning |
|---|---|
| `--num-jobs` | Number of workers |
| `--pool-name` | SkyPilot pool name |
| `--resources FILE` | Resource configuration |
| `--dry-run` | Preview without launching |
| `--continue` | Resume distributed loading |
| `--finalize` | Finalize a distributed load |

`nova dist sweep` is not currently implemented.

Details: [Distributed](../distributed.md).

## `nova inspect`

```bash
nova inspect <path>
```

Reports Parquet file count, vector count, schema, and vector dimensions for local or S3 inputs.

Install with:

```bash
make inspect
```

## Local Development

`nova` runs the first matching `nova-<cmd>` on `PATH`, so put local builds first:

```bash
cargo build -p nova-load
export PATH="$PWD/target/debug:$PATH"
nova --help
```

Python tools installed with `make` use editable installs, so code changes apply without reinstalling.