# Distributed

Distributed runs use `--num-jobs` and `--job-rank`. Each worker runs independently; there is no central coordinator.

You can launch workers yourself or use `nova dist` with SkyPilot.

| Tool | Distribution |
|---|---|
| `nova embed` | Splits input rows/files across workers |
| `nova bf compute` | Splits corpus files across workers |
| `nova load load` | Splits files across workers |
| `nova storm` | Every worker runs the same workload |

## Manual Launch

If your scheduler already provides worker ranks:

```bash
# embed
nova embed cfg.yaml --num-jobs 50 --job-rank $RANK

# load
nova load prepare cfg.yaml
nova load load cfg.yaml --num-jobs 50 --job-rank $RANK
nova load finalize cfg.yaml

# brute force
nova bf compute cfg.yaml --num-jobs 8 --job-rank $RANK
nova bf merge cfg.yaml

# storm
nova storm cfg.yaml
```

## `nova dist`

`nova dist` creates a SkyPilot worker pool and launches one job per rank.

```bash
make dist
```

Common options:

- `--num-jobs N`: number of workers
- `--pool-name NAME`: SkyPilot pool name
- `--resources FILE`: custom SkyPilot resources
- `--dry-run`: preview without launching

Pools are reused by name. Use a new `--pool-name` after changing worker configuration or resources if you want a fresh pool.

### Embed

```bash
nova dist embed configs/embedder/example.yaml --num-jobs 50
```

### Load

```bash
nova dist load configs/loader/example.yaml --num-jobs 50

# after workers finish
nova dist load configs/loader/example.yaml --finalize

# resume interrupted workers
nova dist load configs/loader/example.yaml --num-jobs 50 --continue
```

`--continue` requires the same files and `--num-jobs` as the original run.

### Brute Force

```bash
nova dist bf compute configs/brute_force/example.yaml --num-jobs 8

# after workers finish
nova dist bf merge configs/brute_force/example.yaml
```

`merge` requires a complete, consistent set of worker outputs.

### Storm

```bash
nova dist storm configs/storm/example.yaml --num-jobs 10
```

Every Storm worker runs the same workload, so aggregate load scales with the number of workers.

`nova dist sweep` is not currently implemented.

## Resources

SkyPilot resource settings are separate from tool configs.

Defaults:

| Tool | Default |
|---|---|
| `embed`, `bf` | AWS A10G GPU |
| `load` | AWS, 8+ CPUs |
| `storm` | AWS, 4+ CPUs |

Resource configuration is selected in this order:

1. `--resources FILE`
2. `~/.nova/skypilot/<tool>.yaml`
3. built-in defaults

Templates are available in `configs/skypilot/`.

A resource file may override SkyPilot `resources`, `setup`, and `envs`.

## Environment Variables

Environment variables referenced as `${VAR}` in tool configs are forwarded to workers.

Common credentials include:

- `QDRANT_URL` / `QDRANT_API_KEY` for load and Storm
- `HF_TOKEN` for embed and brute force
- `OPENAI_API_KEY` for embed

AWS credentials are handled by SkyPilot rather than forwarded directly.

## Dry Run

Use `--dry-run` to inspect generated SkyPilot jobs without launching them:

```bash
nova dist load configs/loader/example.yaml --num-jobs 50 --dry-run
```

Generated configs are written under `~/.nova/runs/`.

## Monitoring

```bash
sky jobs queue
sky jobs logs <job-id>
sky jobs pool down <pool>
```

`nova dist` prints the pool name when launching. Pools are not automatically deleted.