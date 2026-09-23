# Parameter Sweeps

`nova sweep` runs `nova load` and `nova storm` across a matrix of collection, index, and search settings, producing one combined report of recall, latency, and throughput.

Ground truth must already exist in `nova bf` format; `nova sweep` does not compute it.

Supported targets are `qdrant`, `milvus`, and `elastic`.

## Config

```yaml
collection_name: my_model_sweep

corpus:
  path: s3://my-bucket/dataset/model
  dense_column: dense_embedding

queries:
  uri: s3://my-bucket/dataset/model/queries.parquet
  column: dense_embedding
  ground_truth_column: hit_ids
  ground_truth_score_column: hit_scores
  limit: 1000

target:
  type: qdrant
  url: ${QDRANT_URL}
  api_key: ${QDRANT_API_KEY}

data_layouts:
  vectors.dense.datatype: [float32, uint8]

index_variants:
  quantization.type: [none, scalar]
  hnsw.m: [8, 16, 32]
  hnsw.ef_construct: [64, 128]

searches:
  top_k: [10]
  hnsw_ef: [64, 128]
  batch_size: [1, 8]

output:
  path: s3://my-bucket/sweeps/model/
```

Environment variables support `${VAR}` and `${VAR:-default}` expansion.

Each sweep currently supports one dense vector named `dense`.

## Running

```bash
nova sweep configs/sweep/my_sweep.yaml
nova sweep configs/sweep/my_sweep.yaml --dry-run
nova sweep configs/sweep/my_sweep.yaml --skip-insert
nova sweep configs/sweep/my_sweep.yaml --cleanup
```

- `--dry-run`: preview the sweep without executing it.
- `--skip-insert`: reuse existing collections.
- `--cleanup`: delete collections created by the run.

## Sweep Axes

`data_layouts`, `index_variants`, and `searches` define Cartesian-product parameter grids using dotted config paths:

```yaml
index_variants:
  hnsw.m: [8, 16, 32]
  hnsw.ef_construct: [64, 128]
```

This produces six index configurations.

A YAML `null` omits that setting from the generated config.

### `data_layouts`

Settings that require creating a new collection belong here, such as vector datatype, distance, size, or shard layout.

Each data layout is loaded once.

### `index_variants`

Settings that can be changed after loading belong here, such as HNSW, quantization, and other index parameters.

Each variant reindexes the existing collection instead of reloading the corpus.

### `searches`

Search and workload parameters are swept without modifying the collection.

Common examples include:

```yaml
searches:
  top_k: [10, 100]
  hnsw_ef: [64, 128]
  concurrency: [16, 32]
  batch_size: [1, 8]
```

Backend-specific search parameters include:

- Qdrant: `hnsw_ef`, `exact`, `quantization`
- Milvus: `ef`, `nprobe`
- Elasticsearch: `num_candidates`

## Execution

For each data layout, Sweep:

1. Loads the corpus once.
2. Applies each index variant using `nova load reindex`.
3. Runs every search configuration using `nova storm`.
4. Records accuracy and performance metrics.

Only changes in `data_layouts` require reloading the corpus.

## Collections

If a target collection already exists, the sweep fails by default.

Use:

```bash
nova sweep my_sweep.yaml --skip-insert
```

to reuse it, or:

```yaml
target:
  recreate: always
```

to delete and rebuild it.

Collections are kept after the run unless `--cleanup` is specified.

## Output

Results are written to:

```text
<output.path>/sweep_results.parquet
```

Each row represents one:

```text
data_layout × index_variant × search
```

and includes:

- tested parameter values
- reindex and search time
- request and query throughput
- p50, p95, p99, and max latency
- recall and RBO metrics
- tie-aware recall metrics
- success or error information

Failed sweep points are recorded as error rows when possible, and remaining points continue.

## Distribution

`nova sweep` currently runs sequentially on one machine. Distributed sweep execution is not yet supported.