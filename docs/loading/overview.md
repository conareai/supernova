

# Load

`nova load` reads pre-embedded Parquet files from S3 or local disk and loads them into a vector store. Qdrant is built in; Elasticsearch, OpenSearch, and Milvus are optional.

![Loading Pipeline](../fig/ingestion_pipelione.svg)

## Config

```yaml
datasource:
  type: s3
  path: s3://my-bucket/dataset/model
  id_expression: "vf_point_id(filename, file_row_number)"
  payload_fields:
    text: text
    source: source

vectors:
  dense:
    type: dense
    column: dense_embedding
    distance: cosine
  sparse:
    type: sparse
    column: sparse_embedding

vectorstore:
  type: qdrant
  url: ${QDRANT_URL}
  api_key: ${QDRANT_API_KEY}
  collection_name: my-collection

loader:
  batch_size: 256
  concurrency: 8
```

Environment variables support `${VAR}` and `${VAR:-default}` expansion.

## Running

Single machine:

```bash
nova load run my.yaml
```

Distributed:

```bash
nova load prepare my.yaml
nova load load my.yaml --num-jobs 32 --job-rank $RANK
nova load finalize my.yaml
```

Workers independently process subsets of the input files. `nova dist load` can provision and launch the workers automatically.

Other commands:

```bash
nova load inspect my.yaml   # dry run
nova load reindex my.yaml   # rebuild with new index settings
nova load delete my.yaml    # delete the collection
```

### Resuming

Use `--continue` to resume an interrupted load:

```bash
nova load load my.yaml --num-jobs 32 --job-rank $RANK --continue
```

Resuming requires the same files, worker count, and a deterministic `id_expression`.

## Datasources

S3:

```yaml
datasource:
  type: s3
  path: s3://my-bucket/dataset/model
```

Local:

```yaml
datasource:
  type: local
  path: /data/dataset/model
```

`file_list` may be used to restrict either source to specific files.

## Point IDs

`id_expression` is a DuckDB expression evaluated for each row. The default `uuid()` generates new IDs on every load.

For stable IDs compatible with `nova bf` ground truth:

```yaml
datasource:
  id_expression: "vf_point_id(filename, file_row_number)"
```

An existing ID column or other DuckDB expression may also be used.

## Payload

`payload_fields` maps stored payload names to DuckDB expressions:

```yaml
payload_fields:
  text: text
  title_upper: upper(title)
```

String, integer, float, and boolean values are supported directly. Cast other types when needed.

## Vectors

Each entry under `vectors` defines a named vector.

| Key | Description |
|---|---|
| `type` | `dense`, `sparse`, or `multivector` |
| `column` | Parquet column |
| `distance` | `cosine`, `dot`, `euclid`, or `manhattan` |
| `size` | Vector dimension; inferred if omitted |
| `datatype` | `float32`, `float16`, or `uint8` |
| `on_disk` | Store vectors on disk |
| `comparator` | Multivector comparator; default `max_sim` |
| `modifier` | Sparse modifier: `none` or `idf` |

## Qdrant Settings

Connection settings are configured under `vectorstore`:

```yaml
vectorstore:
  type: qdrant
  url: http://localhost:6334
  collection_name: my-collection
```

Collection and index settings go under `vectorstore.params`:

```yaml
vectorstore:
  params:
    shard_number: 6
    replication_factor: 2
    on_disk_payload: true

    hnsw:
      m: 16
      ef_construct: 100
      on_disk: true

    quantization:
      type: scalar
      quantile: 0.99

    optimizers:
      indexing_threshold: 20000
```

Unset values use Qdrant defaults.

Supported quantization types are `scalar`, `product`, `binary`, `turbo`, and `none`.

### Reindexing

`nova load reindex` updates HNSW, quantization, and optimizer settings on an existing collection without reloading the data.

```bash
nova load reindex my.yaml
```

## Custom Sharding

`custom_sharding` routes points using a DuckDB expression:

```yaml
vectorstore:
  custom_sharding:
    shard_key: "org_id"
    shards_number: 2
    replication_factor: 2
```

Shard keys must be strings or non-negative integers, and the expression should be deterministic.

## Other Backends

Optional backends support dense vectors:

- `opensearch`
- `elastic`
- `milvus`

Build support with:

```bash
make load LOAD_FEATURES=elastic,opensearch,milvus
```

Backend-specific connection and index settings are configured under `vectorstore`.

## Tuning and Failures

Common loader settings:

| Key | Default | Description |
|---|---:|---|
| `batch_size` | `256` | Points per upsert |
| `concurrency` | CPUs - 1 | Upserts in flight |
| `file_look_ahead` | `2` | Files prepared ahead of upload |
| `file_retries` | `5` | Retries before a file is skipped |
| `upsert_retries` | `5` | Retries before the run aborts |
| `max_points_per_sec` | unlimited | Per-worker rate limit |

If the vector store is overloaded, increase `vectorstore.timeout_s`, then reduce `concurrency` or set `max_points_per_sec`.

Loader limits are per worker, so fleet-wide throughput scales with the number of workers.