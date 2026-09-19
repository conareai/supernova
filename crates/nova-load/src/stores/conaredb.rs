//! ConareDB load backend (feature `conaredb`).
//!
//! Maps each [`Point`] to one ConareDB node in a namespace: the single dense
//! vector becomes the node's vector (L2-normalized, stored fp16), the point id
//! becomes the node's external `id`, and the payload lands as node properties
//! (a `text` property feeds the keyword index). Upserts go through the binary
//! bulk frame (`POST /v1/namespaces/<ns>/bulk-vectors`, `CRBF0002`), which is
//! the engine's bulk-load path: fp16 rows as raw bytes, one frame per batch,
//! with the `defer_compaction` hint so the engine builds its index once, at
//! the trailing compaction `enable_indexing` requests.
//!
//! Scope: **one dense vector + payload**. ConareDB stores one vector per node,
//! so a schema with several named vectors, or sparse / multivector values, is
//! rejected at `ensure_collection` / `upsert_batch`.

use std::fmt;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::config::VectorKind;
use crate::stores::{CollectionSchema, Point, PointId, StoreError, VectorStore, VectorValue};

const BULK_MAGIC_V2: &[u8; 8] = b"CRBF0002";

/// Connection + store settings for a ConareDB backend (`vectorstore: type: conaredb`).
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConareDbConfig {
    /// Engine URL, e.g. `http://10.0.0.4:8080`.
    pub url: String,
    /// Bearer token (`CONAREDB_AUTH_TOKEN`).
    #[serde(default)]
    pub api_key: Option<String>,
    /// Target namespace. Created on first write; `recreate` deletes it first.
    pub namespace: String,
    /// Node label for every point. Defaults to `doc`.
    #[serde(default = "default_label")]
    pub label: String,
    /// Declared final row count. Passed as the engine's `expect_rows` hint so
    /// its index build is sized for the whole load. Unset = no hint.
    #[serde(default)]
    pub expect_rows: Option<u64>,
    /// Per-request timeout in seconds. The trailing compaction that
    /// `enable_indexing` requests builds the index for the whole namespace in
    /// one call, so the default is generous (4 h); a plain upsert never waits
    /// that long.
    #[serde(default = "default_timeout_s")]
    pub timeout_s: u64,
    /// Delete the namespace if it exists before loading.
    #[serde(default)]
    pub recreate: bool,
}

fn default_label() -> String {
    "doc".to_string()
}

fn default_timeout_s() -> u64 {
    4 * 3600
}

/// Manual `Debug` so the token never reaches logs / `--dry-run`.
impl fmt::Debug for ConareDbConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConareDbConfig")
            .field("url", &self.url)
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("namespace", &self.namespace)
            .field("label", &self.label)
            .field("expect_rows", &self.expect_rows)
            .field("timeout_s", &self.timeout_s)
            .field("recreate", &self.recreate)
            .finish()
    }
}

pub struct ConareDbStore {
    client: reqwest::Client,
    base: String,
    token: Option<String>,
    namespace: String,
    label: String,
    expect_rows: Option<u64>,
    recreate: bool,
    /// When `enable_indexing` / `reindex` finished their synchronous
    /// compaction: `wait_for_indexing` reports it as the converged instant.
    converged_at: Mutex<Option<Instant>>,
}

fn other(message: impl Into<String>) -> StoreError {
    StoreError::Other(message.into())
}

fn to_other<E: std::error::Error>(e: E) -> StoreError {
    let mut msg = e.to_string();
    let mut src = e.source();
    while let Some(s) = src {
        msg.push_str(&format!(": {s}"));
        src = s.source();
    }
    StoreError::Other(msg)
}

impl ConareDbConfig {
    pub async fn connect(self) -> Result<ConareDbStore, StoreError> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(self.timeout_s))
            .build()
            .map_err(to_other)?;
        let base = self.url.trim_end_matches('/').to_string();
        let store = ConareDbStore {
            client,
            base,
            token: self.api_key,
            namespace: self.namespace,
            label: self.label,
            expect_rows: self.expect_rows,
            recreate: self.recreate,
            converged_at: Mutex::new(None),
        };
        // Connection errors surface at startup, not mid-load.
        let resp = store.request(reqwest::Method::GET, "/metrics").send().await.map_err(to_other)?;
        if !resp.status().is_success() {
            return Err(other(format!("conaredb /metrics HTTP {}", resp.status())));
        }
        Ok(store)
    }
}

impl fmt::Display for ConareDbStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "conaredb({})", self.namespace)
    }
}

impl ConareDbStore {
    fn request(&self, method: reqwest::Method, path: &str) -> reqwest::RequestBuilder {
        let mut req = self.client.request(method, format!("{}{}", self.base, path));
        if let Some(token) = &self.token {
            req = req.bearer_auth(token);
        }
        req
    }

    // An empty `namespace` addresses the engine's root store (`conaredb serve
    // --path <store>` with no namespaces): the same operations at `/v1/<op>`.
    fn ns_path(&self, op: &str) -> String {
        if self.namespace.is_empty() {
            format!("/v1/{op}")
        } else {
            format!("/v1/namespaces/{}/{}", self.namespace, op)
        }
    }

    fn id_string(id: &PointId) -> String {
        match id {
            PointId::Integer(n) => n.to_string(),
            PointId::String(s) => s.clone(),
        }
    }

    /// One `CRBF0002` frame: header JSON, pad to 2 bytes, fp16 rows (row-major,
    /// little-endian, each row L2-normalized in f32 and rounded once).
    fn bulk_frame(&self, points: &[Point]) -> Result<Vec<u8>, StoreError> {
        let mut dims: Option<usize> = None;
        let mut ids = Vec::with_capacity(points.len());
        let mut props_rows = Vec::with_capacity(points.len());
        let mut values: Vec<u8> = Vec::new();
        for point in points {
            let id = Self::id_string(&point.id);
            if point.vectors.len() != 1 {
                return Err(other(format!(
                    "conaredb stores one vector per node; point {id} carries {}",
                    point.vectors.len()
                )));
            }
            let dense = match point.vectors.values().next() {
                Some(VectorValue::Dense(v)) => v,
                Some(_) => {
                    return Err(other(format!(
                        "conaredb supports dense vectors only; point {id} is sparse or multivector"
                    )));
                }
                None => return Err(other(format!("point {id} has no vector"))),
            };
            match dims {
                None => dims = Some(dense.len()),
                Some(d) if d != dense.len() => {
                    return Err(other(format!(
                        "point {id} has {} dims, batch has {d}",
                        dense.len()
                    )));
                }
                _ => {}
            }
            let norm = dense.iter().map(|x| x * x).sum::<f32>().sqrt();
            if !norm.is_finite() || norm == 0.0 {
                return Err(other(format!("point {id} has a zero or non-finite vector")));
            }
            values.reserve(dense.len() * 2);
            for x in dense {
                let h = half::f16::from_f32(x / norm);
                values.extend_from_slice(&h.to_le_bytes());
            }
            let mut row = serde_json::Map::new();
            row.insert("id".to_string(), Value::String(id.clone()));
            for (k, v) in &point.payload {
                if k != "id" {
                    row.insert(k.clone(), v.clone());
                }
            }
            ids.push(Value::String(id));
            props_rows.push(Value::Object(row));
        }
        let Some(dims) = dims else {
            return Err(other("empty batch"));
        };
        let header = json!({
            "dims": dims,
            "label": self.label,
            "ids": ids,
            "props": {},
            "props_rows": props_rows,
        });
        let blob = serde_json::to_vec(&header).map_err(to_other)?;
        let mut frame = Vec::with_capacity(12 + blob.len() + 1 + values.len());
        frame.extend_from_slice(BULK_MAGIC_V2);
        frame.extend_from_slice(&(blob.len() as u32).to_le_bytes());
        frame.extend_from_slice(&blob);
        if (12 + blob.len()) % 2 == 1 {
            frame.push(0);
        }
        frame.extend_from_slice(&values);
        Ok(frame)
    }

    /// A write request carrying only the compaction hint (`true` collapses the
    /// chain and builds the index at the trailing compaction; `"force"` rewrites
    /// regardless). Synchronous: the engine answers when the pass is done.
    async fn compact(&self, hint: Value) -> Result<Value, StoreError> {
        let body = json!({
            "request_type": "write",
            "query": {"queries": [], "returns": []},
            "parameters": {},
            "compact": hint,
        });
        let resp = self
            .request(reqwest::Method::POST, &self.ns_path("query"))
            .json(&body)
            .send()
            .await
            .map_err(to_other)?;
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(other(format!("conaredb compaction HTTP {status}: {text}")));
        }
        let value: Value = serde_json::from_str(&text).map_err(to_other)?;
        if let Some(error) = value.get("compaction_error") {
            return Err(other(format!("conaredb compaction failed: {error}")));
        }
        Ok(value)
    }
}

#[async_trait]
impl VectorStore for ConareDbStore {
    async fn ensure_collection(&self, schema: &CollectionSchema) -> Result<(), StoreError> {
        if schema.vectors.len() != 1 {
            return Err(other(format!(
                "conaredb stores one vector per node; the schema names {} vectors",
                schema.vectors.len()
            )));
        }
        for (name, spec) in &schema.vectors {
            if spec.kind != VectorKind::Dense {
                return Err(other(format!(
                    "conaredb supports dense vectors only; `{name}` is {:?}",
                    spec.kind
                )));
            }
        }
        if self.recreate {
            self.delete_collection().await?;
        }
        // Namespaces are created on first write; nothing to create here.
        Ok(())
    }

    async fn upsert_batch(&self, points: Vec<Point>) -> Result<(), StoreError> {
        if points.is_empty() {
            return Ok(());
        }
        let frame = self.bulk_frame(&points)?;
        let mut path = format!("{}?defer_compaction=1", self.ns_path("bulk-vectors"));
        if let Some(expect) = self.expect_rows {
            path.push_str(&format!("&expect_rows={expect}"));
        }
        // The engine sheds a frame with 503 `bulk_admission_deferred` while its
        // persister catches up; that is backpressure, retried here rather than
        // surfaced (the loader's own retry budget is for real failures).
        for attempt in 0..600u32 {
            let resp = self
                .request(reqwest::Method::POST, &path)
                .header("content-type", "application/octet-stream")
                .body(frame.clone())
                .send()
                .await
                .map_err(to_other)?;
            let status = resp.status();
            if status.is_success() {
                return Ok(());
            }
            let text = resp.text().await.unwrap_or_default();
            if status.as_u16() == 503 && text.contains("bulk_admission_deferred") {
                tokio::time::sleep(Duration::from_millis(500)).await;
                if attempt % 20 == 19 {
                    tracing::info!("{self}: bulk admission deferred, still waiting");
                }
                continue;
            }
            return Err(other(format!("conaredb bulk-vectors HTTP {status}: {text}")));
        }
        Err(other("conaredb bulk admission deferred for 300 s"))
    }

    async fn point_exists(&self, id: &PointId) -> Result<bool, StoreError> {
        let id = Self::id_string(id);
        let body = json!({
            "request_type": "read",
            "query": {"queries": [{"Query": {
                "name": "exists",
                "condition": null,
                "steps": [{"NWhere": {"Eq": ["id", {"Value": {"String": id}}]}}, {"ValueMap": ["id"]}],
            }}], "returns": []},
            "parameters": {},
        });
        let resp = self
            .request(reqwest::Method::POST, &self.ns_path("query"))
            .json(&body)
            .send()
            .await
            .map_err(to_other)?;
        let status = resp.status();
        if status.as_u16() == 404 {
            return Ok(false);
        }
        let text = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(other(format!("conaredb query HTTP {status}: {text}")));
        }
        let value: Value = serde_json::from_str(&text).map_err(to_other)?;
        Ok(value["results"]["exists"]
            .as_array()
            .is_some_and(|rows| !rows.is_empty()))
    }

    async fn close(&self) -> Result<(), StoreError> {
        Ok(())
    }

    async fn defer_indexing(&self) -> Result<(), StoreError> {
        // Every frame carries `defer_compaction=1`; nothing collection-wide to flip.
        Ok(())
    }

    async fn enable_indexing(&self, _schema: &CollectionSchema) -> Result<(), StoreError> {
        // The trailing compaction: collapses the load's chain and builds the
        // ANN index and the keyword postings for the whole namespace in one
        // synchronous pass.
        let started = Instant::now();
        let response = self.compact(json!(true)).await?;
        tracing::info!(
            "{self}: trailing compaction done in {:.1}s (compacted={})",
            started.elapsed().as_secs_f64(),
            response["compacted"]
        );
        *self.converged_at.lock().expect("converged_at poisoned") = Some(Instant::now());
        Ok(())
    }

    async fn wait_for_indexing(&self) -> Result<Instant, StoreError> {
        // The compaction call above is synchronous: the index is live when it
        // returns. Report that instant (a `finalize` on a worker that never
        // called `enable_indexing` gets "now", which is also correct: nothing
        // is pending).
        let converged = *self.converged_at.lock().expect("converged_at poisoned");
        Ok(converged.unwrap_or_else(Instant::now))
    }

    async fn reindex(&self, _schema: &CollectionSchema) -> Result<(), StoreError> {
        let started = Instant::now();
        self.compact(json!("force")).await?;
        tracing::info!(
            "{self}: forced rewrite done in {:.1}s",
            started.elapsed().as_secs_f64()
        );
        *self.converged_at.lock().expect("converged_at poisoned") = Some(Instant::now());
        Ok(())
    }

    async fn delete_collection(&self) -> Result<(), StoreError> {
        if self.namespace.is_empty() {
            return Err(other(
                "conaredb: the root store (empty namespace) cannot be deleted through the API; \
                 point `namespace` at a namespace or start the engine on an empty --path"
                    .to_string(),
            ));
        }
        let resp = self
            .request(
                reqwest::Method::DELETE,
                &format!("/v1/namespaces/{}", self.namespace),
            )
            .send()
            .await
            .map_err(to_other)?;
        let status = resp.status();
        if status.is_success() || status.as_u16() == 404 {
            return Ok(());
        }
        let text = resp.text().await.unwrap_or_default();
        Err(other(format!("conaredb delete namespace HTTP {status}: {text}")))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    #[test]
    fn bulk_frame_layout_matches_crbf0002() {
        let store = ConareDbStore {
            client: reqwest::Client::new(),
            base: "http://127.0.0.1:1".to_string(),
            token: None,
            namespace: "t".to_string(),
            label: "doc".to_string(),
            expect_rows: None,
            recreate: false,
            converged_at: Mutex::new(None),
        };
        let mut vectors = HashMap::new();
        vectors.insert("dense".to_string(), VectorValue::Dense(vec![3.0, 4.0]));
        let mut payload = serde_json::Map::new();
        payload.insert("text".to_string(), Value::String("alpha".to_string()));
        let point = Point {
            id: PointId::Integer(7),
            vectors,
            payload,
            shard_key: None,
        };
        let frame = store.bulk_frame(&[point]).unwrap();
        assert_eq!(&frame[0..8], BULK_MAGIC_V2);
        let header_len = u32::from_le_bytes(frame[8..12].try_into().unwrap()) as usize;
        let header: Value = serde_json::from_slice(&frame[12..12 + header_len]).unwrap();
        assert_eq!(header["dims"], json!(2));
        assert_eq!(header["ids"], json!(["7"]));
        assert_eq!(header["props_rows"][0]["text"], json!("alpha"));
        let mut offset = 12 + header_len;
        if offset % 2 == 1 {
            assert_eq!(frame[offset], 0);
            offset += 1;
        }
        let values = &frame[offset..];
        assert_eq!(values.len(), 4);
        let x = half::f16::from_le_bytes([values[0], values[1]]).to_f32();
        let y = half::f16::from_le_bytes([values[2], values[3]]).to_f32();
        assert!((x - 0.6).abs() < 1e-3 && (y - 0.8).abs() < 1e-3, "{x} {y}");
    }
}
