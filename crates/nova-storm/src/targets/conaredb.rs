//! ConareDB implementation of [`QueryTarget`] (feature `conaredb`).
//!
//! Dense nearest-neighbour search via `POST /v1/namespaces/<ns>/search` with a
//! `vector` branch. The engine has no multi-query endpoint, so a batch of N
//! queries is N concurrent requests over one connection pool and the batch
//! latency is the wall of the whole fan-out (the same accounting as a single
//! `_msearch` round-trip elsewhere). Per-request ANN knobs (`nprobe`,
//! `oversample`, `rescore`) ride in `query.search_params`. Filters are not
//! supported yet (rejected at construction). Recall uses the returned hit ids;
//! scores are the engine's cosine similarities (higher is better).

use std::fmt;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use base64::Engine as _;
use serde::Deserialize;
use serde_json::{Value, json};

use super::{BatchOutcome, QueryTarget, ScoringProfile};
use crate::config::QueryConfig;
use crate::errors::TargetError;
use crate::queries::QueryVector;

/// Connection + target settings for a ConareDB backend (`target: type: conaredb`).
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConareDbConfig {
    /// Engine (or router) URL, e.g. `http://10.0.0.4:8080`.
    pub url: String,
    /// Bearer token (`CONAREDB_AUTH_TOKEN`).
    #[serde(default)]
    pub api_key: Option<String>,
    pub namespace: String,
    /// Per-request timeout in seconds. Unset = 300 s.
    #[serde(default = "default_timeout_s")]
    pub timeout_s: u64,
}

fn default_timeout_s() -> u64 {
    300
}

impl fmt::Debug for ConareDbConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConareDbConfig")
            .field("url", &self.url)
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("namespace", &self.namespace)
            .field("timeout_s", &self.timeout_s)
            .finish()
    }
}

/// ConareDB search-time tuning (`query.search_params` for a `conaredb` target).
#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct ConareDbSearchParams {
    /// Clusters probed by the IVF stage.
    #[serde(default)]
    pub nprobe: Option<u64>,
    /// Rerank depth multiplier (candidates = top_k × oversample).
    #[serde(default)]
    pub oversample: Option<u64>,
    /// Exact fp16 rerank depth in rows.
    #[serde(default)]
    pub rescore: Option<u64>,
}

pub struct ConareDbTarget {
    client: reqwest::Client,
    base: String,
    token: Option<String>,
    namespace: String,
    top_k: u64,
    params: ConareDbSearchParams,
    collect_ids: bool,
    collect_scores: std::sync::atomic::AtomicBool,
}

fn to_other<E: std::error::Error>(e: E) -> TargetError {
    let mut msg = e.to_string();
    let mut src = e.source();
    while let Some(s) = src {
        msg.push_str(&format!(": {s}"));
        src = s.source();
    }
    TargetError::Other(msg)
}

impl ConareDbConfig {
    pub async fn into_target(self, query: &QueryConfig) -> Result<ConareDbTarget, TargetError> {
        if query.filter.is_some() {
            return Err(TargetError::Other(
                "filters are not yet supported for the conaredb target".to_string(),
            ));
        }
        let params: ConareDbSearchParams = query
            .search_params
            .as_ref()
            .map(|v| serde_yaml::from_value(v.clone()))
            .transpose()
            .map_err(|e| TargetError::Other(format!("conaredb search_params: {e}")))?
            .unwrap_or_default();
        if let Some(rescore) = params.rescore
            && rescore < query.top_k
        {
            return Err(TargetError::Other(format!(
                "conaredb rescore ({rescore}) must be >= top_k ({})",
                query.top_k
            )));
        }
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(self.timeout_s))
            .build()
            .map_err(to_other)?;
        Ok(ConareDbTarget {
            client,
            base: self.url.trim_end_matches('/').to_string(),
            token: self.api_key,
            namespace: self.namespace,
            top_k: query.top_k,
            params,
            collect_ids: query.source.ground_truth_column.is_some(),
            collect_scores: std::sync::atomic::AtomicBool::new(true),
        })
    }
}

impl fmt::Display for ConareDbTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "conaredb({})", self.namespace)
    }
}

fn fail(started: Instant, n: usize, error: String, timed_out: bool) -> BatchOutcome {
    BatchOutcome {
        latency: started.elapsed(),
        ok: false,
        ids: vec![None; n],
        scores: vec![None; n],
        error: Some(error),
        timed_out,
    }
}

impl ConareDbTarget {
    fn body(&self, dense: &[f32]) -> Value {
        let mut bytes = Vec::with_capacity(dense.len() * 4);
        for x in dense {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        let mut vector = json!({
            "embedding_b64": base64::engine::general_purpose::STANDARD.encode(&bytes),
            "top_k": self.top_k,
        });
        if let Some(nprobe) = self.params.nprobe {
            vector["nprobe"] = json!(nprobe);
        }
        if let Some(oversample) = self.params.oversample {
            vector["oversample"] = json!(oversample);
        }
        if let Some(rescore) = self.params.rescore {
            vector["rescore"] = json!(rescore);
        }
        json!({ "branches": { "vector": vector } })
    }

    async fn one(&self, dense: &[f32]) -> Result<Value, (String, bool)> {
        let mut req = self.client.post(format!(
            "{}/v1/namespaces/{}/search",
            self.base, self.namespace
        ));
        if let Some(token) = &self.token {
            req = req.bearer_auth(token);
        }
        let resp = req
            .json(&self.body(dense))
            .send()
            .await
            .map_err(|e| (e.to_string(), e.is_timeout()))?;
        let status = resp.status();
        let text = resp.text().await.map_err(|e| (e.to_string(), false))?;
        if !status.is_success() {
            return Err((format!("search HTTP {status}: {text}"), status.as_u16() == 408));
        }
        serde_json::from_str(&text).map_err(|e| (format!("search: {e}: {text}"), false))
    }
}

#[async_trait]
impl QueryTarget for ConareDbTarget {
    async fn query_batch(&self, queries: &[&QueryVector]) -> BatchOutcome {
        let started = Instant::now();
        if queries.is_empty() {
            return BatchOutcome {
                latency: started.elapsed(),
                ok: true,
                ids: Vec::new(),
                scores: Vec::new(),
                error: None,
                timed_out: false,
            };
        }
        let mut denses = Vec::with_capacity(queries.len());
        for q in queries {
            let Some(dense) = q.vector.as_dense() else {
                return fail(
                    started,
                    queries.len(),
                    "the conaredb target does not support sparse queries".to_string(),
                    false,
                );
            };
            denses.push(dense);
        }
        let results = futures::future::join_all(denses.iter().map(|d| self.one(d))).await;
        let want_scores = self
            .collect_scores
            .load(std::sync::atomic::Ordering::Relaxed);
        let mut ids = Vec::with_capacity(queries.len());
        let mut scores = Vec::with_capacity(queries.len());
        for (i, result) in results.into_iter().enumerate() {
            let value = match result {
                Ok(v) => v,
                Err((message, timed_out)) => {
                    return fail(
                        started,
                        queries.len(),
                        format!("query {i}: {message}"),
                        timed_out,
                    );
                }
            };
            let Some(hits) = value["vector"]["hits"].as_array() else {
                return fail(
                    started,
                    queries.len(),
                    format!("query {i}: no `vector.hits` array: {value}"),
                    false,
                );
            };
            if self.collect_ids {
                let mut query_ids = Vec::with_capacity(hits.len());
                let mut query_scores = Vec::with_capacity(hits.len());
                for h in hits {
                    let Some(id) = h["id"].as_str() else {
                        return fail(
                            started,
                            queries.len(),
                            format!("query {i}: a hit has no string `id`: {h}"),
                            false,
                        );
                    };
                    query_ids.push(id.to_string());
                    if want_scores {
                        query_scores.push(h["score"].as_f64().unwrap_or(f64::NAN) as f32);
                    }
                }
                ids.push(Some(query_ids));
                scores.push(if want_scores { Some(query_scores) } else { None });
            } else {
                ids.push(None);
                scores.push(None);
            }
        }
        BatchOutcome {
            latency: started.elapsed(),
            ok: true,
            ids,
            scores,
            error: None,
            timed_out: false,
        }
    }

    async fn scoring_profile(&self) -> ScoringProfile {
        // Vectors are stored L2-normalized fp16; the vector branch reports the
        // cosine similarity (a dot product on normalized vectors), higher is
        // better. The IVF stage is quantized but every hit is exactly reranked,
        // so scores are exact-path scores.
        ScoringProfile {
            datatype: Some("float16".to_string()),
            distance: Some("cosine".to_string()),
            quantized: false,
        }
    }

    fn disable_score_collection(&self) {
        self.collect_scores
            .store(false, std::sync::atomic::Ordering::Relaxed);
    }
}
