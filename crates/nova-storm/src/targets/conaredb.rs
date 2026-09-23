//! ConareDB v2 implementation of [`QueryTarget`] (feature `conaredb`).
//!
//! Dense nearest-neighbour search via `POST /v2/search` (`conaredb-server`, one
//! namespace per process; see ConareDB's `crates/conaredb-v2/server/API.md`).
//! The server owns every ANN work budget and rejects unknown request fields, so
//! this target sends only `{vector, top_k}` and takes no `search_params`. The
//! engine has no multi-query endpoint, so a batch of N queries is N concurrent
//! requests over one connection pool and the batch latency is the wall of the
//! whole fan-out. Filters are not supported yet (rejected at construction).
//! Scores are the engine's exact fp16 rescore cosine (higher is better).
//!
//! ## Collapsed corpora: the `expand` proxy
//!
//! A corpus with exact-duplicate vectors can be served with every distinct
//! vector stored once: hit ids are then distinct-row ordinals (`id = row +
//! id_offset`), not the corpus's own ids. `expand` maps them back in the client:
//! for each returned distinct row, best first, it emits that row's corpus ids
//! from a CSR posting table (in posting order) until `top_k` ids are filled,
//! each carrying its row's score. The server is asked for `top_k` distinct rows,
//! so rows below the first `top_k` distinct rows are never consulted. This is a
//! DISCLOSED proxy, not the engine returning ids: the lookup (positioned reads
//! of the CSR files) runs inside the measured latency window, and which copy of
//! a duplicate group is returned is the posting order, not a search decision.
//!
//! CSR layout (little-endian, no headers):
//! * `offsets`: `D + 1` unsigned ints of `offsets_bytes` (default 8); row `d`'s
//!   postings are entries `offsets[d] .. offsets[d + 1]`.
//! * `postings`: with `uuids` set, ordinals of `postings_bytes` (default 8) into
//!   the `uuids` table; without it, the postings are the 16-byte ids themselves.
//! * `uuids` (optional): 16 raw bytes per ordinal, rendered as a lowercase
//!   8-4-4-4-12 uuid (the FineWeb `id` column's bytes, as in `ids.npy`).

use std::fmt;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use super::{BatchOutcome, QueryTarget, ScoringProfile};
use crate::config::QueryConfig;
use crate::errors::TargetError;
use crate::queries::QueryVector;

/// Connection + target settings for a ConareDB v2 server (`target: type: conaredb`).
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConareDbConfig {
    /// Server URL, e.g. `http://10.0.0.14:19305`.
    pub url: String,
    /// Bearer token (the server config's `bearer_token`).
    #[serde(default)]
    pub api_key: Option<String>,
    /// Per-request timeout in seconds. Unset = 300 s.
    #[serde(default = "default_timeout_s")]
    pub timeout_s: u64,
    /// Distinct-row -> corpus-id expansion for a collapsed corpus (see module docs).
    #[serde(default)]
    pub expand: Option<ExpandConfig>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpandConfig {
    pub offsets: String,
    #[serde(default = "default_int_bytes")]
    pub offsets_bytes: usize,
    pub postings: String,
    #[serde(default = "default_int_bytes")]
    pub postings_bytes: usize,
    #[serde(default)]
    pub uuids: Option<String>,
    /// Served hit id minus this = distinct row (`ids: "rows"` imports use 1).
    #[serde(default = "default_id_offset")]
    pub id_offset: u64,
}

fn default_timeout_s() -> u64 {
    300
}
fn default_int_bytes() -> usize {
    8
}
fn default_id_offset() -> u64 {
    1
}

impl fmt::Debug for ConareDbConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConareDbConfig")
            .field("url", &self.url)
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("timeout_s", &self.timeout_s)
            .field("expand", &self.expand)
            .finish()
    }
}

/// The opened CSR files. Positioned reads (`pread`) only: no mmap, no cache;
/// the OS page cache decides what is resident, and every lookup is counted in
/// the dispatch latency.
struct Expander {
    offsets: File,
    offsets_bytes: usize,
    rows: u64,
    postings: File,
    postings_bytes: usize,
    postings_len: u64,
    uuids: Option<(File, u64)>,
    id_offset: u64,
}

fn open(path: &str) -> Result<(File, u64), TargetError> {
    let f = File::open(path).map_err(|e| TargetError::Other(format!("conaredb expand: {path}: {e}")))?;
    let len = f
        .metadata()
        .map_err(|e| TargetError::Other(format!("conaredb expand: {path}: {e}")))?
        .len();
    Ok((f, len))
}

fn le_uint(b: &[u8]) -> u64 {
    let mut v = [0u8; 8];
    v[..b.len()].copy_from_slice(b);
    u64::from_le_bytes(v)
}

fn uuid_string(b: &[u8]) -> String {
    let h: String = b.iter().map(|x| format!("{x:02x}")).collect();
    format!("{}-{}-{}-{}-{}", &h[0..8], &h[8..12], &h[12..16], &h[16..20], &h[20..32])
}

impl Expander {
    fn new(c: &ExpandConfig) -> Result<Self, TargetError> {
        for (name, w) in [("offsets_bytes", c.offsets_bytes), ("postings_bytes", c.postings_bytes)] {
            if !(1..=8).contains(&w) {
                return Err(TargetError::Other(format!("conaredb expand: {name} must be 1..=8 (got {w})")));
            }
        }
        let (offsets, olen) = open(&c.offsets)?;
        if olen == 0 || olen % c.offsets_bytes as u64 != 0 {
            return Err(TargetError::Other(format!(
                "conaredb expand: {} is {olen} bytes, not a whole number of {}-byte offsets",
                c.offsets, c.offsets_bytes
            )));
        }
        let rows = olen / c.offsets_bytes as u64 - 1;
        let (postings, plen) = open(&c.postings)?;
        let uuids = c.uuids.as_deref().map(open).transpose()?;
        let entry = if uuids.is_some() { c.postings_bytes as u64 } else { 16 };
        if plen % entry != 0 {
            return Err(TargetError::Other(format!(
                "conaredb expand: {} is {plen} bytes, not a whole number of {entry}-byte postings",
                c.postings
            )));
        }
        if let Some((_, ulen)) = &uuids
            && ulen % 16 != 0
        {
            return Err(TargetError::Other(format!("conaredb expand: uuid table is {ulen} bytes, not 16-byte ids")));
        }
        let e = Expander {
            offsets,
            offsets_bytes: c.offsets_bytes,
            rows,
            postings,
            postings_bytes: entry as usize,
            postings_len: plen / entry,
            uuids: uuids.map(|(f, l)| (f, l / 16)),
            id_offset: c.id_offset,
        };
        // The last offset must close the postings file exactly (a partial CSR is
        // allowed only if it says so by being internally consistent).
        let last = e.offset(rows).map_err(TargetError::Other)?;
        if last != e.postings_len {
            return Err(TargetError::Other(format!(
                "conaredb expand: offsets[{rows}] = {last} but the postings file holds {} entries",
                e.postings_len
            )));
        }
        tracing::info!(
            "conaredb expand: {rows} distinct rows, {} postings{} (disclosed expansion proxy, in the measured path)",
            e.postings_len,
            e.uuids.as_ref().map(|(_, n)| format!(", {n}-id uuid table")).unwrap_or_default()
        );
        Ok(e)
    }

    fn offset(&self, row: u64) -> Result<u64, String> {
        let mut b = [0u8; 8];
        let w = self.offsets_bytes;
        self.offsets
            .read_exact_at(&mut b[..w], row * w as u64)
            .map_err(|e| format!("expand: offsets read at row {row}: {e}"))?;
        Ok(le_uint(&b[..w]))
    }

    /// Up to `want` corpus ids of distinct row `row`, in posting order.
    fn ids(&self, row: u64, want: usize, out: &mut Vec<String>) -> Result<usize, String> {
        if row >= self.rows {
            return Err(format!("expand: distinct row {row} >= the CSR's {} rows", self.rows));
        }
        let (start, end) = (self.offset(row)?, self.offset(row + 1)?);
        if start >= end || end > self.postings_len {
            return Err(format!("expand: row {row} has an empty or out-of-range posting list [{start}, {end})"));
        }
        let n = ((end - start) as usize).min(want);
        let w = self.postings_bytes;
        let mut buf = vec![0u8; n * w];
        self.postings
            .read_exact_at(&mut buf, start * w as u64)
            .map_err(|e| format!("expand: postings read at {start}: {e}"))?;
        for chunk in buf.chunks_exact(w) {
            match &self.uuids {
                None => out.push(uuid_string(chunk)),
                Some((table, len)) => {
                    let ord = le_uint(chunk);
                    if ord >= *len {
                        return Err(format!("expand: ordinal {ord} >= the uuid table's {len} ids"));
                    }
                    let mut u = [0u8; 16];
                    table
                        .read_exact_at(&mut u, ord * 16)
                        .map_err(|e| format!("expand: uuid read at ordinal {ord}: {e}"))?;
                    out.push(uuid_string(&u));
                }
            }
        }
        Ok(n)
    }
}

pub struct ConareDbTarget {
    client: reqwest::Client,
    url: String,
    token: Option<String>,
    top_k: u64,
    expand: Option<Expander>,
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
        if query.search_params.is_some() {
            return Err(TargetError::Other(
                "conaredb v2: the server owns every ANN budget (a request cannot override it); \
                 remove query.search_params and set leaves/rescore in the server config"
                    .to_string(),
            ));
        }
        let expand = self.expand.as_ref().map(Expander::new).transpose()?;
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(self.timeout_s))
            .build()
            .map_err(to_other)?;
        Ok(ConareDbTarget {
            client,
            url: format!("{}/v2/search", self.url.trim_end_matches('/')),
            token: self.api_key,
            top_k: query.top_k,
            expand,
            collect_ids: query.source.ground_truth_column.is_some(),
            collect_scores: std::sync::atomic::AtomicBool::new(true),
        })
    }
}

impl fmt::Display for ConareDbTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "conaredb({}{})", self.url, if self.expand.is_some() { ", expand" } else { "" })
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
    async fn one(&self, dense: &[f32]) -> Result<Value, (String, bool)> {
        let mut req = self.client.post(&self.url);
        if let Some(token) = &self.token {
            req = req.bearer_auth(token);
        }
        let resp = req
            .json(&json!({ "vector": dense, "top_k": self.top_k }))
            .send()
            .await
            .map_err(|e| (e.to_string(), e.is_timeout()))?;
        let status = resp.status();
        let text = resp.text().await.map_err(|e| (e.to_string(), false))?;
        if !status.is_success() {
            let code = status.as_u16();
            return Err((format!("search HTTP {status}: {text}"), code == 408 || code == 504));
        }
        serde_json::from_str(&text).map_err(|e| (format!("search: {e}: {text}"), false))
    }

    /// One response's (ids, scores), best first; expanded when configured.
    fn hits(&self, hits: &[Value]) -> Result<(Vec<String>, Vec<f32>), String> {
        let k = self.top_k as usize;
        let mut ids = Vec::with_capacity(k);
        let mut scores = Vec::with_capacity(k);
        for h in hits {
            let Some(id) = h["id"].as_str() else {
                return Err(format!("a hit has no string `id`: {h}"));
            };
            let score = h["score"].as_f64().unwrap_or(f64::NAN) as f32;
            match &self.expand {
                None => {
                    ids.push(id.to_string());
                    scores.push(score);
                }
                Some(x) => {
                    if ids.len() >= k {
                        break;
                    }
                    let served: u64 = id.parse().map_err(|_| format!("hit id `{id}` is not a row ordinal"))?;
                    let row = served
                        .checked_sub(x.id_offset)
                        .ok_or_else(|| format!("hit id {served} < id_offset {}", x.id_offset))?;
                    let n = x.ids(row, k - ids.len(), &mut ids)?;
                    scores.extend(std::iter::repeat_n(score, n));
                }
            }
        }
        Ok((ids, scores))
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
        let want_scores = self.collect_scores.load(std::sync::atomic::Ordering::Relaxed);
        let mut ids = Vec::with_capacity(queries.len());
        let mut scores = Vec::with_capacity(queries.len());
        for (i, result) in results.into_iter().enumerate() {
            let value = match result {
                Ok(v) => v,
                Err((message, timed_out)) => {
                    return fail(started, queries.len(), format!("query {i}: {message}"), timed_out);
                }
            };
            let Some(hits) = value["hits"].as_array() else {
                return fail(started, queries.len(), format!("query {i}: no `hits` array: {value}"), false);
            };
            // Expansion runs even when recall is off, so latency-only runs pay
            // the same in-path cost as recall runs.
            let (query_ids, query_scores) = match self.hits(hits) {
                Ok(x) => x,
                Err(e) => return fail(started, queries.len(), format!("query {i}: {e}"), false),
            };
            if self.collect_ids {
                ids.push(Some(query_ids));
                scores.push(want_scores.then_some(query_scores));
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
        // Stored vectors are L2-normalized fp16 and every hit is exactly
        // rescored (f32 query x fp16 row), so scores are cosine similarities in
        // fp16 storage precision. Expanded ids carry their distinct row's score.
        ScoringProfile {
            datatype: Some("float16".to_string()),
            distance: Some("cosine".to_string()),
            quantized: false,
        }
    }

    fn disable_score_collection(&self) {
        self.collect_scores.store(false, std::sync::atomic::Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &std::path::Path, name: &str, bytes: &[u8]) -> String {
        let p = dir.join(name);
        std::fs::write(&p, bytes).unwrap();
        p.to_string_lossy().into_owned()
    }

    fn uuid(i: u8) -> [u8; 16] {
        let mut u = [0u8; 16];
        u[0] = i;
        u[15] = 0xab;
        u
    }

    #[test]
    fn expands_rows_in_rank_then_posting_order_until_k() {
        let dir = tempfile::tempdir().unwrap();
        // 3 distinct rows: row0 -> ordinals [4, 1], row1 -> [0], row2 -> [2, 3, 5]
        let offs: Vec<u8> = [0u64, 2, 3, 6].iter().flat_map(|x| x.to_le_bytes()).collect();
        let posts: Vec<u8> = [4u64, 1, 0, 2, 3, 5].iter().flat_map(|x| x.to_le_bytes()).collect();
        let table: Vec<u8> = (0..6u8).flat_map(uuid).collect();
        let cfg = ExpandConfig {
            offsets: write(dir.path(), "o", &offs),
            offsets_bytes: 8,
            postings: write(dir.path(), "p", &posts),
            postings_bytes: 8,
            uuids: Some(write(dir.path(), "u", &table)),
            id_offset: 1,
        };
        let x = Expander::new(&cfg).unwrap();
        let t = ConareDbTarget {
            client: reqwest::Client::new(),
            url: String::new(),
            token: None,
            top_k: 4,
            expand: Some(x),
            collect_ids: true,
            collect_scores: std::sync::atomic::AtomicBool::new(true),
        };
        // served ids = row + 1, best first: row2, row0, row1
        let hits = vec![
            json!({"id": "3", "score": 0.9}),
            json!({"id": "1", "score": 0.8}),
            json!({"id": "2", "score": 0.7}),
        ];
        let (ids, scores) = t.hits(&hits).unwrap();
        let want: Vec<String> = [2u8, 3, 5, 4].iter().map(|&i| uuid_string(&uuid(i))).collect();
        assert_eq!(ids, want);
        assert_eq!(scores, vec![0.9, 0.9, 0.9, 0.8]);
        assert_eq!(ids[0], "02000000-0000-0000-0000-0000000000ab");
        assert!(t.hits(&[json!({"id": "9", "score": 0.1})]).is_err());
    }

    #[test]
    fn inline_uuid_postings_and_consistency_check() {
        let dir = tempfile::tempdir().unwrap();
        let offs: Vec<u8> = [0u32, 1, 2].iter().flat_map(|x| x.to_le_bytes()).collect();
        let posts: Vec<u8> = [uuid(7), uuid(8)].concat();
        let mut cfg = ExpandConfig {
            offsets: write(dir.path(), "o", &offs),
            offsets_bytes: 4,
            postings: write(dir.path(), "p", &posts),
            postings_bytes: 8,
            uuids: None,
            id_offset: 1,
        };
        let x = Expander::new(&cfg).unwrap();
        let mut out = Vec::new();
        assert_eq!(x.ids(1, 10, &mut out).unwrap(), 1);
        assert_eq!(out, vec![uuid_string(&uuid(8))]);
        // offsets that do not close the postings file are refused at startup
        cfg.postings = write(dir.path(), "p2", &uuid(7));
        assert!(Expander::new(&cfg).is_err());
    }
}
