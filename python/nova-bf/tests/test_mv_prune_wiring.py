"""End-to-end wiring for `params.multivector_prune`, on CPU.

The prune's ~130 lines of `compute.py` plumbing -- the config gate, the
vector-type gate, the `MultiVectorBatchSlice` gate, the `NOVA_BF_NO_PRUNE` kill
switch, the `score_cache` bypass and the decline path -- had NO executable
coverage on a machine without a GPU: every other test of this feature is
CUDA-gated. This file closes that, because the wiring is not CUDA-specific even
though the kernels are.

WHAT CPU CAN AND CANNOT SHOW. Without a GPU the float16 pass declines (the
two-pass certification refuses a non-CUDA device), so these cannot exercise the
bound or the kernels. What they CAN prove is the contract the whole design
rests on: **declining is invisible.** A run with the prune enabled must produce
byte-identical output to one with it off, must not raise, and must record the
decline rather than silently scoring something different. A wiring bug that
routed the prune into a dense search, or that let a declined prune leave
`scores` as `None`, or that shared pruned scores through `score_cache`, would
show up here as a changed answer or a crash.
"""
from __future__ import annotations

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq

from nova_bf import compute as cm
from nova_bf.compute import run_compute
from nova_bf.config import load_config

DIM = 8


def _mv_array(docs):
    """`list<list<float32>>` from a list of (n_tokens, DIM) arrays."""
    toks = [t for d in docs for t in d]
    inner = pa.ListArray.from_arrays(
        pa.array(list(range(0, len(toks) * DIM + 1, DIM)), type=pa.int32()),
        pa.array(np.concatenate(toks).astype(np.float32) if toks
                 else np.zeros(0, np.float32), type=pa.float32()),
    )
    outer, n = [0], 0
    for d in docs:
        n += len(d)
        outer.append(n)
    return pa.ListArray.from_arrays(pa.array(outer, type=pa.int32()), inner)


@pytest.fixture
def corpus_and_queries(tmp_path):
    rng = np.random.default_rng(5)
    cdir = tmp_path / "corpus"
    cdir.mkdir()
    # A zero-token document and a single-token one, because those are the rows
    # the prune's -inf handling is most likely to get wrong.
    docs = [rng.standard_normal((n, DIM)) for n in (3, 1, 5, 2)]
    docs.insert(1, np.zeros((0, DIM)))
    pq.write_table(
        pa.table({"id": pa.array([f"d{i}" for i in range(len(docs))]),
                  "multivector_embedding": _mv_array(docs)}),
        cdir / "c.parquet")
    queries = [rng.standard_normal((n, DIM)) for n in (2, 4, 1)]
    qpath = tmp_path / "q.parquet"
    pq.write_table(
        pa.table({"qid": pa.array([f"q{i}" for i in range(len(queries))]),
                  "multivector_embedding": _mv_array(queries)}),
        qpath)
    return cdir, qpath


def _run(tmp_path, cdir, qpath, prune, *, k=3, tag="r"):
    out = tmp_path / f"out_{tag}"
    out.mkdir(exist_ok=True)
    cfg = tmp_path / f"cfg_{tag}.yaml"
    cfg.write_text(f"""
corpus:
  path: {cdir}
  multivector_column: multivector_embedding
  id_column: id
queries:
  path: {qpath}
  multivector_column: multivector_embedding
  id_column: qid
output:
  path: {out}
params:
  io_workers: 1
  multivector_prune: {prune}
searches:
  - name: mv
    k: {k}
    metric: dot
    vector_type: multivector
""")
    cm._MV_PRUNE.update(pairs=0, pruned=0, calls=0, declined=0, sec=0.0)
    t = pq.read_table(run_compute(load_config(str(cfg)))["mv"]).to_pydict()
    return ({q: (tuple(h), tuple(round(float(x), 6) for x in s))
             for q, h, s in zip(t["query_id"], t["hit_ids"], t["hit_scores"])},
            dict(cm._MV_PRUNE))


def test_enabling_the_prune_does_not_change_the_answer(tmp_path,
                                                       corpus_and_queries):
    """The whole contract in one assertion. On CPU the prune declines, so this
    is really "declining is invisible" -- but that is exactly the property a
    wiring bug would break, and it holds on a GPU too because the bound is
    admissible."""
    cdir, qpath = corpus_and_queries
    off, _ = _run(tmp_path, cdir, qpath, "off", tag="off")
    on, _ = _run(tmp_path, cdir, qpath, "fp16", tag="on")
    assert on == off


def test_the_prune_is_reached_and_declines_on_cpu(tmp_path, corpus_and_queries):
    """Guard against the test above passing because the prune was never wired
    in at all: it must actually be CALLED, and must decline rather than run."""
    cdir, qpath = corpus_and_queries
    _, tally = _run(tmp_path, cdir, qpath, "fp16", tag="called")
    assert tally["calls"] > 0, "the prune was never reached from run_compute"
    assert tally["declined"] == tally["calls"], (
        "the prune claims to have run on a CPU-only machine")
    assert tally["pruned"] == 0


def test_off_never_reaches_the_prune(tmp_path, corpus_and_queries):
    cdir, qpath = corpus_and_queries
    _, tally = _run(tmp_path, cdir, qpath, "off", tag="never")
    assert tally["calls"] == 0


def test_the_kill_switch_disables_it(tmp_path, corpus_and_queries, monkeypatch):
    """`NOVA_BF_NO_PRUNE` is a pre-existing escape hatch; the multivector prune
    is gated on it at the call site and must honour it."""
    cdir, qpath = corpus_and_queries
    monkeypatch.setenv("NOVA_BF_NO_PRUNE", "1")
    _, tally = _run(tmp_path, cdir, qpath, "fp16", tag="killed")
    assert tally["calls"] == 0


def test_a_dense_search_never_reaches_the_multivector_prune(tmp_path):
    """The state is forwarded only for `vt == "multivector"`. A dense run with
    the flag set must not touch it -- and must not crash on the attempt."""
    rng = np.random.default_rng(1)
    cdir = tmp_path / "dense"
    cdir.mkdir()
    pq.write_table(pa.table({
        "id": pa.array([f"d{i}" for i in range(6)]),
        "embedding": pa.array(list(rng.standard_normal((6, DIM))),
                              type=pa.list_(pa.float32(), DIM)),
    }), cdir / "c.parquet")
    qpath = tmp_path / "q.parquet"
    pq.write_table(pa.table({
        "qid": pa.array(["q0", "q1"]),
        "embedding": pa.array(list(rng.standard_normal((2, DIM))),
                              type=pa.list_(pa.float32(), DIM)),
    }), qpath)
    out = tmp_path / "out"
    out.mkdir()
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"""
corpus:
  path: {cdir}
  dense_column: embedding
  id_column: id
queries:
  path: {qpath}
  dense_column: embedding
  id_column: qid
output:
  path: {out}
params:
  io_workers: 1
  multivector_prune: fp16
searches:
  - name: d
    k: 2
    metric: dot
    vector_type: dense
""")
    cm._MV_PRUNE.update(pairs=0, pruned=0, calls=0, declined=0, sec=0.0)
    run_compute(load_config(str(cfg)))
    assert cm._MV_PRUNE["calls"] == 0


def test_two_searches_sharing_a_metric_still_agree(tmp_path,
                                                   corpus_and_queries):
    """Pruned scores are member-specific (they depend on that member's running
    threshold) and must bypass `score_cache`. Two members over the same metric
    must each get their own correct answer, not one leaking into the other."""
    cdir, qpath = corpus_and_queries
    out = tmp_path / "out_two"
    out.mkdir()
    cfg = tmp_path / "cfg_two.yaml"
    cfg.write_text(f"""
corpus:
  path: {cdir}
  multivector_column: multivector_embedding
  id_column: id
queries:
  path: {qpath}
  multivector_column: multivector_embedding
  id_column: qid
output:
  path: {out}
params:
  io_workers: 1
  multivector_prune: fp16
searches:
  - name: a
    k: 2
    metric: dot
    vector_type: multivector
  - name: b
    k: 4
    metric: dot
    vector_type: multivector
""")
    res = run_compute(load_config(str(cfg)))
    a = pq.read_table(res["a"]).to_pydict()
    b = pq.read_table(res["b"]).to_pydict()
    # `a` asked for 2 and `b` for 4; `a`'s hits must be `b`'s first two.
    for ha, hb in zip(a["hit_ids"], b["hit_ids"]):
        assert list(ha) == list(hb)[:len(ha)]


def test_the_prune_is_on_by_default():
    """It is opt-OUT: a config that says nothing about it gets the prune.
    Safe because it declines wherever it cannot run, but it is a default that
    changes tie ordering, so pin it."""
    from nova_bf.config import ParamsConfig

    assert ParamsConfig().multivector_prune == "fp16"


def test_the_default_reaches_the_prune_without_being_asked(tmp_path,
                                                           corpus_and_queries):
    """The default must actually plumb through, not just parse."""
    cdir, qpath = corpus_and_queries
    out = tmp_path / "out_default"
    out.mkdir()
    cfg = tmp_path / "cfg_default.yaml"
    cfg.write_text(f"""
corpus:
  path: {cdir}
  multivector_column: multivector_embedding
  id_column: id
queries:
  path: {qpath}
  multivector_column: multivector_embedding
  id_column: qid
output:
  path: {out}
params:
  io_workers: 1
searches:
  - name: mv
    k: 3
    metric: dot
    vector_type: multivector
""")
    cm._MV_PRUNE.update(pairs=0, pruned=0, calls=0, declined=0, sec=0.0)
    run_compute(load_config(str(cfg)))
    assert cm._MV_PRUNE["calls"] > 0, "the default did not reach the prune"


def test_the_setting_is_recorded_in_the_output_metadata(tmp_path,
                                                        corpus_and_queries):
    """A ground-truth file must say whether it was produced with the prune:
    two files that disagree only in tie order should be able to explain why."""
    cdir, qpath = corpus_and_queries
    for mode in ("off", "fp16"):
        out = tmp_path / f"out_meta_{mode}"
        out.mkdir()
        cfg = tmp_path / f"cfg_meta_{mode}.yaml"
        cfg.write_text(f"""
corpus:
  path: {cdir}
  multivector_column: multivector_embedding
  id_column: id
queries:
  path: {qpath}
  multivector_column: multivector_embedding
  id_column: qid
output:
  path: {out}
params:
  io_workers: 1
  multivector_prune: {mode}
searches:
  - name: mv
    k: 3
    metric: dot
    vector_type: multivector
""")
        res = run_compute(load_config(str(cfg)))
        meta = pq.read_schema(res["mv"]).metadata
        assert meta[b"nova_bf.multivector_prune"] == mode.encode()


def test_bare_off_in_yaml_is_accepted():
    """YAML 1.1 parses a bare `off` as the boolean False, so the documented
    default spelling would otherwise be a ValidationError whose message says
    `input_value=False` and never mentions quoting. Found by writing the
    obvious config in the tests above."""
    from nova_bf.config import ParamsConfig

    assert ParamsConfig(multivector_prune=False).multivector_prune == "off"
    assert ParamsConfig(multivector_prune="off").multivector_prune == "off"
    assert ParamsConfig(multivector_prune="fp16").multivector_prune == "fp16"


def test_bare_on_in_yaml_is_still_rejected():
    """`on` is NOT mapped: there could be more than one way to enable this, so
    silently choosing one would be guessing at intent."""
    import pydantic

    from nova_bf.config import ParamsConfig

    with pytest.raises(pydantic.ValidationError):
        ParamsConfig(multivector_prune=True)
