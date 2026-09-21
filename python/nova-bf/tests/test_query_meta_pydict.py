"""`_query_meta_pydict` converts query metadata without boxing the vectors.

Two properties, and the second exists because a reviewer reasonably asked
whether a missing id/payload/filter column could be swallowed here and only
surface later as a `KeyError` at `d[id_column]`.

It cannot: the caller hands the SAME `cols` list to `store.read_columns`, and
pyarrow refuses a column the file does not have before this function is ever
reached. These tests pin that, so the helper stays free to assume it -- and so
that if the read path ever becomes tolerant of missing columns, the failure
shows up here rather than as a confusing downstream KeyError.
"""
from __future__ import annotations

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq

from nova_bf.compute import _query_meta_pydict
from nova_bf.io import Store

DIM = 4


def _table(n=3):
    return pa.table({
        "qid": pa.array([f"q{i}" for i in range(n)]),
        "title": pa.array([f"t{i}" for i in range(n)]),
        "embedding": pa.array([np.arange(DIM, dtype=np.float32) + i
                               for i in range(n)],
                              type=pa.list_(pa.float32(), DIM)),
    })


def test_the_vector_column_is_excluded_but_the_rest_is_converted():
    t = _table()
    d = _query_meta_pydict(t, ["embedding", "qid", "title"], "embedding")
    assert set(d) == {"qid", "title"}, "the vector column must not be boxed"
    assert d["qid"] == ["q0", "q1", "q2"]
    assert d["title"] == ["t0", "t1", "t2"]


def test_values_are_unchanged_by_the_exclusion():
    """Whatever is kept must be exactly what `to_pydict()` would have given."""
    t = _table()
    full = t.to_pydict()
    d = _query_meta_pydict(t, ["embedding", "qid", "title"], "embedding")
    for c in ("qid", "title"):
        assert d[c] == full[c]


def test_a_column_missing_from_the_table_raises_here():
    """Not silently dropped. This is unreachable in production -- the read
    already refused -- but if it ever became reachable, it must be loud."""
    with pytest.raises(KeyError, match="does not exist"):
        _query_meta_pydict(_table(), ["embedding", "qid", "nope"], "embedding")


def test_the_read_refuses_a_missing_column_first(tmp_path):
    """The real guarantee: a bad `id_column` fails at READ time, naming the
    column, long before any metadata conversion. Reproduces the failure mode a
    typo actually produces."""
    pq.write_table(_table(), tmp_path / "q.parquet")
    store = Store(str(tmp_path))
    with pytest.raises(Exception, match="doi"):
        store.read_columns(str(tmp_path / "q.parquet"), ["embedding", "doi"])


def test_no_metadata_columns_at_all_is_an_empty_dict():
    """A query file with only vectors and no id/payload/filter columns."""
    t = _table()
    assert _query_meta_pydict(t, ["embedding"], "embedding") == {}
