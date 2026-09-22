"""`Store.read_columns`'s opt-in parallel ranged-read path.

The path is built from FileSystem API (`open_input_file`/`read_at`), so it must
produce tables identical to the normal reader on a POSIX root. These tests
monkeypatch the size constants so a tiny fixture file exercises the multi-range
path; the real _RANGED_GET_MIN_BYTES (256 MB) exists to keep small files off it
in production, not because the path is wrong below that size.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from nova_bf import io as io_mod
from nova_bf.io import Store


@pytest.fixture
def corpus(tmp_path):
    """A local parquet root holding one multi-column file."""
    rng = np.random.default_rng(0)
    table = pa.table({
        "id": pa.array(range(512)),
        "vector": pa.array(rng.random((512, 16)).tolist(), pa.list_(pa.float32())),
        "text": pa.array([f"doc-{i}" for i in range(512)]),
    })
    store = Store(str(tmp_path))
    store.write("corpus.parquet", table)
    return store, store.list_parquets()[0].read_path


def _force_ranged(monkeypatch, chunk_bytes: int = 4096) -> None:
    """Make every file big enough to qualify, split into many small ranges."""
    monkeypatch.setattr(io_mod, "_RANGED_GET_MIN_BYTES", 0)
    monkeypatch.setattr(io_mod, "_RANGED_GET_BYTES", chunk_bytes)


def test_ranged_read_matches_normal_read_on_posix_root(corpus, monkeypatch):
    _, path = corpus
    plain = Store(path, ranged_get=False).read_columns(path, columns=None)

    _force_ranged(monkeypatch)
    ranged = Store(path, ranged_get=True).read_columns(path, columns=None)

    assert ranged.equals(plain)


def test_ranged_read_honors_column_projection(corpus, monkeypatch):
    _, path = corpus
    plain = Store(path, ranged_get=False).read_columns(path, columns=["id", "text"])

    _force_ranged(monkeypatch)
    ranged = Store(path, ranged_get=True).read_columns(path, columns=["id", "text"])

    assert ranged.column_names == ["id", "text"]
    assert ranged.equals(plain)


def test_ranged_path_is_actually_taken_on_a_local_file(corpus, monkeypatch):
    """Guards the gate itself: an equality assertion alone would still pass if
    `read_columns` quietly fell back to the normal reader."""
    _, path = corpus
    _force_ranged(monkeypatch)
    store = Store(path, ranged_get=True)

    calls: list[int] = []
    real = store._ranged_download
    monkeypatch.setattr(
        store, "_ranged_download", lambda p, size: (calls.append(size), real(p, size))[1]
    )
    store.read_columns(path, columns=["id"])

    assert len(calls) == 1
    assert calls[0] > 0


def test_multiple_ranges_are_issued(corpus, monkeypatch):
    """A chunk size larger than the file would exercise a single read and prove
    nothing about the concurrent assembly, so pin the chunk small enough that
    the buffer is stitched from several ranges."""
    _, path = corpus
    _force_ranged(monkeypatch, chunk_bytes=1024)
    store = Store(path, ranged_get=True)
    size = store.fs.get_file_info(path).size
    assert size > 1024, "fixture too small to split into multiple ranges"

    assert store.read_columns(path, columns=None).num_rows == 512


def test_small_local_file_skips_the_ranged_path(corpus):
    """With the real threshold in force, a tiny file takes the normal reader."""
    _, path = corpus
    store = Store(path, ranged_get=True)
    assert store.fs.get_file_info(path).size < io_mod._RANGED_GET_MIN_BYTES

    def boom(*_args, **_kwargs):
        raise AssertionError("ranged path taken below _RANGED_GET_MIN_BYTES")

    store._ranged_download = boom
    assert store.read_columns(path, columns=["id"]).num_rows == 512


def test_ranged_get_defaults_off(corpus):
    _, path = corpus
    assert Store(path).ranged_get is False


# --- the per-call `ranged` override -------------------------------------------
#
# The store-wide `ranged_get` is the right default for the vector scan, which
# wants most of a file's bytes. It is the wrong default for a caller that wants
# one narrow column, because the ranged path buys the WHOLE file first. These
# pin the override that lets such a caller opt out. See `read_columns`.

def _counting_download(store, monkeypatch):
    """Record every whole-file download `store` performs."""
    calls: list[int] = []
    real = store._ranged_download
    monkeypatch.setattr(
        store, "_ranged_download",
        lambda p, size: (calls.append(size), real(p, size))[1],
    )
    return calls


def test_ranged_false_skips_the_whole_file_download(corpus, monkeypatch):
    """THE point of the override: no whole-file buffer is allocated at all.

    Without it, reading one narrow column off a 4.3 GB pubmed shard downloaded
    all 4.3 GB to keep ~180 KB of ids -- 1055 s per rank, and at a 32-way
    fan-out ~137 GB resident, which OOM-killed a 16-rank GT run.
    """
    _, path = corpus
    _force_ranged(monkeypatch)
    store = Store(path, ranged_get=True)
    calls = _counting_download(store, monkeypatch)

    got = store.read_columns(path, columns=["id"], ranged=False)

    assert calls == [], "the whole file was downloaded despite ranged=False"
    assert got.column_names == ["id"]


def test_ranged_false_returns_the_same_table_as_the_ranged_path(corpus, monkeypatch):
    """Opting out changes how the bytes arrive, never what comes back."""
    _, path = corpus
    _force_ranged(monkeypatch)
    store = Store(path, ranged_get=True)

    assert (store.read_columns(path, columns=["id"], ranged=False)
            .equals(store.read_columns(path, columns=["id"], ranged=True)))


def test_ranged_none_still_follows_the_store_setting(corpus, monkeypatch):
    """The default must not change behaviour for existing callers."""
    _, path = corpus
    _force_ranged(monkeypatch)

    store = Store(path, ranged_get=True)
    calls = _counting_download(store, monkeypatch)
    store.read_columns(path, columns=["id"])            # no `ranged=`
    assert len(calls) == 1, "a ranged store stopped using the ranged path"

    off = Store(path, ranged_get=False)
    off_calls = _counting_download(off, monkeypatch)
    off.read_columns(path, columns=["id"])
    assert off_calls == []


def test_ranged_true_turns_the_path_on_for_a_non_ranged_store(corpus, monkeypatch):
    """The override is symmetric: it can turn the path ON as well as off.

    NOTE it is `True`, not the env var, that does this. No production caller
    passes `ranged=True` today -- `NOVA_BF_ID_PASS_RANGED_GET=1` yields
    `ranged=None` (defer to the store), never `True`, so that var cannot
    enable ranged reads on a store configured with `io_ranged_get: false`.
    This case pins the third state so the tri-state stays whole; it is not
    the env var's path. See `test_id_pass_projection.py`.

    `ranged=True` is still subject to the `_RANGED_GET_MIN_BYTES` gate -- it
    opts into the branch, it does not bypass the size floor, which is why this
    needs `_force_ranged` to zero it for a tiny fixture.
    """
    _, path = corpus
    _force_ranged(monkeypatch)
    store = Store(path, ranged_get=False)
    calls = _counting_download(store, monkeypatch)

    store.read_columns(path, columns=["id"], ranged=True)

    assert len(calls) == 1
