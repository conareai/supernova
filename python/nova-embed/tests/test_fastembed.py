"""fastembed backends, against a stub `fastembed` module (no model downloads)."""

from __future__ import annotations

import asyncio
import sys
import types

import numpy as np
import pytest

pytest.importorskip("obstore")  # nova_embed.embedders package pulls in storage

from nova_embed.config import EmbedderEntry
from nova_embed.embedders.engine import build_engine
from nova_embed.models import MultiVectorEmbedding, SparseEmbedding


class _StubModel:
    """Records construction kwargs; embed/query_embed tag outputs by mode."""

    instances: list["_StubModel"] = []

    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        self.embedding_size = 4
        _StubModel.instances.append(self)

    def embed(self, texts, batch_size):
        return (self._row(t, 1.0) for t in texts)

    def query_embed(self, texts, batch_size):
        return (self._row(t, 2.0) for t in texts)


class _Dense(_StubModel):
    def _row(self, text, tag):
        return np.full(4, tag * len(text), dtype=np.float32)


class _Sparse(_StubModel):
    def _row(self, text, tag):
        return types.SimpleNamespace(
            indices=np.array([len(text)]), values=np.array([tag], dtype=np.float32)
        )


class _Late(_StubModel):
    def _row(self, text, tag):
        return np.full((len(text), 4), tag, dtype=np.float32)


@pytest.fixture(autouse=True)
def stub_fastembed(monkeypatch):
    mod = types.ModuleType("fastembed")
    mod.TextEmbedding = _Dense
    mod.SparseTextEmbedding = _Sparse
    mod.LateInteractionTextEmbedding = _Late
    monkeypatch.setitem(sys.modules, "fastembed", mod)
    _StubModel.instances.clear()


def entry(kind, **overrides) -> EmbedderEntry:
    data = {
        "name": f"fe_{kind}",
        "kind": kind,
        "type": "fastembed",
        "input_column": "text",
        "modality": "text",
    }
    data.update(overrides)
    return EmbedderEntry.model_validate(data)


def run(engine, texts):
    return asyncio.run(engine.embed([{"text": t} for t in texts]))


def test_dense():
    engine = build_engine([entry("dense")])
    (vec,) = run(engine, ["abc"])["fe_dense"]
    assert list(vec) == [3.0] * 4
    (spec,) = engine.output_specs
    assert spec.model_name == "BAAI/bge-small-en-v1.5"
    assert spec.dimensions == 4


def test_sparse_keeps_bm25_default():
    engine = build_engine([entry("sparse")])
    (vec,) = run(engine, ["abcd"])["fe_sparse"]
    assert vec == SparseEmbedding(indices=[4], values=[1.0])
    assert _StubModel.instances[0].model_name == "Qdrant/bm25"


def test_multivector():
    engine = build_engine([entry("multivector")])
    (vec,) = run(engine, ["ab"])["fe_multivector"]
    assert isinstance(vec, MultiVectorEmbedding)
    assert np.asarray(vec.vectors).shape == (2, 4)
    assert engine.output_specs[0].model_name == "colbert-ir/colbertv2.0"


@pytest.mark.parametrize("kind", ["dense", "sparse", "multivector"])
def test_query_mode_uses_query_embed(kind):
    doc = run(build_engine([entry(kind)]), ["abc"])[f"fe_{kind}"][0]
    qry = run(build_engine([entry(kind, query=True)]), ["abc"])[f"fe_{kind}"][0]
    as_array = lambda v: np.asarray(
        v.values if isinstance(v, SparseEmbedding)
        else v.vectors if isinstance(v, MultiVectorEmbedding) else v
    )
    assert not np.array_equal(as_array(doc), as_array(qry))


def test_extra_kwargs_reach_fastembed():
    build_engine([entry("dense", model="m", threads=3, cache_dir="/c")])
    (inst,) = _StubModel.instances
    assert inst.model_name == "m"
    assert inst.kwargs == {"threads": 3, "cache_dir": "/c"}


def test_image_modality_rejected_before_load():
    with pytest.raises(ValueError, match="modality"):
        build_engine([entry("dense", modality="image", input_column="img")])
    assert _StubModel.instances == []
