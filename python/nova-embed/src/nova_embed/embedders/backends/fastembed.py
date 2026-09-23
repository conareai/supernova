"""
fastembed backends: ONNX-runtime models from Qdrant's fastembed library.

One `type: fastembed` name, three kinds, each wrapping the matching fastembed
class:

* dense       — ``TextEmbedding`` (bge-small, MiniLM, ...)
* sparse      — ``SparseTextEmbedding`` (BM25, BM42, SPLADE)
* multivector — ``LateInteractionTextEmbedding`` (ColBERT-style)

fastembed encodes documents and queries differently for some models (ColBERT
pads queries with [MASK] tokens, BM25 gives every query term weight 1),
so every entry takes `query: true` to route through ``query_embed`` when the
input column holds queries rather than corpus passages.

Example config entry:
    embedders:
      - name: bm25
        kind: sparse
        type: fastembed
        model: Qdrant/bm25
        input_column: text
        modality: text
"""

import asyncio
import logging
import threading

from typing import Any

from nova_embed.embedders.base import Embedder, OutputKind
from nova_embed.models import MultiVectorEmbedding, SparseEmbedding
from nova_embed.registry import EMBEDDERS

logger = logging.getLogger(__name__)


class _FastEmbedBase(Embedder):
    """Shared load + encode. Subclasses pick the fastembed class and convert."""

    default_model: str

    def __init__(
        self,
        model: str | None = None,
        batch_size: int = 256,
        cache_dir: str | None = None,
        # embed with query_embed() instead of embed(): set on query-side runs
        query: bool = False,
        # forwarded to the fastembed constructor: threads, providers, cuda, ...
        **fastembed_kwargs: Any,
    ):
        model = model or self.default_model
        logger.info(
            "Loading fastembed %s model %s%s",
            self.output_kind.value, model, " (query mode)" if query else "",
        )
        self._model = self._load(model, cache_dir=cache_dir, **fastembed_kwargs)
        self._model_name = model
        self._batch_size = batch_size
        self._query = query
        # onnxruntime sessions are safe to share, but serializing keeps one
        # batch's threads from contending with another's (same contract as the
        # other local backends)
        self._encode_lock = threading.Lock()

    def _load(self, model: str, **kwargs: Any):
        raise NotImplementedError

    def _convert(self, row) -> Any:
        raise NotImplementedError

    @property
    def model_name(self) -> str:
        return self._model_name

    def _encode(self, texts: list[str]) -> list[Any]:
        fn = self._model.query_embed if self._query else self._model.embed
        with self._encode_lock:
            rows = list(fn(texts, batch_size=self._batch_size))
        return [self._convert(r) for r in rows]

    async def embed(self, texts: list[str]) -> list[Any]:
        return await asyncio.to_thread(self._encode, texts)


@EMBEDDERS.register("fastembed")
class FastEmbedDenseEmbedder(_FastEmbedBase):
    output_kind = OutputKind.DENSE
    default_model = "BAAI/bge-small-en-v1.5"

    def _load(self, model: str, **kwargs: Any):
        from fastembed import TextEmbedding

        return TextEmbedding(model_name=model, **kwargs)

    @property
    def dimensions(self) -> int:
        return self._model.embedding_size

    def _convert(self, row) -> Any:
        # float32 ndarray row, kept as-is (see sentence_transformer's note on
        # Python-float bloat); pyarrow writes it straight to list<float32>
        return row


@EMBEDDERS.register("fastembed")
class FastEmbedSparseEmbedder(_FastEmbedBase):
    output_kind = OutputKind.SPARSE
    default_model = "Qdrant/bm25"

    def _load(self, model: str, **kwargs: Any):
        from fastembed import SparseTextEmbedding

        return SparseTextEmbedding(model_name=model, **kwargs)

    @property
    def max_tokens(self) -> int:
        # BM25 is lexical (word tokens), no hard transformer limit.
        # Return a large sentinel so per-entry max_length governs instead.
        return 100_000

    def _convert(self, row) -> SparseEmbedding:
        return SparseEmbedding(indices=row.indices.tolist(), values=row.values.tolist())


@EMBEDDERS.register("fastembed")
class FastEmbedMultiVectorEmbedder(_FastEmbedBase):
    output_kind = OutputKind.MULTIVECTOR
    default_model = "colbert-ir/colbertv2.0"

    def _load(self, model: str, **kwargs: Any):
        from fastembed import LateInteractionTextEmbedding

        return LateInteractionTextEmbedding(model_name=model, **kwargs)

    @property
    def dimensions(self) -> int:
        return self._model.embedding_size

    def _convert(self, row) -> MultiVectorEmbedding:
        # (num_tokens, dim) ndarray, passed through like bge_m3's colbert head
        return MultiVectorEmbedding(vectors=row)
