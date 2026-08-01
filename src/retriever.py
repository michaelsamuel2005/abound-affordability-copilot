"""RAG retriever over the policy corpus.

    policy markdown  ->  section-aligned chunks (one policy rule per chunk,
    heading + body, stable [POL-xxx]/[DQ-xxx] IDs)  ->  dense vectors  ->
    FAISS IndexFlatIP over L2-normalised vectors (= exact cosine similarity)
    ->  top-k retrieval with a minimum-score cut-off.

Chunking is **section-aligned by design**: each policy rule is an atomic unit
(40–130 words), so splitting mid-rule or overlapping windows would only smear
meaning across chunks. Fixed-size chunking with overlap is the right tool for
long unstructured prose — not for a rulebook.

Embedding backends (pluggable, same FAISS path):
  * tfidf  (default)          — scikit-learn TF-IDF vectors; no heavy deps, fully
                                offline and deterministic → used by CI and evals;
  * minilm (EMBEDDINGS=minilm) — sentence-transformers `all-MiniLM-L6-v2`
                                (384-dim), the same encoder as the PubMedQA
                                project. Embeddings are cached to disk keyed by a
                                corpus hash, so the index rebuilds only when the
                                policy text actually changes.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import faiss
import numpy as np

from config import retrieval_config

ROOT = Path(__file__).resolve().parents[1]
POLICY_DIR = ROOT / "policy"
INDEX_DIR = ROOT / "index"

_SECTION_RE = re.compile(r"### \[((?:POL|DQ)-\d+)\]\s*(.+?)\n(.*?)(?=\n### |\Z)", re.S)
_META_RE = {
    "doc_id": re.compile(r"^doc_id:\s*(\S+)", re.M),
    "version": re.compile(r"^version:\s*(\S+)", re.M),
    "effective": re.compile(r"^effective:\s*(\S+)", re.M),
}


def load_chunks(policy_dir: Path = POLICY_DIR) -> list[dict]:
    """Parse every policy markdown file into section chunks with metadata."""
    chunks: list[dict] = []
    for path in sorted(policy_dir.glob("*.md")):
        text = path.read_text()
        meta = {
            k: (rx.search(text).group(1) if rx.search(text) else "") for k, rx in _META_RE.items()
        }
        title_m = re.search(r"^# (.+)$", text, re.M)
        for m in _SECTION_RE.finditer(text):
            pid, title, body = m.group(1), m.group(2).strip(), " ".join(m.group(3).split())
            chunks.append(
                {
                    "policy_id": pid,
                    "doc_id": meta["doc_id"] or path.stem,
                    "doc_title": title_m.group(1) if title_m else path.stem,
                    "version": meta["version"],
                    "effective": meta["effective"],
                    "title": title,
                    "body": body,
                    "text": f"[{pid}] {title}. {body}",
                }
            )
    ids = [c["policy_id"] for c in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate policy IDs across corpus")
    return chunks


def corpus_hash(chunks: list[dict]) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c["text"].encode())
    return h.hexdigest()[:16]


class TfidfEmbedder:
    """Deterministic, dependency-light default. Fit on the corpus at startup
    (16 chunks — milliseconds), so no persistence needed."""

    name = "tfidf"

    def __init__(self, corpus: list[str]):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.mat = self.vec.fit_transform(corpus).toarray().astype("float32")
        self.dim = self.mat.shape[1]

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.vec.transform(texts).toarray().astype("float32")


class MiniLMEmbedder:
    """sentence-transformers all-MiniLM-L6-v2 (384-dim). Corpus embeddings are
    cached under index/ keyed by corpus hash."""

    name = "minilm"
    model_name = "all-MiniLM-L6-v2"

    def __init__(self, corpus: list[str], cache_key: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(self.model_name)
        cache = INDEX_DIR / f"minilm_{cache_key}.npz" if cache_key else None
        if cache and cache.exists():
            self.mat = np.load(cache)["mat"].astype("float32")
        else:
            self.mat = self._enc(corpus)
            if cache:
                INDEX_DIR.mkdir(exist_ok=True)
                np.savez_compressed(cache, mat=self.mat)
        self.dim = self.mat.shape[1]

    def _enc(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(list(texts), batch_size=32, show_progress_bar=False), dtype="float32"
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._enc(texts)


class PolicyRetriever:
    def __init__(self, chunks: list[dict] | None = None, backend: str | None = None):
        import os

        cfg = retrieval_config()
        self.cfg = cfg
        self.chunks = chunks if chunks is not None else load_chunks()
        if not self.chunks:
            raise ValueError("policy corpus is empty")
        self.by_id = {c["policy_id"]: c for c in self.chunks}
        self.hash = corpus_hash(self.chunks)
        corpus = [c["text"] for c in self.chunks]
        backend = (backend or os.getenv("EMBEDDINGS", "tfidf")).lower()
        t0 = time.perf_counter()
        if backend == "minilm":
            self.embedder = MiniLMEmbedder(corpus, cache_key=self.hash)
            self.min_score = cfg.min_score_minilm
        else:
            self.embedder = TfidfEmbedder(corpus)
            self.min_score = cfg.min_score_tfidf
        mat = self.embedder.mat.copy()
        faiss.normalize_L2(mat)
        self.dim = int(mat.shape[1])
        self.index = faiss.IndexFlatIP(self.dim)  # exact inner product == cosine (normalised)
        self.index.add(mat)
        self.build_ms = (time.perf_counter() - t0) * 1000
        self._write_manifest()

    def _write_manifest(self) -> None:
        INDEX_DIR.mkdir(exist_ok=True)
        (INDEX_DIR / "manifest.json").write_text(
            json.dumps(
                {
                    "embedder": self.embedder.name,
                    "dim": self.dim,
                    "n_chunks": len(self.chunks),
                    "corpus_hash": self.hash,
                    "index_type": "IndexFlatIP(normalised)=cosine",
                    "policy_versions": sorted(
                        {f"{c['doc_id']}@{c['version']}" for c in self.chunks}
                    ),
                },
                indent=2,
            )
        )

    def retrieve(self, query: str, k: int | None = None) -> tuple[list[dict], dict]:
        """Top-k chunks above the minimum score. Returns (results, meta)."""
        k = k or self.cfg.top_k
        t0 = time.perf_counter()
        q = self.embedder.embed([query]).copy()
        faiss.normalize_L2(q)
        scores, idx = self.index.search(q, min(k, len(self.chunks)))
        out = []
        for s, i in zip(scores[0], idx[0], strict=True):
            if i < 0 or float(s) < self.min_score:
                continue
            c = dict(self.chunks[i])
            c["score"] = round(float(s), 4)
            out.append(c)
        meta = {
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            "k": k,
            "min_score": self.min_score,
            "returned": len(out),
            "empty": not out,
            "ids": [c["policy_id"] for c in out],
            "scores": [c["score"] for c in out],
        }
        return out, meta
