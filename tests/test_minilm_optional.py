"""OPTIONAL MiniLM integration tests — skipped unless sentence-transformers is
installed AND RUN_MINILM=1 is set (they download/run the real encoder, so they
are not part of the offline deterministic suite or CI).

    pip install sentence-transformers
    RUN_MINILM=1 pytest tests/test_minilm_optional.py -v
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_MINILM") != "1", reason="set RUN_MINILM=1 to run MiniLM integration tests"
)
pytest.importorskip("sentence_transformers")


def test_minilm_retriever_builds_and_retrieves():
    from retriever import INDEX_DIR, PolicyRetriever

    r = PolicyRetriever(backend="minilm")
    assert r.embedder.name == "minilm" and r.dim == 384
    results, meta = r.retrieve("gambling harmful spend", k=4)
    assert "POL-005" in [c["policy_id"] for c in results], meta["ids"]
    assert (INDEX_DIR / f"minilm_{r.hash}.npz").exists()  # corpus-hash-keyed cache


def test_minilm_cache_reused():
    from retriever import PolicyRetriever

    r1 = PolicyRetriever(backend="minilm")
    r2 = PolicyRetriever(backend="minilm")  # second init loads the .npz cache
    assert (r1.embedder.mat == r2.embedder.mat).all()
