"""Retriever tests: corpus parsing, per-topic retrieval quality (each policy
section reachable by its natural query), min-score cut-off, cosine setup."""

import pytest

from retriever import corpus_hash, load_chunks


def test_corpus_parses_17_sections_with_metadata():
    chunks = load_chunks()
    assert len(chunks) == 17
    ids = [c["policy_id"] for c in chunks]
    assert len(set(ids)) == 17
    assert {c["doc_id"] for c in chunks} == {"lending_policy", "data_quality_policy"}
    for c in chunks:
        assert c["version"] and c["effective"] and c["title"] and c["body"]


def test_corpus_hash_stable_and_content_sensitive():
    chunks = load_chunks()
    assert corpus_hash(chunks) == corpus_hash(load_chunks())
    tampered = [dict(c) for c in chunks]
    tampered[0]["text"] += " x"
    assert corpus_hash(tampered) != corpus_hash(chunks)


@pytest.mark.parametrize(
    "query,expected_id",
    [
        ("gambling harmful spend", "POL-005"),
        ("insufficient data thin file minimum history transactions", "POL-007"),
        ("refunds reversals netted against original spending never income", "DQ-003"),
        ("cash withdrawals unverifiable spending", "DQ-005"),
        ("unclassified unknown transactions manual review", "DQ-001"),
        ("internal transfers own accounts excluded money moved", "DQ-002"),
        ("income volatility conservative estimate verification", "POL-004"),
        ("overdraft returned direct debit financial distress", "POL-006"),
        ("debt-to-income limit", "POL-003"),
        ("benefits vulnerability manual review forbearance", "POL-008"),
        ("duplicate postings detected removed", "DQ-004"),
        ("maximum affordable amount reduce", "POL-009"),
        ("classification plausibility transfer legs balance essential spending", "DQ-007"),
    ],
)
def test_topic_queries_hit_their_section(retriever, query, expected_id):
    results, meta = retriever.retrieve(query, k=4)
    assert expected_id in [c["policy_id"] for c in results], meta["ids"]


def test_results_ranked_and_thresholded(retriever):
    results, meta = retriever.retrieve("gambling", k=17)
    scores = [c["score"] for c in results]
    assert scores == sorted(scores, reverse=True)
    assert all(s >= retriever.min_score for s in scores)
    assert meta["latency_ms"] >= 0 and meta["k"] == 17


def test_nonsense_query_returns_empty(retriever):
    results, meta = retriever.retrieve("zzz qqq xyzzy plugh", k=4)
    assert results == [] and meta["empty"] is True


def test_manifest_written(retriever):
    import json

    from retriever import INDEX_DIR

    man = json.loads((INDEX_DIR / "manifest.json").read_text())
    assert man["n_chunks"] == 17
    assert man["index_type"].startswith("IndexFlatIP")
    assert man["corpus_hash"] == retriever.hash
