from vtu_rag.services.search.query_builder import (
    SearchFilters,
    build_bm25_body,
    build_hybrid_body,
    build_vector_body,
)


def test_empty_filters_produce_no_clauses():
    assert SearchFilters().to_clauses() == []


def test_filters_are_normalised():
    clauses = SearchFilters(
        branch="CSE", scheme="2022", semester=3, subject_code="bcs303", module_numbers=[2, 1, 2]
    ).to_clauses()
    assert {"term": {"branch": "cse"}} in clauses
    assert {"term": {"subject_code": "BCS303"}} in clauses
    assert {"terms": {"module_number": [1, 2]}} in clauses
    assert {"term": {"semester": 3}} in clauses


def test_bm25_body_applies_filters():
    body = build_bm25_body("paging", SearchFilters(subject_code="BCS303"), size=7)
    assert body["size"] == 7
    bool_q = body["query"]["bool"]
    assert bool_q["must"][0]["multi_match"]["query"] == "paging"
    assert bool_q["filter"] == [{"term": {"subject_code": "BCS303"}}]


def test_vector_body_without_filters_has_no_filter_key():
    body = build_vector_body([0.1, 0.2], SearchFilters(), size=3)
    knn = body["query"]["knn"]["embedding"]
    assert knn == {"vector": [0.1, 0.2], "k": 3}


def test_hybrid_body_filters_both_legs():
    filters = SearchFilters(semester=4)
    body = build_hybrid_body("dynamic programming", [0.5], filters, size=5, candidates=40)
    lexical, semantic = body["query"]["hybrid"]["queries"]
    assert lexical["bool"]["filter"] == [{"term": {"semester": 4}}]
    knn = semantic["knn"]["embedding"]
    assert knn["k"] == 40
    assert knn["filter"] == {"bool": {"filter": [{"term": {"semester": 4}}]}}


def test_cache_key_is_order_insensitive_for_modules():
    a = SearchFilters(subject_code="bcs303", module_numbers=[2, 1])
    b = SearchFilters(subject_code="BCS303", module_numbers=[1, 2])
    assert a.cache_key() == b.cache_key()
