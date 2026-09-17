"""Builds OpenSearch query bodies for BM25, vector and hybrid search."""

from dataclasses import dataclass, field

# Field boosts for the lexical leg: headings and titles are strong topical signals
BM25_FIELDS = ["text", "section_heading^2", "module_title^1.5", "subject_name", "note_title"]

SOURCE_FIELDS = [
    "chunk_id",
    "note_id",
    "chunk_index",
    "branch",
    "scheme",
    "semester",
    "subject_code",
    "subject_name",
    "module_number",
    "module_title",
    "note_title",
    "source_uri",
    "source_type",
    "section_heading",
    "text",
    "page_start",
    "page_end",
]


@dataclass
class SearchFilters:
    branch: str | None = None
    scheme: str | None = None
    semester: int | None = None
    subject_code: str | None = None
    module_numbers: list[int] = field(default_factory=list)

    def to_clauses(self) -> list[dict]:
        clauses: list[dict] = []
        if self.branch:
            clauses.append({"term": {"branch": self.branch.lower()}})
        if self.scheme:
            clauses.append({"term": {"scheme": self.scheme}})
        if self.semester:
            clauses.append({"term": {"semester": self.semester}})
        if self.subject_code:
            clauses.append({"term": {"subject_code": self.subject_code.upper()}})
        if self.module_numbers:
            clauses.append({"terms": {"module_number": sorted(set(self.module_numbers))}})
        return clauses

    def cache_key(self) -> str:
        return (
            f"{self.branch or ''}|{self.scheme or ''}|{self.semester or ''}|"
            f"{(self.subject_code or '').upper()}|{','.join(map(str, sorted(self.module_numbers)))}"
        )


def _bm25_query(query: str, filters: SearchFilters) -> dict:
    return {
        "bool": {
            "must": [
                {
                    "multi_match": {
                        "query": query,
                        "fields": BM25_FIELDS,
                        "type": "best_fields",
                        "fuzziness": "AUTO",
                        "prefix_length": 2,
                    }
                }
            ],
            "filter": filters.to_clauses(),
        }
    }


def _knn_query(vector: list[float], filters: SearchFilters, k: int) -> dict:
    knn: dict = {"vector": vector, "k": k}
    clauses = filters.to_clauses()
    if clauses:
        knn["filter"] = {"bool": {"filter": clauses}}
    return {"knn": {"embedding": knn}}


def build_bm25_body(query: str, filters: SearchFilters, size: int) -> dict:
    return {"size": size, "_source": SOURCE_FIELDS, "query": _bm25_query(query, filters)}


def build_vector_body(vector: list[float], filters: SearchFilters, size: int) -> dict:
    return {"size": size, "_source": SOURCE_FIELDS, "query": _knn_query(vector, filters, size)}


def build_hybrid_body(
    query: str, vector: list[float], filters: SearchFilters, size: int, candidates: int = 50
) -> dict:
    """Hybrid query; must be executed with the normalisation search pipeline."""
    k = max(size, candidates)
    return {
        "size": size,
        "_source": SOURCE_FIELDS,
        "query": {
            "hybrid": {"queries": [_bm25_query(query, filters), _knn_query(vector, filters, k)]}
        },
    }
