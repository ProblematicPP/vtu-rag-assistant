"""OpenSearch index mapping and the hybrid-search pipeline definition."""


def chunk_index_body(dimensions: int) -> dict:
    return {
        "settings": {
            "index": {
                "knn": True,
                "number_of_shards": 1,
                "number_of_replicas": 0,
            },
            "analysis": {
                "analyzer": {
                    "notes_text": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "english_stop", "english_stemmer"],
                    }
                },
                "filter": {
                    "english_stop": {"type": "stop", "stopwords": "_english_"},
                    "english_stemmer": {"type": "stemmer", "language": "light_english"},
                },
            },
        },
        "mappings": {
            "dynamic": "strict",
            "properties": {
                "chunk_id": {"type": "integer"},
                "note_id": {"type": "integer"},
                "chunk_index": {"type": "integer"},
                "branch": {"type": "keyword"},
                "scheme": {"type": "keyword"},
                "semester": {"type": "integer"},
                "subject_code": {"type": "keyword"},
                "subject_name": {
                    "type": "text",
                    "analyzer": "notes_text",
                    "fields": {"raw": {"type": "keyword"}},
                },
                "module_number": {"type": "integer"},
                "module_title": {"type": "text", "analyzer": "notes_text"},
                "note_title": {"type": "text", "analyzer": "notes_text"},
                "source_uri": {"type": "keyword"},
                "source_type": {"type": "keyword"},
                "section_heading": {"type": "text", "analyzer": "notes_text"},
                "text": {"type": "text", "analyzer": "notes_text"},
                "page_start": {"type": "integer"},
                "page_end": {"type": "integer"},
                "word_count": {"type": "integer"},
                "indexed_at": {"type": "date"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": dimensions,
                    "method": {
                        "name": "hnsw",
                        # Lucene engine supports efficient filtering inside the k-NN query
                        "engine": "lucene",
                        "space_type": "cosinesimil",
                        "parameters": {"m": 16, "ef_construction": 128},
                    },
                },
            },
        },
    }


def hybrid_pipeline_body(bm25_weight: float, vector_weight: float) -> dict:
    """Normalises BM25 and k-NN scores to [0, 1] and blends them with the given weights."""
    return {
        "description": "VTU notes hybrid search: min-max normalised BM25 + vector scores",
        "phase_results_processors": [
            {
                "normalization-processor": {
                    "normalization": {"technique": "min_max"},
                    "combination": {
                        "technique": "arithmetic_mean",
                        "parameters": {"weights": [bm25_weight, vector_weight]},
                    },
                }
            }
        ],
    }
