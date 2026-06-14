# Bioinformatics RAG System

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

A production-grade Retrieval-Augmented Generation (RAG) API for bioinformatics literature. Documents are chunked, embedded, and stored in a hybrid index. At query time, dense vector search and BM25 lexical search are fused by weighted scoring, then a cross-encoder reranks the candidate pool before the top passages are sent to an LLM with numbered citation metadata.

---

## Architecture

```
Document (PDF / TXT)
        │
        ▼
  Chunking Service
  ├── sliding-window  (RecursiveCharacterTextSplitter, 900 tok / 200 overlap)
  └── semantic        (cosine-similarity boundary detection, threshold 0.72)
        │
        ▼
  ┌─────────────────────────────┐
  │  sentence-transformers      │  all-MiniLM-L6-v2 (configurable)
  │  HuggingFaceEmbeddings      │
  └──────────┬──────────────────┘
             │ embed
     ┌───────┴───────┐
     ▼               ▼
  FAISS index     BM25Okapi index
  (dense)         (lexical, rank-bm25)
     │               │
     └──────┬────────┘
            ▼
   Weighted score fusion
   (min-max normalise each leg → α·dense + β·BM25)
            │
            ▼
   Cross-encoder rerank          ← NEW
   (top_k × 3 pool → CrossEncoder.predict → re-sort → top_k)
            │
            ▼
   LLM prompt with [1]…[n] citation markers
   (OpenAI gpt-4o-mini by default; extractive fallback when no key)
```

---

## Features

| Feature | Detail |
|---|---|
| Hybrid retrieval | FAISS dense + BM25 lexical, independently normalised |
| Actual BM25 scores | `BM25Okapi.get_scores()` — not positional-rank proxies |
| Weighted score fusion | Configurable `vector_weight` / `bm25_weight` (must sum to 1.0) |
| Cross-encoder reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` re-scores a 3× candidate pool; toggle per request |
| Two chunking strategies | `sliding` (default) or `semantic` |
| PDF + plain text ingestion | PyMuPDF for PDF extraction |
| Persistent FAISS index | Survives restarts; docstore in `.jsonl` for BM25 rebuild |
| Weaviate backend | Optional; switch via `RAG_BACKEND=weaviate` |
| Citation metadata | Every retrieved chunk surfaced with source, chunk ID, and 180-char preview |
| Hallucination risk signal | Term-overlap heuristic (`low` / `medium` / `high`) |
| Confidence score | Mean cross-encoder score of top-3 results |
| OpenAI fallback | Returns extractive snippets when `OPENAI_API_KEY` is absent |

---

## Setup

### Requirements

- Python 3.10+
- `pip install -r src/rag_datastore/requirements.txt`

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required for LLM answer generation |
| `RAG_BACKEND` | `faiss` | Vector backend: `faiss` or `weaviate` |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | HuggingFace cross-encoder model ID |
| `WEAVIATE_URL` | `http://localhost:8080` | Weaviate endpoint (only when `RAG_BACKEND=weaviate`) |
| `WEAVIATE_INDEX` | `BioinfoRag` | Weaviate class name |

### Start the API

```bash
cd src/rag_datastore
uvicorn app.api:app --reload --port 8000
```

Interactive docs available at `http://localhost:8000/docs`.

---

## API Endpoints

### `GET /health`

Returns indexed document count.

```bash
curl http://localhost:8000/health
# {"status":"ok","indexed_documents":142}
```

---

### `POST /ingest`

Upload a PDF or plain-text file to be chunked and indexed.

| Form field | Type | Default | Description |
|---|---|---|---|
| `file` | file | required | `.pdf` or `.txt` |
| `chunking_strategy` | string | `sliding` | `sliding` or `semantic` |

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@paper.pdf" \
  -F "chunking_strategy=sliding"
```

```json
{
  "status": "ok",
  "file": "paper.pdf",
  "chunking_strategy": "sliding",
  "chunks_added": 38,
  "total_documents": 180
}
```

---

### `POST /query`

Query the indexed corpus. Returns an LLM answer with grounded citations.

| Field | Type | Default | Description |
|---|---|---|---|
| `question` | string | required | Natural-language query |
| `top_k` | int | `6` | Final number of passages passed to the LLM |
| `vector_weight` | float | `0.6` | Dense retrieval weight (must sum to 1.0 with `bm25_weight`) |
| `bm25_weight` | float | `0.4` | BM25 lexical weight |
| `model` | string | `gpt-4o-mini` | OpenAI model for answer generation |
| `use_reranker` | bool | `true` | Enable cross-encoder reranking |

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the key steps in RNA-seq differential expression analysis?",
    "top_k": 6,
    "vector_weight": 0.6,
    "bm25_weight": 0.4,
    "use_reranker": true
  }'
```

```json
{
  "answer": "RNA-seq differential expression analysis typically involves ... [1] ... [3]",
  "model": "gpt-4o-mini",
  "confidence": 0.8241,
  "hallucination_risk": "low",
  "latency_ms": 1842.5,
  "citations": [
    {
      "source": "paper.pdf",
      "chunk_id": 12,
      "preview": "Differential expression is assessed using DESeq2 or edgeR after..."
    }
  ],
  "retrieval": [
    { "source": "paper.pdf", "chunk_id": 12, "score": 4.7821 },
    { "source": "paper.pdf", "chunk_id": 7,  "score": 3.1204 }
  ]
}
```

**Score in `retrieval`** is the raw cross-encoder logit when `use_reranker=true`, or the normalised fusion score when `false`.

---

### `POST /reindex`

Rebuild both FAISS and BM25 indexes from the persisted docstore (useful after a crash or manual docstore edit).

```bash
curl -X POST http://localhost:8000/reindex
# {"status":"ok","documents":180}
```

---

## Python client example

```python
import requests

BASE = "http://localhost:8000"

# Ingest
with open("paper.pdf", "rb") as f:
    resp = requests.post(f"{BASE}/ingest", files={"file": f}, data={"chunking_strategy": "semantic"})
print(resp.json())

# Query with reranking disabled (faster, slightly less accurate)
resp = requests.post(f"{BASE}/query", json={
    "question": "How does CRISPR-Cas9 achieve sequence specificity?",
    "top_k": 4,
    "use_reranker": False,
})
result = resp.json()
print(result["answer"])
for c in result["citations"]:
    print(f"  [{c['source']} chunk {c['chunk_id']}] {c['preview']}")
```

---

## Retrieval pipeline details

### Dense leg

`HuggingFaceEmbeddings` (`sentence-transformers/all-MiniLM-L6-v2` by default) embeds the query and retrieves the top-k nearest neighbours from FAISS using L2 distance. Scores are converted to similarity via `1 / (1 + L2_distance)` before normalisation.

### Lexical leg

`rank_bm25.BM25Okapi` is built from the same document corpus (whitespace-tokenised, lowercased). `get_scores()` returns actual BM25 term-frequency–inverse-document-frequency scores, not a rank proxy.

### Weighted fusion

Both score vectors are min-max normalised to [0, 1] independently, then combined:

```
fused_score = vector_weight × dense_score + bm25_weight × bm25_score
```

Documents appearing in only one leg receive 0.0 for the absent leg.

### Cross-encoder reranking

The top `top_k × 3` fused candidates are passed as `(query, passage)` pairs to a cross-encoder (`CrossEncoder.predict`). The cross-encoder is loaded lazily on the first query and cached for subsequent requests. Results are re-sorted by cross-encoder logit and trimmed to `top_k`.

Set `use_reranker=false` in the request body to skip this step and return fusion-ranked results directly (lower latency).

---

## File organisation

```
src/
├── rag_datastore/
│   ├── app/
│   │   ├── api.py             # FastAPI routes
│   │   └── rag_service.py     # HybridRetriever, ChunkingService, QueryConfig
│   ├── faiss_index_store/     # persisted FAISS index + documents.jsonl
│   ├── scripts/
│   │   ├── retriever.py       # standalone FAISS retriever CLI
│   │   ├── evaluate_rag.py    # evaluation harness
│   │   ├── rag_gemini_langchain.py
│   │   └── rag_openai_langchain.py
│   └── requirements.txt
└── ui_app/
    ├── app.py
    ├── llm_resources.py
    └── utilities.py

analysis/
└── logs/
    └── log.md
```
