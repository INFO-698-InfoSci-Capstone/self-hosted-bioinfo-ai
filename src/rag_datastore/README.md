# RAG Datastore (Production Upgrade)

This project now includes a production-oriented RAG backend for bioinformatics knowledge retrieval with:

- **Chunking strategies**: sliding window + semantic chunking
- **Hybrid retrieval**: BM25 + dense vector search (FAISS or Weaviate backend)
- **Citation grounding**: answer generation constrained to cited chunks
- **Evaluation pipeline**: latency vs. accuracy benchmark grid
- **Safety signals**: confidence score + hallucination-risk heuristic

## API Endpoints

Run the API:

```bash
uvicorn src.rag_datastore.app.api:app --reload --port 8000
```

### `POST /ingest`
Upload `.txt`, `.md`, or `.pdf` files and choose chunking strategy.

Form fields:
- `file`: uploaded file
- `chunking_strategy`: `sliding` or `semantic`

### `POST /query`
Query the knowledge system.

JSON body example:

```json
{
  "question": "How does single-cell RNA-seq support aging clock research?",
  "top_k": 6,
  "vector_weight": 0.6,
  "bm25_weight": 0.4,
  "model": "gpt-4o-mini"
}
```

Response includes:
- grounded answer text
- confidence score
- hallucination risk (`low`/`medium`/`high`)
- citations and retrieval scores
- end-to-end latency in ms

### `POST /reindex`
Rebuilds FAISS/Weaviate and BM25 indexes from persisted document chunks.

## Weaviate Support

Set backend + connection settings before starting API:

```bash
export RAG_BACKEND=weaviate
export WEAVIATE_URL=http://localhost:8080
export WEAVIATE_INDEX=BioinfoRag
```

Default backend is `faiss` if `RAG_BACKEND` is not set.

## Evaluation Pipeline (Latency vs. Accuracy)

Create a JSONL eval file (`question` + `expected_contains` list), then run:

```bash
python src/rag_datastore/scripts/evaluate_rag.py \
  --eval-set path/to/eval_set.jsonl \
  --index-dir src/rag_datastore/faiss_index_store \
  --topk-grid 3,5,8,12
```

Output table reports:
- top-k setting
- accuracy (containment-style)
- average latency
- p95 latency
