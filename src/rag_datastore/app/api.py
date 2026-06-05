from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .rag_service import HybridRetriever, QueryConfig

INDEX_DIR = Path(__file__).resolve().parents[1] / "faiss_index_store"
retriever = HybridRetriever(
    index_dir=str(INDEX_DIR),
    backend=os.getenv("RAG_BACKEND", "faiss"),
    reranker_model=os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
)

app = FastAPI(title="Bioinformatics RAG API", version="1.0.0")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: int = Field(default=6, ge=1, le=20)
    vector_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    bm25_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    model: str = Field(default="gpt-4o-mini")
    use_reranker: bool = Field(default=True)


class ReindexResponse(BaseModel):
    status: str
    documents: int


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "indexed_documents": len(retriever.docs)}


@app.post("/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    chunking_strategy: Literal["sliding", "semantic"] = Form(default="sliding"),
):
    suffix = Path(file.filename or "upload.txt").suffix or ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        chunk_count = retriever.ingest_file(tmp_path, strategy=chunking_strategy)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return {
        "status": "ok",
        "file": file.filename,
        "chunking_strategy": chunking_strategy,
        "chunks_added": chunk_count,
        "total_documents": len(retriever.docs),
    }


@app.post("/query")
def query_knowledge(request: QueryRequest):
    if abs((request.vector_weight + request.bm25_weight) - 1.0) > 1e-6:
        raise HTTPException(status_code=400, detail="vector_weight + bm25_weight must equal 1.0")

    try:
        config = QueryConfig(
            top_k=request.top_k,
            vector_weight=request.vector_weight,
            bm25_weight=request.bm25_weight,
            model=request.model,
            use_reranker=request.use_reranker,
        )
        return retriever.query(request.question, config=config)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/reindex", response_model=ReindexResponse)
def reindex() -> ReindexResponse:
    result = retriever.reindex()
    return ReindexResponse(**result)
