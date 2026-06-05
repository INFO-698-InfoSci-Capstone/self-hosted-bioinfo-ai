from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import fitz
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_weaviate.vectorstores import WeaviateVectorStore
from openai import OpenAI
import weaviate


@dataclass
class QueryConfig:
    top_k: int = 6
    vector_weight: float = 0.6
    bm25_weight: float = 0.4
    model: str = "gpt-4o-mini"
    use_reranker: bool = True


class ChunkingService:
    """Supports sliding-window and lightweight semantic chunking."""

    def __init__(self, embeddings: HuggingFaceEmbeddings):
        self.embeddings = embeddings

    def chunk(self, text: str, strategy: str = "sliding", metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        metadata = metadata or {}
        if strategy == "semantic":
            docs = self._semantic_chunk(text, metadata)
        else:
            docs = self._sliding_chunk(text, metadata)

        for idx, doc in enumerate(docs):
            doc.metadata["chunk_id"] = idx
            doc.metadata.setdefault("source", metadata.get("source", "uploaded_document"))
        return docs

    def _sliding_chunk(self, text: str, metadata: Dict[str, Any]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        return splitter.create_documents([text], metadatas=[metadata])

    def _semantic_chunk(self, text: str, metadata: Dict[str, Any]) -> List[Document]:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if len(sentences) <= 1:
            return [Document(page_content=text, metadata=metadata)]

        embeddings = self.embeddings.embed_documents(sentences)
        chunks: List[str] = []
        current_chunk = [sentences[0]]

        for i in range(1, len(sentences)):
            similarity = self._cosine(embeddings[i - 1], embeddings[i])
            if similarity < 0.72 or len(" ".join(current_chunk)) > 1000:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentences[i]]
            else:
                current_chunk.append(sentences[i])
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return [Document(page_content=chunk, metadata=metadata) for chunk in chunks]

    @staticmethod
    def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
        norm_b = math.sqrt(sum(y * y for y in b)) or 1.0
        return dot / (norm_a * norm_b)


class HybridRetriever:
    def __init__(
        self,
        index_dir: str,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        backend: str = "faiss",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.backend = backend
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.chunker = ChunkingService(self.embeddings)
        self.docs: List[Document] = []
        self.vector_store: Optional[Any] = None
        self.bm25_index: Optional[BM25Okapi] = None
        self._reranker_model = reranker_model
        self._reranker: Optional[Any] = None
        self._load_state()

    def _get_reranker(self) -> Any:
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(self._reranker_model)
        return self._reranker

    def ingest_text(self, text: str, source: str, strategy: str = "sliding") -> int:
        metadata = {"source": source}
        new_docs = self.chunker.chunk(text, strategy=strategy, metadata=metadata)
        self.docs.extend(new_docs)
        self._rebuild_indexes(persist=True)
        return len(new_docs)

    def ingest_file(self, file_path: str, strategy: str = "sliding") -> int:
        text = self._read_text(file_path)
        return self.ingest_text(text, source=os.path.basename(file_path), strategy=strategy)

    def query(self, query: str, config: QueryConfig) -> Dict[str, Any]:
        if not self.docs or self.vector_store is None or self.bm25_index is None:
            raise ValueError("No indexed documents found. Call /ingest first.")

        start = time.perf_counter()

        # Dense retrieval
        vector_hits = self.vector_store.similarity_search_with_score(query, k=config.top_k)
        vector_norm = self._normalize([(doc, 1.0 / (1.0 + score)) for doc, score in vector_hits])

        # Lexical retrieval with actual BM25 scores
        tokenized_query = query.lower().split()
        bm25_raw_scores = self.bm25_index.get_scores(tokenized_query)
        top_bm25_idx = sorted(range(len(bm25_raw_scores)), key=lambda i: bm25_raw_scores[i], reverse=True)[: config.top_k]
        bm25_norm = self._normalize([(self.docs[i], float(bm25_raw_scores[i])) for i in top_bm25_idx])

        # Weighted score fusion
        combined = self._fuse(vector_norm, bm25_norm, config.vector_weight, config.bm25_weight)
        # Take 3× top_k as the reranking candidate pool
        fusion_candidates = sorted(combined.values(), key=lambda x: x[1], reverse=True)[: config.top_k * 3]

        # Cross-encoder rerank
        if config.use_reranker and fusion_candidates:
            reranker = self._get_reranker()
            pairs = [(query, doc.page_content) for doc, _ in fusion_candidates]
            ce_scores = reranker.predict(pairs)
            reranked: List[Tuple[Document, float]] = sorted(
                ((doc, float(s)) for (doc, _), s in zip(fusion_candidates, ce_scores)),
                key=lambda x: x[1],
                reverse=True,
            )[: config.top_k]
        else:
            reranked = [(doc, score) for doc, score in fusion_candidates[: config.top_k]]

        selected_docs = [doc for doc, _ in reranked]
        context = "\n\n".join(d.page_content for d in selected_docs)
        answer, model_name = self._generate_answer(query, selected_docs, config.model)

        confidence = self._confidence_score(reranked)
        hallucination_risk = self._hallucination_risk(answer, context)

        latency_ms = (time.perf_counter() - start) * 1000
        citations = [
            {
                "source": str(d.metadata.get("source", "unknown")),
                "chunk_id": self._safe_int(d.metadata.get("chunk_id", -1)),
                "preview": str(d.page_content[:180]),
            }
            for d in selected_docs
        ]

        retrieval_rows = [
            {
                "source": str(d.metadata.get("source", "unknown")),
                "chunk_id": self._safe_int(d.metadata.get("chunk_id", -1)),
                "score": round(self._safe_float(score), 4),
            }
            for d, score in reranked
        ]

        return {
            "answer": str(answer),
            "model": str(model_name),
            "confidence": round(self._safe_float(confidence), 4),
            "hallucination_risk": str(hallucination_risk),
            "latency_ms": round(self._safe_float(latency_ms), 2),
            "citations": citations,
            "retrieval": retrieval_rows,
        }

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _safe_int(value: Any, default: int = -1) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def reindex(self) -> Dict[str, Any]:
        self._rebuild_indexes(persist=True)
        return {"status": "ok", "documents": len(self.docs)}

    def _rebuild_indexes(self, persist: bool = True) -> None:
        if not self.docs:
            self.vector_store = None
            self.bm25_index = None
            return

        if self.backend == "weaviate":
            self.vector_store = self._build_weaviate_store()
        else:
            self.vector_store = FAISS.from_documents(self.docs, self.embeddings)

        tokenized = [doc.page_content.lower().split() for doc in self.docs]
        self.bm25_index = BM25Okapi(tokenized)

        if persist:
            if self.backend == "faiss" and self.vector_store is not None:
                self.vector_store.save_local(str(self.index_dir))
            self._save_docstore()

    def _load_state(self) -> None:
        docstore_file = self.index_dir / "documents.jsonl"
        faiss_file = self.index_dir / "index.faiss"
        pkl_file = self.index_dir / "index.pkl"

        if docstore_file.exists():
            self.docs = [
                Document(page_content=record["text"], metadata=record["metadata"])
                for record in (json.loads(line) for line in docstore_file.read_text(encoding="utf-8").splitlines() if line)
            ]

        if self.docs:
            if self.backend == "faiss" and faiss_file.exists() and pkl_file.exists():
                self.vector_store = FAISS.load_local(
                    str(self.index_dir),
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
            elif self.backend == "weaviate":
                self.vector_store = self._build_weaviate_store()

            tokenized = [doc.page_content.lower().split() for doc in self.docs]
            self.bm25_index = BM25Okapi(tokenized)

    def _save_docstore(self) -> None:
        docstore_file = self.index_dir / "documents.jsonl"
        with docstore_file.open("w", encoding="utf-8") as f:
            for doc in self.docs:
                f.write(json.dumps({"text": doc.page_content, "metadata": doc.metadata}) + "\n")

    def _build_weaviate_store(self) -> WeaviateVectorStore:
        weaviate_url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
        index_name = os.getenv("WEAVIATE_INDEX", "BioinfoRag")
        client = weaviate.connect_to_local(host=weaviate_url.replace("http://", "").replace("https://", ""))
        return WeaviateVectorStore.from_documents(
            self.docs,
            self.embeddings,
            client=client,
            index_name=index_name,
            text_key="text",
        )

    @staticmethod
    def _read_text(file_path: str) -> str:
        path = Path(file_path)
        if path.suffix.lower() == ".pdf":
            with fitz.open(file_path) as pdf:
                return "\n".join(page.get_text() for page in pdf)
        return path.read_text(encoding="utf-8", errors="ignore")

    @staticmethod
    def _normalize(scored_docs: Iterable[Tuple[Document, float]]) -> Dict[str, Tuple[Document, float]]:
        docs = list(scored_docs)
        if not docs:
            return {}
        values = [score for _, score in docs]
        min_v, max_v = min(values), max(values)
        denom = (max_v - min_v) or 1.0
        normalized: Dict[str, Tuple[Document, float]] = {}
        for doc, score in docs:
            normalized[HybridRetriever._doc_key(doc)] = (doc, (score - min_v) / denom)
        return normalized

    @staticmethod
    def _fuse(
        vector_scores: Dict[str, Tuple[Document, float]],
        bm25_scores: Dict[str, Tuple[Document, float]],
        vector_weight: float,
        bm25_weight: float,
    ) -> Dict[str, Tuple[Document, float]]:
        all_docs = set(vector_scores) | set(bm25_scores)
        fused = {}
        for key in all_docs:
            doc = vector_scores.get(key, bm25_scores.get(key))[0]
            v_score = vector_scores.get(key, (doc, 0.0))[1]
            b_score = bm25_scores.get(key, (doc, 0.0))[1]
            fused[key] = (doc, vector_weight * v_score + bm25_weight * b_score)
        return fused

    @staticmethod
    def _doc_key(doc: Document) -> str:
        source = doc.metadata.get("source", "unknown")
        chunk = doc.metadata.get("chunk_id", -1)
        return f"{source}:{chunk}:{hash(doc.page_content)}"

    @staticmethod
    def _confidence_score(ranked_docs: List[Tuple[Document, float]]) -> float:
        if not ranked_docs:
            return 0.0
        top_scores = [score for _, score in ranked_docs[:3]]
        return sum(top_scores) / len(top_scores)

    @staticmethod
    def _hallucination_risk(answer: str, context: str) -> str:
        answer_terms = set(re.findall(r"[A-Za-z]{4,}", answer.lower()))
        context_terms = set(re.findall(r"[A-Za-z]{4,}", context.lower()))
        if not answer_terms:
            return "high"
        overlap = len(answer_terms & context_terms) / len(answer_terms)
        if overlap >= 0.7:
            return "low"
        if overlap >= 0.45:
            return "medium"
        return "high"

    @staticmethod
    def _build_cited_context(docs: List[Document]) -> str:
        cited_parts = []
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "unknown")
            chunk = doc.metadata.get("chunk_id", "?")
            cited_parts.append(f"[{i}] ({source}#chunk-{chunk}) {doc.page_content}")
        return "\n\n".join(cited_parts)

    def _generate_answer(self, query: str, docs: List[Document], model: str) -> Tuple[str, str]:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return (
                "OpenAI API key not configured. Returning grounded extractive summary only:\n\n"
                + "\n".join(f"- {d.page_content[:220]}..." for d in docs[:3]),
                "extractive-fallback",
            )

        cited_context = self._build_cited_context(docs)
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a bioinformatics RAG assistant. Answer using only the provided context. "
                        "Every factual claim must include citation markers like [1] that map to the context entries."
                    ),
                },
                {"role": "user", "content": f"Question: {query}\n\nContext:\n{cited_context}"},
            ],
        )
        return response.choices[0].message.content or "", model
