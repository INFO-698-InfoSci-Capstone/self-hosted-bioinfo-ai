from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import streamlit as st
from langchain_core.documents import Document
from langchain_core.runnables import chain

# --- load once and cache ----------------------------------------------------
@st.cache_resource
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_resource
def load_vectordb(index_folder: str) -> FAISS:
    """
    Load the on‑disk FAISS index and return a LangChain retriever.
    """
    vectordb = FAISS.load_local(
        index_folder,
        get_embeddings(),
        allow_dangerous_deserialization=True,  # OK for trusted local disk
    )   # :contentReference[oaicite:0]{index=0}
    return vectordb

class Generator:
    def __init__(self, gen):
        self.gen = gen

    def __iter__(self):
        self.value = yield from self.gen
        return self.value

@chain
def retriever(query: str) -> List[Document]:
    vectorstore = st.session_state.vectordb

    docs, scores = zip(*vectorstore.similarity_search_with_score(query, k=5))
    for doc, score in zip(docs, scores):
        doc.metadata["score"] = score

    return docs