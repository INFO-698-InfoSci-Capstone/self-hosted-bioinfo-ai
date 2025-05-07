from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

import streamlit as st
import fitz
from langchain_core.documents import Document
from langchain_core.runnables import chain

from datetime import datetime

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

@st.cache_resource
def load_text_splitter() -> RecursiveCharacterTextSplitter:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, 
                                                   chunk_overlap=200,
                                                   separators=["\n\n", "\n", ".", "!", "?", " ", ""]
                                                   )
    return text_splitter

def make_docs_from_uploads(files) -> list[Document]:
    """Return a list of Documents created from uploaded Streamlit files."""
    docs = []
    for f in files:
        if f.type == "application/pdf":
            pdf_bytes = f.read()              # read file once
            with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
                text = "".join(p.get_text() for p in pdf)
        elif f.type == "text/plain":                                 # fallback: treat as text
            text = f.getvalue().decode("utf-8", errors="ignore")
        else:
            text = f.read().decode("utf-8", errors="ignore")

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "title": f.name.split(".")[0].strip(), 
                    "year": datetime.today().year,
                    "journal": "", "doi": "", "uploaded": True},
            )
        )
        text_splitter = load_text_splitter()
        doc_splits = text_splitter.split_documents(docs)

    return doc_splits