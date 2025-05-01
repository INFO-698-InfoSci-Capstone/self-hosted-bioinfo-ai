#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
rag_openai_langchain.py

Loads a persisted FAISS store and runs a simple Retrieval-QA loop.

Dependencies:
    pip install langchain-huggingface langchain-community langchain-openai sentence-transformers faiss-cpu
"""

import os
# Disable tokenizers parallelism warnings (huggingface)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Updated embeddings import from langchain-huggingface
from langchain_huggingface import HuggingFaceEmbeddings
# FAISS vectorstore from langchain-community
from langchain_community.vectorstores import FAISS
# Updated OpenAI LLM import
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA


def main():
    # 1. Ensure your OpenAI key is set
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("Please set your OPENAI_API_KEY environment variable")

    # 2. Initialize your embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 3. Hardcode your FAISS store path
    index_folder = "../faiss_index_store"  # ← change to your actual path

    # 4. Load the persisted FAISS store with dangerous deserialization allowed
    vectorstore = FAISS.load_local(
        index_folder,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # 5. Create a Retriever (top-5 results)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # 6. Build a Retrieval-QA chain
    llm = OpenAI(temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
    )

    # 7. Interactive loop
    print(f"Loaded FAISS index from: {index_folder!r}")
    print("Type your question and press enter (or CTRL+C to quit).")
    while True:
        query = input("\n❓ Question: ")
        if not query.strip():
            print("Please enter a non-empty question.")
            continue
        # Use invoke instead of run to avoid deprecation warning
        result = qa_chain.invoke({"query": query})
        # Extract answer from result dict (usually under 'result')
        answer = result.get("result") or next(iter(result.values()))
        print(f"\n💡 Answer: {answer}")


if __name__ == "__main__":
    main()

