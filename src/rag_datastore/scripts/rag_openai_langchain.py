#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
rag_openai_langchain.py

Loads a persisted FAISS store and runs a simple Retrieval-QA loop.
"""

import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
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
        answer = qa_chain.run(query)
        print(f"\n💡 Answer: {answer}")

if __name__ == "__main__":
    main()

