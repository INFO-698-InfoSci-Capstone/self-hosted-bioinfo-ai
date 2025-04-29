#!/usr/bin/env python3

import argparse

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


class FaissRetriever:
    def __init__(
        self,
        index_folder: str,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        # 1) Set up the HF embedder (must match what you used to build the index)
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

        # 2) Load the persisted FAISS+docstore from the folder
        #    (index.pkl + any metadata are re-hydrated automatically)
        self.vector_store = FAISS.load_local(
            folder_path=index_folder,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
        )

    def retrieve(self, query: str, k: int = 3) -> list[str]:
        """
        Returns the top-k document chunks (as strings) most similar to `query`.
        """
        docs = self.vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]


def main():
    parser = argparse.ArgumentParser(description="FAISS Retriever CLI")
    parser.add_argument(
        "--index_folder",
        type=str,
        default="faiss_index_store",
        help="Path to folder containing your saved FAISS index (index.pkl)",
    )
    parser.add_argument(
        "--embed_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HuggingFace model you used to build the index",
    )
    parser.add_argument(
        "--query", type=str, required=True, help="The user query to retrieve for"
    )
    parser.add_argument(
        "--top_k", type=int, default=3, help="Number of top chunks to return"
    )

    args = parser.parse_args()

    retriever = FaissRetriever(
        index_folder=args.index_folder,
        embedding_model=args.embed_model,
    )

    snippets = retriever.retrieve(args.query, k=args.top_k)

    print("\n--- Retrieved snippets ---\n")
    for i, text in enumerate(snippets, 1):
        # print the first 300 characters of each chunk
        print(f"{i}. {text[:300].replace(chr(10), ' ')}...\n")


if __name__ == "__main__":
    main()

