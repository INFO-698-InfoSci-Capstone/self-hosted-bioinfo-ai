#!/usr/bin/env python3
"""
LangChain-based RAG script using FAISS and Google Gemini (non-Vertex API).
Usage:
  python rag_gemini_langchain.py \
    --index_folder path/to/faiss_index_store \
    --embed_model sentence-transformers/all-MiniLM-L6-v2 \
    --gemini_model text-bison-001 \
    --query "Your question here" \
    --top_k 3
"""
import os
import argparse
from dotenv import load_dotenv

import google.generativeai as genai
from langchain.llms.base import LLM
from pydantic import PrivateAttr
from typing import Optional, List, Mapping, Any
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.schema import LLMResult, Generation

# Module-level default for the API key environment variable
DEFAULT_API_KEY_ENV = "GOOGLE_GEMINI_API_KEY"

class GoogleGemini(LLM):
    """
    Custom LangChain LLM wrapper for Google Gemini via google-generativeai.
    """
    model_name: str
    _client: Any = PrivateAttr()

    def __init__(
        self,
        model_name: str,
        api_key_env: Optional[str] = None,
        **kwargs: Any
    ):
        # Load .env and configure the API key
        load_dotenv()
        env_var = api_key_env if api_key_env else DEFAULT_API_KEY_ENV
        api_key = os.getenv(env_var)
        if not api_key:
            raise RuntimeError(f"Environment variable {env_var} not set")
        genai.configure(api_key=api_key)
        # Initialize base LLM fields
        super().__init__(model_name=model_name, **kwargs)
        # Instantiate and store the Google Gemini client privately
        object.__setattr__(self, '_client', genai.GenerativeModel(model_name=model_name))

    @property
    def _llm_type(self) -> str:
        return "google_gemini"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # Use the private _client to generate text
        response = self._client.generate(
            prompt,
            temperature=0.2,
            max_output_tokens=1024,
            top_k=40,
            top_p=0.8,
        )
        return response.text

    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_name": self.model_name}

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
        generations: List[List[Generation]] = []
        for prompt in prompts:
            text = self._call(prompt, stop)
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)


def main():
    parser = argparse.ArgumentParser(description="LangChain RAG with Google Gemini")
    parser.add_argument(
        "--index_folder", required=True,
        help="Path to FAISS index folder"
    )
    parser.add_argument(
        "--embed_model", default="sentence-transformers/all-MiniLM-L6-v2",
        help="HuggingFace sentence-transformers model for embeddings"
    )
    parser.add_argument(
        "--gemini_model", default="text-bison-001",
        help="Google Gemini model name (e.g., text-bison-001)"
    )
    parser.add_argument(
        "--query", required=True,
        help="Input question to answer"
    )
    parser.add_argument(
        "--top_k", type=int, default=3,
        help="Number of docs to retrieve"
    )
    args = parser.parse_args()

    # Load vectorstore and retriever
    embeddings = HuggingFaceEmbeddings(model_name=args.embed_model)
    db = FAISS.load_local(
        folder_path=args.index_folder,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )
    retriever = db.as_retriever(search_kwargs={"k": args.top_k})

    # Initialize the custom Gemini LLM
    llm = GoogleGemini(model_name=args.gemini_model)

    # Build RetrievalQA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
    )

    # Run query
    answer = qa.run(args.query)
    print("\n=== Answer ===\n")
    print(answer)

if __name__ == "__main__":
    main()
