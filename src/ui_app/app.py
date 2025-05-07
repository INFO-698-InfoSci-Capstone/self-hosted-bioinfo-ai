import streamlit as st

from utilities import (initiate_client, 
                        left_column_content, right_column_content)
from llm_resources import load_vectordb

st.set_page_config(page_title="Bioinformatics Lab Assistant",
                   page_icon="🧬",
                   layout="wide")

st.markdown(
    """
    <style>
    
    [data-testid="stMainBlockContainer"] {
            padding-top: 2%;
            padding-bottom: 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------  MAIN APP  ----------------------------------- #
def main() -> None:
    st.title("🧬 UA-LLM for bioinformatics research")

    # Session‑state initialisation
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant",
             "content": "Hello 👋 — how can I help you in the lab today?"}
        ]
    
    if "client" not in st.session_state:
        st.session_state.client = initiate_client()

    if "vectordb" not in st.session_state:
        # st.session_state.retriever = load_retriever("faiss_index_store")  # path on disk
        st.session_state.vectordb = load_vectordb("faiss_index_store")

    if "uploading" not in st.session_state:
        st.session_state.uploading = False

    # --- Layout: chat (left) • uploads (right) ---
    left_col, right_col = st.columns([3, 1], gap="small")

    # ------------------  RIGHT: file‑upload area  ------------------ #
    with right_col:
        right_column_content()

    # ------------------  LEFT: chat area  ------------------ #
    with left_col:
        left_column_content()


if __name__ == "__main__":
    main()
