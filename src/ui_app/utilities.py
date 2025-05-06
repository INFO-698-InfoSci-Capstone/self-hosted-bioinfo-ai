import streamlit as st
from openai import OpenAI

from langchain_community.vectorstores import FAISS
from llm_resources import retriever, Generator

# -----------------------------  UTILITIES  ---------------------------------- #
@st.cache_resource
def initiate_client() -> OpenAI | None:
    """Return an authenticated OpenAI client, or None if auth fails."""
    try:
        api_key = st.secrets["openai"]["API_KEY"]
        return OpenAI(api_key=api_key)
    except Exception:
        return None

# --------------------------- OLD FUNCTION THAT DOES NOT WORK ON CONTEXT ------ #
def stream_llm_reply(client: OpenAI, conversation: list[dict]):
    """
    Yield tokens from GPT‑3.5‑turbo in streaming mode and return the full reply.
    """
    reply_text = ""
    if client is None:
        yield "**⚠️ OpenAI client not initialised – check your API key.**"
        return

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=conversation,
        stream=True,
    )

    for chunk in response:
        delta = chunk.choices[0].delta
        if delta.content:
            reply_text += delta.content
            yield delta.content
    return reply_text

def answer_with_context(client: OpenAI, vectordb: FAISS, question: str):
    """
    Pull relevant docs, build a context‑augmented prompt, stream the answer.
    Returns (assistant_reply, reference_block).
    """
    # 1. Retrieve docs
    # docs = retriever.get_relevant_documents(question)            # :contentReference[oaicite:1]{index=1}
    docs = retriever.invoke(question)            # :contentReference[oaicite:1]{index=1}
    print("--------------------------------------------------------------------------------------------")
    for doc in docs:
        print(doc.metadata['score'], ": ", doc.metadata['title'])
    context = "\n\n---\n\n".join(d.page_content for d in docs)

    # 2. Build messages
    messages = [
        {"role": "system",
         "content": "You are a helpful bioinformatics assistant. cite each fact with [n] "
                    "where n is the nth reference document you use."},
        {"role": "system", "content": f"Context:\n{context}"},
        {"role": "user",   "content": question},
    ]

    # 3. Stream
    reply = ""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo", messages=messages, stream=True
    )
    for chunk in response:
        if chunk.choices[0].delta.content:
            reply += chunk.choices[0].delta.content
            yield chunk.choices[0].delta.content  # token stream

    # 4. Very small citation block (just file names here)
    cites = []
    for i, d in enumerate(docs, 1):
        m = d.metadata
        if m.get('score', '') <=1:
            cites.append(
                f"[{i}] {m.get('title','')} ({m.get('year','')} {m.get('journal','')}). doi:{m.get('doi','')}"
            )
    refs =  "\n".join(cites) if cites != [] else ""
    
    return reply, refs


def right_column_content():
    st.subheader("Context files (optional)")
    uploaded_files = st.file_uploader(
        "Upload notes, papers, or data files you might want to reference later.",
        accept_multiple_files=True,
        type=None,
        label_visibility="collapsed",
    )
    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) stored in session "
                    "— we'll wire them into the LLM in the next step.")
        
def left_column_content():
    chat_msg_placeholder = st.container(height=500, border=False)
    input_msg_placeholder = st.container() 

    for msg in st.session_state.messages:
        with chat_msg_placeholder.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)
    
    user_prompt = input_msg_placeholder.chat_input("Type your question here…")

    if user_prompt:
        # save + echo user message
        st.session_state.messages.append(
            {"role": "user", "content": user_prompt}
        )
        with chat_msg_placeholder.chat_message("user"):
            st.markdown(user_prompt)

        # stream assistant reply
        with chat_msg_placeholder.chat_message("assistant"):
            reply_stream = st.empty()
            full_reply = ""
            # response_generator = Generator(stream_llm_reply(st.session_state.client, 
            #                                 st.session_state.messages))
            response_generator = Generator(answer_with_context(st.session_state.client,
                                                                st.session_state.vectordb,
                                                                user_prompt))
            for token in response_generator:
                full_reply += token
                reply_stream.markdown(full_reply + "▌")
            reply_stream.markdown(full_reply, unsafe_allow_html=True)  # remove cursor
            
            # simple foot‑note style references
            _, refs = response_generator.value
            if refs != "":
                full_reply += "\n\n"
                full_reply += f"<sub>{refs}<\sub>"
                st.caption(f"{refs}")