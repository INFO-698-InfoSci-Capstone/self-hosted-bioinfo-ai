import streamlit as st
from openai import OpenAI

# -----------------------------  UTILITIES  ---------------------------------- #
@st.cache_resource
def initiate_client() -> OpenAI | None:
    """Return an authenticated OpenAI client, or None if auth fails."""
    try:
        api_key = st.secrets["openai"]["API_KEY"]
        return OpenAI(api_key=api_key)
    except Exception:
        return None

def stream_llm_reply(client: OpenAI, conversation: list[dict]) -> str:
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
            st.markdown(msg["content"])
    
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
            for token in stream_llm_reply(st.session_state.client, 
                                            st.session_state.messages):
                full_reply += token
                reply_stream.markdown(full_reply + "▌")
            reply_stream.markdown(full_reply)  # remove cursor

        # add assistant reply to history
        st.session_state.messages.append(
            {"role": "assistant", "content": full_reply}
        )