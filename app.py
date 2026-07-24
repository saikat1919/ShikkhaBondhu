from dotenv import load_dotenv
load_dotenv()

import os
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

# Streamlit Cloud supplies keys via st.secrets. Locally there is no secrets.toml
# and st.secrets raises rather than coming back empty, so fall through to the
# env vars already loaded from .env above.
try:
    for _key in ("GROQ_API_KEY", "HF_TOKEN"):
        if _key in st.secrets:
            os.environ[_key] = st.secrets[_key]
except StreamlitSecretNotFoundError:
    pass


from langchain_core.messages import HumanMessage, AIMessage

from session_ingestion.session_ingestion import process_uploaded_pdfs
from session_retrieval.session_retrieval import build_session_retriever, delete_session_collection
from session_answer_generation.chat_generation import build_rag_chain, answer_question

st.set_page_config(page_title="InfoBee", page_icon="🐝", layout="centered")

st.title("InfoBee")
st.caption(
    "Upload one or more PDFs and ask questions about them. Answers come only "
    "from the uploaded documents. Everything is processed in memory for this "
    "session only -- nothing is saved once you close or refresh the page."
)

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_filenames" not in st.session_state:
    st.session_state.processed_filenames = []

MAX_TURNS = 10

with st.sidebar:
    st.header("Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF(s)",
        type=["pdf"],
        accept_multiple_files=True,
    )

    process_clicked = st.button("Process documents", type="primary", disabled=not uploaded_files)

    if process_clicked:
        with st.spinner("Reading and indexing your documents..."):
            try:
                documents = process_uploaded_pdfs(uploaded_files)

                if not documents:
                    st.error(
                        "No extractable text was found in the uploaded file(s). "
                        "If this is a scanned/image-only PDF, text extraction "
                        "won't work without OCR."
                    )
                else:
                    delete_session_collection(st.session_state.vector_store)

                    retriever, vector_store = build_session_retriever(documents)
                    st.session_state.vector_store = vector_store
                    st.session_state.rag_chain = build_rag_chain(retriever)
                    st.session_state.chat_history = []
                    st.session_state.processed_filenames = [f.name for f in uploaded_files]
                    st.success(f"Indexed {len(documents)} chunks from {len(uploaded_files)} file(s).")
            except Exception as e:
                import traceback
                st.error(f"Something went wrong while processing your documents: {e}")
                st.code(traceback.format_exc())

    if st.session_state.processed_filenames:
        st.markdown("**Currently loaded:**")
        for name in st.session_state.processed_filenames:
            st.markdown(f"- {name}")

        if st.button("Clear session"):
            delete_session_collection(st.session_state.vector_store)
            st.session_state.rag_chain = None
            st.session_state.vector_store = None
            st.session_state.chat_history = []
            st.session_state.processed_filenames = []
            st.rerun()

if st.session_state.rag_chain is None:
    st.info("Upload one or more PDFs and click **Process documents** to get started.")
else:
    for message in st.session_state.chat_history:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(message.content)

    query = st.chat_input("Ask a question about your document(s)...")

    if query:
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Searching your documents..."):
                try:
                    answer = answer_question(
                        st.session_state.rag_chain,
                        query,
                        st.session_state.chat_history,
                    )
                except Exception as e:
                    answer = f"Sorry, something went wrong while generating an answer: {e}"
                st.markdown(answer)

        st.session_state.chat_history.append(HumanMessage(content=query))
        st.session_state.chat_history.append(AIMessage(content=answer))

        if len(st.session_state.chat_history) > MAX_TURNS * 2:
            st.session_state.chat_history = st.session_state.chat_history[-MAX_TURNS * 2:]