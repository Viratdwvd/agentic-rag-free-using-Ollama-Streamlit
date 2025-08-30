
import os
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Config ---
PERSIST_DIR = ".chroma"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = os.environ.get("CHAT_MODEL", "llama3:3b")  # e.g., "llama3:8b" or "qwen2.5:7b"
TOP_K = 4

st.set_page_config(page_title="Local Multi-Agent RAG (Free)", layout="wide")
st.title("🧩 Local Multi‑Agent RAG — FREE Stack (Ollama + Chroma + Streamlit)")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    CHAT_MODEL = st.text_input("Chat Model", CHAT_MODEL)
    TOP_K = st.slider("Top‑K Chunks", 1, 12, TOP_K)

# Load Vector DB
try:
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
except Exception as e:
    st.error(f"Could not load Chroma store. Did you run `python ingest.py`?\n{e}")
    st.stop()

# Simple retrieval‑augmented chat
user_q = st.text_input("Ask a question about your docs:")
if st.button("Ask") or user_q:
    retr = vectordb.similarity_search(user_q, k=TOP_K)
    context = "\n\n".join([f"Source {i+1}:\n" + doc.page_content for i, doc in enumerate(retr)])
    prompt = f"""You are a helpful RAG assistant. Use the context to answer the user's question.
If unsure, say you don't know. Be concise.

Context:
{context}

Question: {user_q}
Answer:
"""
    llm = Ollama(model="phi3:3.8b")
    answer = llm.invoke(prompt)
    st.subheader("Answer")
    st.write(answer)

    with st.expander("Show retrieved chunks"):
        for i, doc in enumerate(retr, 1):
            st.markdown(f"**Chunk {i} — {doc.metadata.get('source','unknown')}**")
            st.write(doc.page_content)
