
# import os
# import streamlit as st
# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.llms import Ollama
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# # --- Config ---
# PERSIST_DIR = ".chroma"
# EMBED_MODEL = "nomic-embed-text"
# CHAT_MODEL = os.environ.get("CHAT_MODEL", "llama3:3b")  # e.g., "llama3:8b" or "qwen2.5:7b"
# TOP_K = 4

# st.set_page_config(page_title="Local Multi-Agent RAG (Free)", layout="wide")
# st.title("🧩 Local Multi‑Agent RAG — FREE Stack (Ollama + Chroma + Streamlit)")

# # Sidebar controls
# with st.sidebar:
#     st.header("Settings")
#     CHAT_MODEL = st.text_input("Chat Model", CHAT_MODEL)
#     TOP_K = st.slider("Top‑K Chunks", 1, 12, TOP_K)

# # Load Vector DB
# try:
#     embeddings = OllamaEmbeddings(model=EMBED_MODEL)
#     vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
# except Exception as e:
#     st.error(f"Could not load Chroma store. Did you run `python ingest.py`?\n{e}")
#     st.stop()

# # Simple retrieval‑augmented chat
# user_q = st.text_input("Ask a question about your docs:")
# if st.button("Ask") or user_q:
#     retr = vectordb.similarity_search(user_q, k=TOP_K)
#     context = "\n\n".join([f"Source {i+1}:\n" + doc.page_content for i, doc in enumerate(retr)])
#     prompt = f"""You are a helpful RAG assistant. Use the context to answer the user's question.
# If unsure, say you don't know. Be concise.

# Context:
# {context}

# Question: {user_q}
# Answer:
# """
#     llm = Ollama(model="phi3:3.8b")
#     answer = llm.invoke(prompt)
#     st.subheader("Answer")
#     st.write(answer)

#     with st.expander("Show retrieved chunks"):
#         for i, doc in enumerate(retr, 1):
#             st.markdown(f"**Chunk {i} — {doc.metadata.get('source','unknown')}**")
#             st.write(doc.page_content)



import os
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time
from typing import List, Dict, Any

# --- Configuration ---
PERSIST_DIR = ".chroma"
EMBED_MODEL = "nomic-embed-text"
DEFAULT_CHAT_MODEL = os.environ.get("CHAT_MODEL", "llama3:3b")
TOP_K = 4

# --- Styling ---
def load_custom_css():
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    
    .sub-header {
        text-align: center;
        color: #6C757D;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-style: italic;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .source-chip {
        background-color: #E3F2FD;
        color: #1976D2;
        padding: 0.2rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 0.2rem;
        display: inline-block;
    }
    
    .answer-box {
        background-color: #red;
        border-left: 4px solid #28A745;
        padding: 1.5rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .context-box {
        background-color: #FFF3CD;
        border: 1px solid #FFEAA7;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .sidebar-info {
        background-color: black;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .status-success {
        color: #28A745;
        font-weight: 600;
    }
    
    .status-error {
        color: #DC3545;
        font-weight: 600;
    }
    
    .loading-spinner {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Utility Functions ---
@st.cache_resource
def initialize_vectordb():
    """Initialize and cache the vector database connection."""
    try:
        embeddings = OllamaEmbeddings(model=EMBED_MODEL)
        vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
        return vectordb, None
    except Exception as e:
        return None, str(e)

def format_sources(docs: List[Any]) -> str:
    """Format document sources as chips."""
    sources = []
    for doc in docs:
        source = doc.metadata.get('source', 'Unknown')
        # Extract filename from path
        source_name = os.path.basename(source) if source != 'Unknown' else 'Unknown'
        sources.append(f'<span class="source-chip">{source_name}</span>')
    return ' '.join(set(sources))  # Remove duplicates

def create_prompt(context: str, question: str) -> str:
    """Create an enhanced prompt for the RAG system."""
    return f"""You are an intelligent research assistant with access to a knowledge base. Your task is to provide accurate, helpful, and well-structured answers based on the provided context.

Guidelines:
- Use the context information to answer the question comprehensively
- If the context doesn't contain sufficient information, clearly state what you don't know
- Structure your response clearly with main points and supporting details
- Be concise but thorough
- Cite relevant information from the context when applicable

Context Information:
{context}

Question: {question}

Please provide a detailed and helpful answer:"""

# --- Main Application ---
def main():
    # Page configuration
    st.set_page_config(
        page_title="Local Multi-Agent RAG System",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load custom CSS
    load_custom_css()
    
    # Header
    st.markdown('<h1 class="main-header">🧠 Local Multi-Agent RAG System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Intelligent Document Q&A with Ollama + Chroma + Streamlit</p>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model Settings
        st.subheader("Model Settings")
        chat_model = st.text_input(
            "Chat Model", 
            value=DEFAULT_CHAT_MODEL,
            help="Enter the Ollama model name (e.g., llama3:3b, qwen2.5:7b)"
        )
        
        top_k = st.slider(
            "Top-K Retrieved Chunks", 
            min_value=1, 
            max_value=15, 
            value=TOP_K,
            help="Number of most relevant document chunks to retrieve"
        )
        
        # System Status
        st.subheader("📊 System Status")
        vectordb, error = initialize_vectordb()
        
        if vectordb:
            try:
                # Try to get collection info
                collection = vectordb._collection
                doc_count = collection.count()
                st.markdown(f"""
                <div class="sidebar-info">
                    <div class="status-success">✅ Vector DB Connected</div>
                    <div><strong>Documents:</strong> {doc_count}</div>
                    <div><strong>Embedding Model:</strong> {EMBED_MODEL}</div>
                    <div><strong>Storage:</strong> {PERSIST_DIR}</div>
                </div>
                """, unsafe_allow_html=True)
            except:
                st.markdown(f"""
                <div class="sidebar-info">
                    <div class="status-success">✅ Vector DB Connected</div>
                    <div><strong>Embedding Model:</strong> {EMBED_MODEL}</div>
                    <div><strong>Storage:</strong> {PERSIST_DIR}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="sidebar-info">
                <div class="status-error">❌ Connection Failed</div>
                <div><small>{error}</small></div>
            </div>
            """, unsafe_allow_html=True)
        
        # Help Section
        with st.expander("ℹ️ How to Use"):
            st.markdown("""
            1. **Ensure Setup**: Run `python ingest.py` to populate the vector database
            2. **Configure Model**: Default uses phi3:3.8b (working model from your setup)
            3. **Ask Questions**: Enter your question about the documents
            4. **Review Results**: Check the answer and source chunks
            5. **Adjust Settings**: Modify Top-K for more/fewer context chunks
            """)
        
        # Clear History
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Main Content Area
    if not vectordb:
        st.error("❌ **Vector Database Not Available**")
        st.warning("""
        Please ensure you have:
        1. Run `python ingest.py` to create the vector database
        2. Installed required dependencies: `pip install chromadb langchain-community`
        3. Started Ollama service and pulled the embedding model
        """)
        st.stop()
    
    # Question Input
    col1, col2 = st.columns([4, 1])
    with col1:
        user_question = st.text_input(
            "💬 Ask a question about your documents:",
            placeholder="e.g., What are the main topics discussed in the documents?",
            key="question_input"
        )
    with col2:
        ask_button = st.button("🚀 Ask", type="primary")
    
    # Process Question
    if (ask_button or user_question) and user_question.strip():
        process_question(user_question, vectordb, chat_model, top_k)
    
    # Display Chat History
    if st.session_state.chat_history:
        st.subheader("💬 Chat History")
        for i, (q, a, sources) in enumerate(reversed(st.session_state.chat_history[-5:]), 1):
            with st.expander(f"Q{len(st.session_state.chat_history)-i+1}: {q[:50]}..." if len(q) > 50 else f"Q{len(st.session_state.chat_history)-i+1}: {q}"):
                st.markdown(f'<div class="answer-box">{a}</div>', unsafe_allow_html=True)
                if sources:
                    st.markdown(f"**Sources:** {sources}", unsafe_allow_html=True)

def process_question(question: str, vectordb, chat_model: str, top_k: int):
    """Process user question and generate response."""
    with st.spinner("🔍 Searching knowledge base..."):
        try:
            # Retrieve relevant documents
            start_time = time.time()
            retrieved_docs = vectordb.similarity_search(question, k=top_k)
            retrieval_time = time.time() - start_time
            
            if not retrieved_docs:
                st.warning("No relevant documents found for your question.")
                return
            
            # Prepare context
            context_parts = []
            for i, doc in enumerate(retrieved_docs, 1):
                source = doc.metadata.get('source', 'Unknown')
                context_parts.append(f"[Source {i}] {doc.page_content}")
            
            context = "\n\n".join(context_parts)
            
            # Generate response
            with st.spinner("🤖 Generating answer..."):
                prompt = create_prompt(context, question)
                
                try:
                    llm = Ollama(model="phi3:3.8b")
                    start_time = time.time()
                    answer = llm.invoke(prompt)
                    generation_time = time.time() - start_time
                    
                    # Display results
                    st.subheader("✨ Answer")
                    st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)
                    
                    # Performance metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("⚡ Retrieval Time", f"{retrieval_time:.2f}s")
                    with col2:
                        st.metric("🧠 Generation Time", f"{generation_time:.2f}s")
                    with col3:
                        st.metric("📄 Chunks Retrieved", len(retrieved_docs))
                    
                    # Sources
                    sources_html = format_sources(retrieved_docs)
                    st.markdown(f"**🔗 Sources:** {sources_html}", unsafe_allow_html=True)
                    
                    # Store in chat history
                    st.session_state.chat_history.append((question, answer, sources_html))
                    
                    # Show retrieved chunks
                    with st.expander("📋 View Retrieved Context Chunks"):
                        for i, doc in enumerate(retrieved_docs, 1):
                            source = doc.metadata.get('source', 'Unknown Source')
                            source_name = os.path.basename(source)
                            
                            st.markdown(f"""
                            <div class="context-box">
                                <strong>📄 Chunk {i} from {source_name}</strong>
                                <p>{doc.page_content}</p>
                            </div>
                            """, unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"❌ **Model Error**: {str(e)}")
                    st.info("💡 **Suggestions:**")
                    st.markdown("""
                    - Check if the model name is correct
                    - Ensure Ollama is running: `ollama serve`
                    - Verify the model is installed: `ollama list`
                    - Try pulling the model: `ollama pull {model_name}`
                    """.format(model_name=chat_model))
        
        except Exception as e:
            st.error(f"❌ **Retrieval Error**: {str(e)}")
            st.info("This usually indicates an issue with the vector database or embedding model.")

if __name__ == "__main__":
    main()
