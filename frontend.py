import streamlit as st
import shutil
import os
from vector_database import (
    update_vector_db,
    clear_vector_db,
    is_db_loaded,
    enhanced_retrieval,
    get_sources,
    FAISS_DB_PATH
)
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Initialize the LLM
llm_model = ChatGroq(
    model="deepseek-r1-distill-llama-70b",
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# Enhanced prompt template for better responses with source citation
custom_prompt_template = """
You are an AI assistant that answers questions based solely on the provided context.

Context information:
{context}

Instruction:
- Answer the question using only the information from the context above.
- If the answer isn't in the context, say "I don't have enough information to answer this question."
- Be concise and direct in your response.
- At the end of your response, mention the source documents you used in the format: [Source: filename1, filename2]
- Do not mention that you're using context or explain your reasoning.
- If the question is ambiguous or requires clarification, provide the most relevant answer based on the context.

Question: {question}

Answer:
"""


def get_context(documents):
    if not documents:
        return "No relevant information found."

    context = "\n\n".join([f"From {os.path.basename(doc.metadata.get('source', 'unknown'))}:\n{doc.page_content}"
                           for doc in documents])
    return context


def answer_query(documents, query):
    if not documents:
        return "I don't have enough information to answer this question based on the uploaded files."

    context = get_context(documents)
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain = prompt | llm_model

    try:
        response = chain.invoke({"question": query, "context": context}).content

        # Extract source filenames for additional citation
        source_files = list(set([os.path.basename(doc.metadata.get('source', 'unknown'))
                                 for doc in documents if 'source' in doc.metadata]))

        # Add source citation if not already in response
        if source_files and "[Source:" not in response:
            response += f"\n\n[Source: {', '.join(source_files)}]"

        return response
    except Exception as e:
        return f"Error generating response: {str(e)}"


# Streamlit App
st.set_page_config(
    page_title="Assignments Genie",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'db_initialized' not in st.session_state:
    st.session_state.db_initialized = is_db_loaded()
if 'query_count' not in st.session_state:
    st.session_state.query_count = 0
if 'session_id' not in st.session_state:
    st.session_state.session_id = os.urandom(16).hex()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Add a session timeout mechanism (10 queries max)
MAX_QUERIES = 10


def check_session_limit():
    if st.session_state.query_count >= MAX_QUERIES:
        st.error("Session limit reached (10 queries). Please refresh to start a new session.")
        return True
    return False


# Custom CSS for WhatsApp-like chat styling
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem !important;
        font-weight: 900 !important;
        text-align: center;
        color: #FF6B35;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        background: linear-gradient(135deg, #FF6B35, #FF8E53);
        -webkit-background-clip: text;
        padding: 10px;
    }
    .sub-header {
        font-size: 1.8rem !important;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
        font-weight: 500;
    }

    .stButton button {
        width: auto !important;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
        margin: 0 auto;
        display: block;
    }

    .stButton button:hover {
        background-color: #45a049;
    }
    .user-message {
        background-color: #DCF8C6;
        color: #000;
        padding: 12px;
        border-radius: 15px;
        margin-bottom: 10px;
        margin-left: 20%;
        margin-right: 5%;
        box-shadow: 0 1px 1px rgba(0,0,0,0.1);
        position: relative;
        word-wrap: break-word;
    }
    .assistant-message {
        background-color: #ECE5DD;
        color: #000;
        padding: 12px;
        border-radius: 15px;
        margin-bottom: 10px;
        margin-left: 5%;
        margin-right: 20%;
        box-shadow: 0 1px 1px rgba(0,0,0,0.1);
        position: relative;
        word-wrap: break-word;
    }
    .thinking-text {
        color: #888;
        font-style: italic;
        text-align: center;
        margin: 10px 0;
    }
    .upload-section {
        padding: 25px;
        border-radius: 15px;
        margin-bottom: 25px;
    }
    .chat-container {
        padding: 15px;
        max-height: 500px;
        overflow-y: auto;
        background-color: #fafafa;
        border-radius: 10px;
        margin-bottom: 15px;
    }
    /* Make file uploader more visible */
    .stFileUploader {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 20px;
        background-color: #f9f9f9;
    }
    .source-citation {
        font-size: 0.9em;
        color: #666;
        font-style: italic;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Main app content
if not st.session_state.db_initialized:
    # Show upload interface when no files are processed
    st.markdown('<h1 class="main-header">🧞‍♂️ Assignments Genie</h1>', unsafe_allow_html=True)
    st.markdown('<h3 class="sub-header">Drop stupid assignments. Get answers. Chill.</h3>', unsafe_allow_html=True)

    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    # File upload section
    uploaded_files = st.file_uploader(
        "Upload Files (PDF, PPT, DOCX, XLSX, CSV)",
        type=["pdf", "pptx", "ppt", "docx", "doc", "xlsx","CSV"],
        accept_multiple_files=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_files:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Process Files"):
                # Clear previous files
                if os.path.exists('uploads/'):
                    shutil.rmtree('uploads/')

                # Save uploaded files
                files_directory = 'uploads/'
                os.makedirs(files_directory, exist_ok=True)

                file_paths = []
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(files_directory, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    file_paths.append(file_path)

                # Process files
                try:
                    with st.spinner("Processing files..."):
                        # Clear old database and create new one
                        clear_vector_db(FAISS_DB_PATH)
                        update_vector_db(file_paths, FAISS_DB_PATH)
                        st.session_state.processed_files = [f.name for f in uploaded_files]
                        st.session_state.db_initialized = True
                        st.session_state.query_count = 0

                    st.success(f"✅ {len(uploaded_files)} file(s) processed successfully!")
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

else:
    # Show chat interface when files are processed
    st.markdown('<h1 class="main-header">🧞‍♂️ Assignments Genie</h1>', unsafe_allow_html=True)

    # Display processed files
    if st.session_state.processed_files:
        with st.expander("📁 Processed Files", expanded=False):
            for file in st.session_state.processed_files:
                st.write(f"- {file}")

    # Chat container
    # st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    st.markdown(
        '<div class="chat-container" style="margin-top: -15px; padding-top: 5px;">',
        unsafe_allow_html=True
    )

    # Display chat history
    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            st.markdown(f'<div class="user-message"><b>You:</b><br>{chat["content"]}</div>', unsafe_allow_html=True)
        elif chat["role"] == "thinking":
            st.markdown(f'<div class="thinking-text">{chat["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-message"><b>Genie:</b><br>{chat["content"]}</div>',
                        unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Query input at the bottom
    user_query = st.chat_input(
        "Ask anything about the uploaded files...",
        key="query_input"
    )

    if user_query:
        if check_session_limit():
            st.stop()

        try:
            # Add user query to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_query})

            # Show thinking message
            thinking_placeholder = st.empty()
            thinking_placeholder.markdown('<div class="thinking-text">Thinking...</div>', unsafe_allow_html=True)

            with st.spinner(""):
                retrieved_docs = enhanced_retrieval(user_query, k=5)

            # Update thinking message
            thinking_placeholder.markdown('<div class="thinking-text">Generating answer...</div>',
                                          unsafe_allow_html=True)

            with st.spinner(""):
                response = answer_query(retrieved_docs, user_query)

            # Remove thinking message
            thinking_placeholder.empty()

            # Add assistant response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})

            # Update query count
            st.session_state.query_count += 1

            # Show detailed sources in a separate expander
            sources = get_sources(retrieved_docs)
            if sources:
                with st.expander("📚 View Detailed Sources"):
                    for source_name, excerpts in sources.items():
                        st.write(f"**{source_name}:**")
                        for i, excerpt in enumerate(excerpts, 1):
                            st.write(f"{i}. {excerpt}")
                        st.write("---")

            st.rerun()

        except Exception as e:
            st.error(f"Error: {str(e)}")

# Session info and clear button in sidebar
with st.sidebar:
    st.info(f"Session ID: {st.session_state.session_id}")
    st.write(f"Queries used: {st.session_state.query_count}/{MAX_QUERIES}")

    if st.button("🗑️ Clear All & Start New Session"):
        clear_vector_db(FAISS_DB_PATH)
        if os.path.exists('uploads/'):
            shutil.rmtree('uploads/')
        st.session_state.processed_files = []
        st.session_state.db_initialized = False
        st.session_state.query_count = 0
        st.session_state.session_id = os.urandom(16).hex()
        st.session_state.chat_history = []
        st.success("New session started! All files cleared.")
        st.rerun()

st.caption("Assignments Genie - Your AI study assistant | Session auto-clears after 10 queries")