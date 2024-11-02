import streamlit as st
import os
import sys
from pathlib import Path
import tempfile
from dotenv import load_dotenv

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.pdf_handler import PDFHandler
from modules.vector_store import VectorStore
from modules.qa_chain import QAChain

# Custom CSS for dark mode and styling
custom_css = """


<style>
    /* Custom styles for dark mode */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }

    /* Remove the default padding and margin from the Streamlit app */
    .stContainer {
        padding-top: 0 !important; /* Remove top padding */
        margin-top: 0 !important; /* Remove top margin */
    }

    /* Style for main headers */
    .main-header {
        color: #FF4B4B;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }

    /* Style for subheaders */
    .sub-header {
        color: #00BFFF;
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 1rem;
    }

    /* Container styles */
    .qa-input-container, 
    .chat-history-container {
        width: 100%;
        max-width: 100%;
        margin: 0 auto;
        padding: 1rem 0;
    }

    /* Style for question boxes */
    .question-box {
        background-color: #262730;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        width: 100%;
        display: block;
        box-sizing: border-box;
    }

    /* Style for answer boxes */
    .answer-box {
        background-color: #1E1E1E;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0 1.5rem 0;
        border-left: 4px solid #00BFFF;
        width: 100%;
        display: block;
        box-sizing: border-box;
    }

    /* Style for source boxes */
    .source-box {
        background-color: #2D2D2D;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #FF4B4B;
        width: 100%;
        box-sizing: border-box;
    }

    /* Style for success messages */
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #1E472E;
        border: 1px solid #28A745;
        color: #FAFAFA;
        margin: 1rem 0;
        width: 100%;
        box-sizing: border-box;
    }

    /* Style for info messages */
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #1E3647;
        border: 1px solid #17A2B8;
        color: #FAFAFA;
        margin: 1rem 0;
        width: 100%;
        box-sizing: border-box;
    }

    /* Override Streamlit's default container padding */
    .stMarkdown {
        width: 100% !important;
        padding: 0 !important;
    }

    /* Make sure columns take full width */
    .row-widget.stButton, 
    .row-widget.stTextInput {
        width: 100%;
    }
    .stButton>button {
        margin-top: 25px;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #000000;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background-color: #000000;
        padding: 1rem;
    }
    
    /* Style sidebar hr separator */
    [data-testid="stSidebar"] hr {
        border-color: #000000;
        margin: 1rem 0;
    }
    
    /* Style sidebar text and headers */
    [data-testid="stSidebar"] .main-header,
    [data-testid="stSidebar"] .sub-header {
        color: #FF4B4B;
    }
    
    /* Style sidebar file uploader */
    [data-testid="stSidebar"] [data-testid="stFileUploader"] {
        background-color: #1E3647;
        border-radius: 0.5rem;
        
        padding: 1rem;
    }
</style>
"""






# Configure Streamlit page
st.set_page_config(
    page_title="PDF Question Answering System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/your-repo',
        'Report a bug': "https://github.com/yourusername/your-repo/issues",
        'About': "# PDF QA System\nThis is a RAG-based PDF question answering system."
    }
)

# Apply custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = set()
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None
    if 'pdf_handler' not in st.session_state:
        st.session_state.pdf_handler = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def custom_success_box(text):
    """Display a custom styled success message."""
    st.markdown(f'<div class="success-box">{text}</div>', unsafe_allow_html=True)

def custom_info_box(text):
    """Display a custom styled info message."""
    st.markdown(f'<div class="info-box">{text}</div>', unsafe_allow_html=True)

def initialize_components():
    """Initialize all necessary components."""
    try:
        # Load environment variables
        load_dotenv()
        
        # Check for required environment variables
        if not os.getenv('GEMINI_API_KEY'):
            st.error("⚠️ Error: GEMINI_API_KEY not found in environment variables!")
            st.stop()
            
        if not os.getenv('CHROMA_PERSIST_DIR'):
            persist_dir = str(Path.cwd() / "data" / "chroma")
            os.environ['CHROMA_PERSIST_DIR'] = persist_dir
        
        # Create persist directory if it doesn't exist
        Path(os.getenv('CHROMA_PERSIST_DIR')).mkdir(parents=True, exist_ok=True)
        
        # Initialize components if not already initialized
        if not st.session_state.pdf_handler:
            st.session_state.pdf_handler = PDFHandler()
        
        if not st.session_state.vector_store:
            st.session_state.vector_store = VectorStore(
                persist_directory=os.getenv('CHROMA_PERSIST_DIR'),
                collection_name="documents"
            )
        
        if not st.session_state.qa_chain:
            st.session_state.qa_chain = QAChain()
            
    except Exception as e:
        st.error(f"⚠️ Error initializing components: {str(e)}")
        st.stop()

def process_pdf(uploaded_file):
    """Process an uploaded PDF file."""
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Process PDF with progress bar
        progress_bar = st.progress(0)
        
        # Extract text
        progress_bar.progress(25)
        st.info("📄 Extracting text from PDF...")
        text = st.session_state.pdf_handler.extract_text_from_pdf(tmp_path)
        
        # Split text
        progress_bar.progress(50)
        st.info("✂️ Splitting text into chunks...")
        chunks = st.session_state.pdf_handler.split_text(text)
        
        # Add to vector store
        progress_bar.progress(75)
        st.info("🔄 Adding documents to vector store...")
        st.session_state.vector_store.add_documents(chunks)
        
        # Finish up
        progress_bar.progress(100)
        
        # Add to processed files
        st.session_state.processed_files.add(uploaded_file.name)
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        return True
        
    except Exception as e:
        st.error(f"⚠️ Error processing PDF: {str(e)}")
        return False

def display_sidebar():
    """Display and handle sidebar elements."""
    with st.sidebar:
        st.markdown('<p class="main-header">📁 Document Upload</p>', unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type="pdf",
            accept_multiple_files=True,
            key="pdf_uploader"
        )
        
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in st.session_state.processed_files:
                    st.write(f"📄 Processing: {uploaded_file.name}")
                    if process_pdf(uploaded_file):
                        custom_success_box(f"✅ Successfully processed: {uploaded_file.name}")
        
        if st.session_state.processed_files:
            st.markdown('<p class="sub-header">📚 Processed Documents</p>', unsafe_allow_html=True)
            for file in st.session_state.processed_files:
                st.markdown(f"- 📄 {file}")
        
        st.markdown('<p class="sub-header">ℹ️ About</p>', unsafe_allow_html=True)
        custom_info_box("""\
        This application allows you to:
        1. 📄 Upload PDF documents
        2. ❓ Ask questions about their content
        3. 🤖 Get AI-powered answers with source references
        """)

def handle_question(question: str):
    """Process a question and display the answer."""
    try:
        with st.spinner('🤔 Analyzing documents...'):
            retriever = st.session_state.vector_store.as_retriever()
            answer, sources = st.session_state.qa_chain.process_query(retriever, question)
            
            # Store in chat history
            st.session_state.chat_history.append((question, answer))
            
            # Display current Q&A
            st.markdown('<div class="qa-container">', unsafe_allow_html=True)
            st.markdown(f'<div class="question-box"> {question}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="answer-box">💡 {answer}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Display sources
            st.markdown('<p class="sub-header">📚 Sources:</p>', unsafe_allow_html=True)
            for i, doc in enumerate(sources, 1):
                with st.expander(f"📄 Source {i}"):
                    st.markdown(f'<div class="source-box">{doc.page_content}</div>', unsafe_allow_html=True)
                    
    except Exception as e:
        st.error(f"⚠️ Error processing question: {str(e)}")

def main():
    # Display custom title
    st.markdown('<p class="main-header">📚 PDF Question Answering System</p>', unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Initialize components
    initialize_components()
    
    # Display sidebar
    display_sidebar()
    
    # Main panel
    if not st.session_state.processed_files:
        custom_info_box("👈 Please upload some PDF documents using the sidebar to get started!")
    else:
        st.markdown(custom_css, unsafe_allow_html=True)

        # Display a custom label above the input and button container
        st.markdown('<p class="sub-header">❓ Ask a Question</p>', unsafe_allow_html=True)

        # Use a container for the input and button
        st.markdown('<div class="qa-input-container">', unsafe_allow_html=True)

        # Adjust columns and remove the text input's internal label
        col1, col2 = st.columns([4, 1])
        with col1:
            question = st.text_input("", placeholder="Enter your question about the documents...", key="question_input")
        with col2:
            ask_button = st.button("🔍 Ask", use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

        if ask_button and question:
            handle_question(question)
        
        # Display chat history
        if st.session_state.chat_history:
            st.markdown('<p class="sub-header">💬 Chat History</p>', unsafe_allow_html=True)
            st.markdown('<div class="chat-history-container">', unsafe_allow_html=True)
            for q, a in reversed(st.session_state.chat_history):
                st.markdown(f'<div class="question-box">❓ {q}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="answer-box">💡 {a}</div>', unsafe_allow_html=True)
                st.markdown("---")
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()