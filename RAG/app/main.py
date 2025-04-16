import streamlit as st
from services.extraction import load_document
from services.chunking import chunk_document
from utils.record_preparation import prepare_record  # Updated import
from services.search import search_documents
from services.synthesizer import Synthesizer
from database.vector_store import VectorStore
from config.settings import get_settings
import os

# Initialize vector store
vec = VectorStore()

PDF_STORAGE_PATH = '/Users/dhawalpanchal/Documents/practice projects/docparser/document_store/pdfs/'

def save_uploaded_file(uploaded_file):
    # Ensure directory exists
    os.makedirs(PDF_STORAGE_PATH, exist_ok=True)

    file_path = PDF_STORAGE_PATH + uploaded_file.name
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path


def show_retrieval_debug(user_input, ai_response):
    """Show debugging information for retrieval process"""
    with st.expander("Debug: Document Retrieval Analysis"):
        st.write(f"**Original Query:** {user_input}")
        st.write("### Retrieved Documents Analysis")
        
        # Display the response details if available
        if hasattr(ai_response, 'thought_process'):
            st.write("### Thought Process")
            for thought in ai_response.thought_process:
                st.write(f"- {thought}")
                
        # Display context information if available
        if hasattr(ai_response, 'context'):
            st.write("### Context Used")
            st.write(ai_response.context)

# Add configuration options to the Streamlit UI
def add_configuration_options():
    st.sidebar.header("Model Configuration")
    
    model_options = {
        "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
        "gpt-4o": "gpt-4o-2024-08-06",
        "gpt-3.5-turbo": "gpt-3.5-turbo"
    }
    
    embedding_model_options = {
        "text-embedding-3-large": "text-embedding-3-large",
        "text-embedding-3-small": "text-embedding-3-small",
    }

    selected_model = st.sidebar.selectbox(
        "Select LLM Model", 
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x],
        index=0
    )

    selected_embedding_model = st.sidebar.selectbox(
        "Select Embedding Model", 
        options=list(embedding_model_options.keys()),
        format_func=lambda x: embedding_model_options[x],
        index=0
    )

    chunk_size = st.sidebar.slider(
        "Chunk Size", 
        min_value=200, 
        max_value=2000, 
        value=800, 
        step=100,
        help="Size of document chunks in characters"
    )
    
    overlap = st.sidebar.slider(
        "Chunk Overlap", 
        min_value=0, 
        max_value=500, 
        value=150, 
        step=50,
        help="Overlap between document chunks"
    )  

    st.sidebar.checkbox(
        "Debug Mode", 
        value=False, 
        key="debug_mode",
        help="Show detailed retrieval information"
    )
    
    return {
        "model": selected_model,
        "embedding_model": selected_embedding_model,
        "chunk_size": chunk_size,
        "chunk_overlap": overlap
    }

# Main Streamlit UI
def main():
    # Get configuration from sidebar
    config = add_configuration_options()
    
    # Initialize settings with the selected configuration
    settings = get_settings(frozenset(config.items()))

    # Initialize vector store with the updated settings
    vec = VectorStore()
    st.title("📘 DocuMind AI")
    st.markdown("### Your Intelligent Document Assistant")
    st.markdown("---")
    
    st.sidebar.markdown("---")

    # File Upload Section
    uploaded_pdf = st.file_uploader(
        "Upload Research Document (PDF)",
        type="pdf",
        help="Select a PDF document for analysis",
        accept_multiple_files=False
    )

    if uploaded_pdf:
        saved_path = save_uploaded_file(uploaded_pdf)
        
        with st.spinner("Processing document..."):
            raw_docs = load_document(saved_path)
            
            processed_chunks = chunk_document(raw_docs,config["chunk_size"],config["chunk_overlap"])
        
            for i, chunk in processed_chunks:
                print(f"Chunk {i}: {chunk}")
        
            records_df = prepare_record(processed_chunks, saved_path, vec)

            print (records_df['content'])
            
            vec.create_tables()
            vec.create_index()  # DiskAnnIndex
            vec.upsert(records_df)

        st.success("✅ Document processed successfully! Ask your questions below.")
        
        user_input = st.chat_input("Enter your question about the document...")
        
        if user_input:
            with st.chat_message("user"):
                st.write(user_input)
            
            with st.spinner("Analyzing document..."):
                
                # Search for relevant documents
                relevant_docs = search_documents(user_input)
                
                # Generate answer based on retrieved documents
                ai_response = Synthesizer.generate_response(question=user_input, context=relevant_docs)
                 
                if st.session_state.get('debug_mode', False):
                    show_retrieval_debug(user_input,ai_response)
                
            with st.chat_message("assistant", avatar="🤖"):
                st.write(ai_response.answer)


if __name__ == "__main__":
    main()