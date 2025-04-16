import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
#from extraction import load_document
#rom chucking import chunk_document 
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI models
EMBEDDING_MODEL = OpenAIEmbeddings(
    model="text-embedding-3-large", 
    openai_api_key=OPENAI_API_KEY
)

LANGUAGE_MODEL = ChatOpenAI(
    model="gpt-4o-mini-2024-07-18", 
    openai_api_key=OPENAI_API_KEY,
    temperature=0.1  # Low temperature for factual responses
)

# Initialize vector store
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)

# Strong anti-hallucination prompt
PROMPT_TEMPLATE = """
You are an AI research assistant that ONLY provides information found in the provided document excerpts.

Document Excerpts:
{document_context}

User Query: {user_query}

Important Instructions:
1. ONLY use information from the provided document excerpts
2. If the necessary information isn't in the document excerpts, respond with: "I don't have enough information in the provided documents to answer this question."
3. Do NOT use any prior knowledge outside the provided documents
4. Do NOT make assumptions or inferences beyond what's explicitly stated in the documents
5. Include exact quotes from the documents when possible
6. Be concise and factual

Your Answer:
"""
PDF_STORAGE_PATH = '/Users/dhawalpanchal/Documents/practice projects/docparser/document_store/pdfs/'

def save_uploaded_file(uploaded_file):
    file_path = PDF_STORAGE_PATH + uploaded_file.name
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

def load_pdf_documents(file_path):
    document_loader = PDFPlumberLoader(file_path)
    documents = document_loader.load()
    
    # Console debugging
    print(f"\n{'='*50}\nDocument Analysis:")
    for i, doc in enumerate(documents):
        print(f"\nPage {i+1}:")
        #print(f"Content: {doc.page_content[:]}...")
        print(f"Metadata: {doc.metadata}")
    print(f"{'='*50}\n")
    
    return documents

def chunk_documents(raw_documents):
    """Create meaningful chunks from the document"""
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=2000,  # Larger chunks for more context
        chunk_overlap=500,  # Significant overlap to maintain continuity
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True
    )
    chunks = text_processor.split_documents(raw_documents)
    
    # Enhance metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        # Ensure page number is present
        if "page" not in chunk.metadata:
            chunk.metadata["page"] = chunk.metadata.get("page_number", "unknown")
    
    return chunks

def find_related_documents(query):
    """Enhanced document retrieval"""
    # Expand the query for better retrieval
    expanded_query = expand_query(query)
    
    # Get initial documents
    docs = DOCUMENT_VECTOR_DB.similarity_search(
        expanded_query, 
        k=5,  # Retrieve more candidates
        fetch_k=10  # Consider more candidates
    )
    
    # Filter by confidence
    filtered_docs = filter_by_confidence(query, docs)
    
    # Show debugging info in development
    if st.session_state.get('debug_mode', False):
        show_retrieval_debug(query, expanded_query, docs, filtered_docs)
    
    return filtered_docs

def expand_query(query):
    """Expand the query with related terms"""
    expansion_prompt = f"""
    Given this user query about a document, generate 3-5 related search terms that 
    would help find relevant information. Return ONLY a comma-separated list.
    
    USER QUERY: {query}
    
    RELATED SEARCH TERMS:
    """
    
    try:
        expanded_terms = LANGUAGE_MODEL.invoke(expansion_prompt).content.strip()
        expanded_query = f"{query}, {expanded_terms}"
        return expanded_query
    except Exception as e:
        print(f"Query expansion error: {str(e)}")
        return query  # Fallback to original query

def filter_by_confidence(query, documents, similarity_threshold=0.65):
    """Filter documents based on relevance confidence"""
    if not documents:
        return []
        
    filtered_docs = []
    for doc in documents:
        relevance_prompt = f"""
        Rate how relevant this document excerpt is to the query on a scale of 1-10.
        Return ONLY the numeric score.
        
        QUERY: {query}
        
        DOCUMENT EXCERPT: {doc.page_content}
        
        RELEVANCE SCORE (1-10):
        """
        
        try:
            score_text = LANGUAGE_MODEL.invoke(relevance_prompt).content.strip()
            import re
            score_match = re.search(r'\b([0-9]|10)\b', score_text)
            score = float(score_match.group(0)) if score_match else 5.0
            score_normalized = score / 10.0  # Convert to 0-1 scale
            
            doc.metadata["relevance_score"] = score_normalized
            
            if score_normalized >= similarity_threshold:
                filtered_docs.append(doc)
                
        except Exception as e:
            print(f"Scoring error: {str(e)}")
            filtered_docs.append(doc)  # Include on error
    
    # If everything got filtered, return highest scoring doc
    if not filtered_docs and documents:
        documents.sort(key=lambda x: x.metadata.get("relevance_score", 0), reverse=True)
        filtered_docs = [documents[0]]
    
    return filtered_docs

def generate_answer(user_query, context_documents):
    """Generate answer based on retrieved documents"""
    # Sort by relevance score
    context_documents.sort(
        key=lambda x: x.metadata.get("relevance_score", 0), 
        reverse=True
    )
    
    # Format context with page numbers and relevance
    formatted_contexts = []
    for i, doc in enumerate(context_documents):
        page_info = f"[Page {doc.metadata.get('page', 'unknown')}]"
        score_info = f"[Relevance: {doc.metadata.get('relevance_score', 'N/A'):.2f}]" if isinstance(doc.metadata.get('relevance_score'), float) else "[Relevance: N/A]"
        formatted_contexts.append(f"{page_info} {score_info}\n{doc.page_content}")
    
    context_text = "\n\n---\n\n".join(formatted_contexts)
    
    # Generate answer
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    return response_chain.invoke({
        "user_query": user_query, 
        "document_context": context_text
    })

def show_retrieval_debug(original_query, expanded_query, initial_docs, filtered_docs):
    """Show debugging information for retrieval process"""
    with st.expander("Debug: Document Retrieval Analysis"):
        st.write(f"**Original Query:** {original_query}")
        st.write(f"**Expanded Query:** {expanded_query}")
        
        st.write("### Initial Documents Retrieved")
        for i, doc in enumerate(initial_docs):
            st.write(f"**Doc {i+1}** (Page {doc.metadata.get('page', 'unknown')})")
            st.write(f"Relevance: {doc.metadata.get('relevance_score', 'N/A')}")
            st.text(doc.page_content[:200] + "...")
            
        st.write("### Filtered Documents")
        for i, doc in enumerate(filtered_docs):
            st.write(f"**Doc {i+1}** (Page {doc.metadata.get('page', 'unknown')})")
            st.write(f"Relevance: {doc.metadata.get('relevance_score', 'N/A')}")
            st.text(doc.page_content[:200] + "...")

# Add configuration options to the Streamlit UI
def add_configuration_options():
    st.sidebar.header("Model Configuration")
    
    model_options = {
        "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
        "gpt-4o": "gpt-4o-2024-08-06",
        "gpt-3.5-turbo": "GPT-3.5 Turbo (Fastest)"
    }
    
    selected_model = st.sidebar.selectbox(
        "Select LLM Model", 
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x],
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
    
    # Return configuration options
    return {
        "model": selected_model,
        "chunk_size": chunk_size,
        "chunk_overlap": overlap
    }

# Main Streamlit UI
def main():
    st.title("📘 DocuMind AI")
    st.markdown("### Your Intelligent Document Assistant")
    st.markdown("---")
    
    # Add configuration sidebar
    config = add_configuration_options()
    
    # Update models based on configuration
    global LANGUAGE_MODEL
    LANGUAGE_MODEL = ChatOpenAI(
        model=config["model"],
        openai_api_key=OPENAI_API_KEY,
        temperature=0.1
    )
    
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
            #raw_docs = load_document(saved_path)
            raw_docs = load_pdf_documents(saved_path)
            
            # Use configuration for chunking
            text_processor = RecursiveCharacterTextSplitter(
                chunk_size=config["chunk_size"],
                chunk_overlap=config["chunk_overlap"],
                separators=["\n\n", "\n", ". ", " ", ""],
                add_start_index=True
            )
            processed_chunks = text_processor.split_documents(raw_docs)
            #processed_chunks = chunk_document(raw_docs,config["chunk_size"],config["chunk_overlap"])

            DOCUMENT_VECTOR_DB.add_documents(processed_chunks)
        
        st.success("✅ Document processed successfully! Ask your questions below.")
        
        user_input = st.chat_input("Enter your question about the document...")
        
        if user_input:
            with st.chat_message("user"):
                st.write(user_input)
            
            with st.spinner("Analyzing document..."):
                relevant_docs = find_related_documents(user_input)
                ai_response = generate_answer(user_input, relevant_docs)
                
            with st.chat_message("assistant", avatar="🤖"):
                st.write(ai_response.content)

if __name__ == "__main__":
    main()