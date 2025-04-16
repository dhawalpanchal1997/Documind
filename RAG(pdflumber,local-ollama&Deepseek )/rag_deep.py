import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    
    /* Chat Input Styling */
    .stChatInput input {
        background-color: #1E1E1E !important;
        color: #FFFFFF !important;
        border: 1px solid #3A3A3A !important;
    }
    
    /* User Message Styling */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #1E1E1E !important;
        border: 1px solid #3A3A3A !important;
        color: #E0E0E0 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Assistant Message Styling */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background-color: #2A2A2A !important;
        border: 1px solid #404040 !important;
        color: #F0F0F0 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Avatar Styling */
    .stChatMessage .avatar {
        background-color: #00FFAA !important;
        color: #000000 !important;
    }
    
    /* Text Color Fix */
    .stChatMessage p, .stChatMessage div {
        color: #FFFFFF !important;
    }
    
    .stFileUploader {
        background-color: #1E1E1E;
        border: 1px solid #3A3A3A;
        border-radius: 5px;
        padding: 15px;
    }
    
    h1, h2, h3 {
        color: #00FFAA !important;
    }
    </style>
    """, unsafe_allow_html=True)

PROMPT_TEMPLATE = """
You are an expert research assistant analyzing documents. Use ONLY the information from the provided context to answer the query.
If the context doesn't contain the information needed, state that clearly.

Relevant Document Sections:
{document_context}

User Query: {user_query}

Instructions:
1. Answer based solely on the provided context
2. Cite specific sections when possible
3. Be precise and factual
4. If uncertain, explain your uncertainty rather than guessing

Answer:
"""
PDF_STORAGE_PATH = '/Users/dhawalpanchal/Documents/practice projects/docparser/document_store/pdfs/'
EMBEDDING_MODEL = OllamaEmbeddings(model="nomic-embed-text")
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
LANGUAGE_MODEL = OllamaLLM(model="llama3.2")


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
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=500,
        add_start_index=True
    )
    chunks = text_processor.split_documents(raw_documents)
    # Console debugging
    print(f"\n{'='*50}\nChunk Analysis:")
    print(f"Total Chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(f"Length: {len(chunk.page_content)} characters")
        #print(f"Preview: {chunk.page_content[:]}...")
        print(f"Metadata: {chunk.metadata}")
    print(f"{'='*50}\n")
    
    return chunks

def index_documents(document_chunks):
    # Console debugging
    print(f"\n{'='*50}\nEmbedding Analysis:")
    print(f"Using Model: {EMBEDDING_MODEL.model}")
    
    for i, chunk in enumerate(document_chunks):
        print(f"\nEmbedding Chunk {i+1}:")
        #print(f"Content: {chunk.page_content[:]}...")
        
        # Get embeddings for visualization (optional - can be resource intensive)
        try:
            embedding = EMBEDDING_MODEL.embed_documents([chunk.page_content])[0]
            #print(f"Embedding dimension: {len(embedding)}")
            #print(f"Embedding : {embedding[:]}")
        except Exception as e:
            print(f"Embedding generation error: {str(e)}")
        
    
    DOCUMENT_VECTOR_DB.add_documents(document_chunks)
    #print(f"\nTotal vectors in store: {len(DOCUMENT_VECTOR_DB.docstore._dict)}")
    print(f"{'='*50}\n")

def expand_query(query, model):
    """Expand a user query to improve retrieval by generating related terms."""
    expansion_prompt = f"""
    Given the following user query, generate 3-5 related search terms or phrases that would help
    find relevant information. Format as a comma-separated list.
    
    USER QUERY: {query}
    
    RELATED SEARCH TERMS:
    """
    
    expanded_terms = model.invoke(expansion_prompt).strip()
    expanded_query = f"{query}, {expanded_terms}"
    
    # Log the expansion for debugging
    print(f"Original query: {query}")
    print(f"Expanded query: {expanded_query}")
    
    return expanded_query

# Then modify your find_related_documents function
def find_related_documents(query):
    expanded_query = expand_query(query, LANGUAGE_MODEL)
    return DOCUMENT_VECTOR_DB.similarity_search(expanded_query, k=4)

def generate_answer(user_query, context_documents):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    return response_chain.invoke({"user_query": user_query, "document_context": context_text})


# UI Configuration


st.title("📘 DocuMind AI")
st.markdown("### Your Intelligent Document Assistant")
st.markdown("---")

# File Upload Section
uploaded_pdf = st.file_uploader(
    "Upload Research Document (PDF)",
    type="pdf",
    help="Select a PDF document for analysis",
    accept_multiple_files=False

)

if uploaded_pdf:
    saved_path = save_uploaded_file(uploaded_pdf)
    raw_docs = load_pdf_documents(saved_path)
    processed_chunks = chunk_documents(raw_docs)
    index_documents(processed_chunks)
    
    st.success("✅ Document processed successfully! Ask your questions below.")
    
    user_input = st.chat_input("Enter your question about the document...")
    
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        
        with st.spinner("Analyzing document..."):
            relevant_docs = find_related_documents(user_input)
            ai_response = generate_answer(user_input, relevant_docs)
            
        with st.chat_message("assistant", avatar="🤖"):
            st.write(ai_response)