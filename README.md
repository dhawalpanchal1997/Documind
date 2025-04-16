# 📘 DocuMind AI

**DocuMind AI** is an intelligent research assistant built with Streamlit and LangChain, designed to answer user questions by deeply analyzing PDF documents using Retrieval-Augmented Generation (RAG). It leverages OpenAI's cutting-edge language and embedding models with a strong anti-hallucination framework.

## 🚀 Features

- 🔍 **PDF Parsing & Chunking**: Automatically splits uploaded PDFs into overlapping chunks for better context retention.
- 🧠 **RAG-Based QA**: Answers are strictly grounded in document excerpts using a tailored anti-hallucination prompt.
- 🔐 **OpenAI Integration**: Uses `gpt-4o` or `gpt-3.5-turbo` via OpenAI's API.
- 🛠️ **Query Expansion & Filtering**: Expands queries for better recall and filters retrieved chunks by relevance score.
- 📊 **Interactive UI**: Streamlit-based interface with sidebar configuration and chat-style Q&A.
- 🧪 **Debug Mode**: View detailed insights on document retrieval and filtering process.

## 🏗️ Architecture

- **Frontend**: Streamlit
- **Backend**: LangChain with OpenAI LLMs & Embeddings
- **Vector Store**: In-Memory (InMemoryVectorStore)
- **PDF Loader**: PDFPlumber
- **Environment Management**: `python-dotenv`

## 🧰 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Your `requirements.txt` should include:

```text
streamlit
langchain
langchain-community
langchain-openai
langchain-core
openai
python-dotenv
pdfplumber
```

> Make sure your `.env` file contains your OpenAI key:
>
> ```
> OPENAI_API_KEY=your_key_here
> ```

## 📂 Folder Structure

```
.
├── rag_deep_openai.py         # Main app logic
├── document_store/
│   └── pdfs/                  # PDF storage
├── .env                       # OpenAI key (not committed)
└── README.md
```

## 🖥️ How to Use

1. **Run the app**:

   ```bash
   streamlit run rag_deep_openai.py
   ```

2. **Upload a PDF document** via the Streamlit interface.
3. **Ask a question** about the content.
4. **Receive grounded answers** backed by the document’s excerpts.

## ⚙️ Configuration Options

Configure these via the sidebar in the Streamlit app:

- LLM Model (gpt-4o-mini, gpt-4o, gpt-3.5-turbo)
- Chunk Size & Overlap
- Enable Debug Mode for verbose document retrieval analysis

## 🧠 Prompt Engineering

A strong system prompt is used to prevent hallucinations by:

- Forcing answers to rely **only** on retrieved document content
- Returning "I don't have enough information..." when needed
- Including **direct quotes** from source material

## 📌 TODOs / Future Enhancements

- Add persistent vector storage (e.g., FAISS or Chroma)
- Multi-document support
- UI enhancements & document highlighting
- Authentication & session-based context tracking

## 📄 License

MIT License
