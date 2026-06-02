# 🤖 BuddyAI

BuddyAI is an AI-powered document intelligence assistant built using **Retrieval-Augmented Generation (RAG)**. Users can upload any PDF document and interact with it through natural language questions. The system retrieves the most relevant information from the document and uses **Mistral AI** to generate accurate, context-aware responses.

---

## 📸 Application Preview

### Home Screen

![Home Screen](about_home.png)<img width="1817" height="895" alt="about_home png" src="https://github.com/user-attachments/assets/76da1121-7a3d-4a9c-a714-ebb1575c5e53" />


### Document Chat Interface

![Chat Interface](answer.png)<img width="1827" height="787" alt="answer png" src="https://github.com/user-attachments/assets/b4284a55-b713-4bcb-a559-fecf28dc7cb5" />


### Processed

![Retrieved Context](process.png)<img width="1873" height="826" alt="process png" src="https://github.com/user-attachments/assets/2f838895-c058-40cf-8ba3-f3b676ecf0a6" />

## 🌐 Live Demo

Experience BuddyAI live:

👉 **BuddyAI Application:** https://buddyai-iumt4gpp38zhfzfajs3vyk.streamlit.app/

Upload a PDF document and interact with it using natural language questions powered by Retrieval-Augmented Generation (RAG), ChromaDB, and Mistral AI.
* 🌐 Deployed and accessible via Streamlit Cloud

## 🚀 Features

* 📄 Upload and analyze any PDF document
* 🔍 Semantic search using vector embeddings
* 🧠 Retrieval-Augmented Generation (RAG)
* 🤖 Powered by Mistral AI
* 📚 Dynamic chunking based on document size
* ⚡ ChromaDB vector database integration
* 🎯 Context-grounded responses to reduce hallucinations
* 💬 Interactive Streamlit user interface
* 🔎 View retrieved context for transparency

---

## 🏗️ Project Architecture

```text
PDF Upload
    │
    ▼
Document Loader
    │
    ▼
Text Chunking
    │
    ▼
HuggingFace Embeddings
    │
    ▼
ChromaDB Vector Store
    │
    ▼
Similarity Search
    │
    ▼
Retrieved Context
    │
    ▼
Mistral AI
    │
    ▼
Final Answer
```

---

## 🧠 Tech Stack

### Frontend

* Streamlit

### LLM

* Mistral AI

### Framework

* LangChain

### Vector Database

* ChromaDB

### Embeddings

* Sentence Transformers
* HuggingFace Embeddings

### Document Processing

* PyPDFLoader
* PDFPlumber

---

## 📂 Project Structure

```text
BuddyAI/
│
├── app.py                # Streamlit application
├── database.py           # Vector database creation pipeline
├── main.py               # CLI-based RAG implementation
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/AyushChaki/BuddyAI.git
cd BuddyAI
```

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Environment

Windows:

```bash
.venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root.

```env
MISTRAL_API_KEY=your_api_key_here
```

---

## ▶️ Run Application

```bash
streamlit run app.py
```

The application will launch in your browser.

---

## 💡 How It Works

1. User uploads a PDF document.
2. The document is loaded and processed.
3. Text is split into chunks using Recursive Character Text Splitting.
4. Chunks are converted into embeddings using HuggingFace models.
5. Embeddings are stored in ChromaDB.
6. User asks a question.
7. Relevant chunks are retrieved through semantic similarity search.
8. Retrieved context is passed to Mistral AI.
9. BuddyAI generates a context-aware answer.

---

## 🎯 Use Cases

* Research Paper Analysis
* Academic Study Assistant
* Document Question Answering
* Technical Documentation Search
* Enterprise Knowledge Retrieval
* Book and Report Summarization

---

## 📈 Future Improvements

* Multi-PDF Support
* Chat History Memory
* Hybrid Search (BM25 + Vector Search)
* Citation-Based Responses
* Support for DOCX and TXT Files
* Cloud Vector Database Integration

---

## 👨‍💻 Author

**Ayush Chaki**

* GitHub: https://github.com/AyushChaki
* LinkedIn: https://www.linkedin.com/in/ayush-chaki-a49165310/

---

## ⭐ Support

If you found this project useful, consider giving it a star on GitHub.
