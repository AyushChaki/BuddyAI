import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

st.set_page_config(
    page_title="BuddyAI",
    page_icon="🤖",
    layout="wide"
)
st.markdown("""
<style>

.stApp {
    background: #F5F1EA;
}

/* Global Text */
html, body, [class*="css"] {
    color: #111827 !important;
}

/* Hero Section */
.hero {
    background: white;
    padding: 35px;
    border-radius: 24px;
    border: 1px solid #E5E7EB;
    box-shadow: 0px 6px 24px rgba(0,0,0,0.08);
    margin-bottom: 25px;
}

.hero-title {
    font-size: 52px;
    font-weight: 800;
    color: #111827;
    margin-bottom: 10px;
}

.hero-subtitle {
    font-size: 18px;
    color: #4B5563;
    line-height: 1.7;
}

/* Cards */
.info-card {
    background: white;
    color: #111827;
    padding: 30px;
    border-radius: 20px;
    border: 1px solid #E5E7EB;
    box-shadow: 0px 6px 20px rgba(0,0,0,0.06);
}

/* Answer Box */
.answer-box {
    background: white;
    color: #111827;
    padding: 28px;
    border-radius: 20px;
    border-left: 6px solid #8B6F47;
    box-shadow: 0px 6px 20px rgba(0,0,0,0.08);
    line-height: 1.8;
    font-size: 17px;
}

/* Metrics */
[data-testid="stMetricValue"] {
    color: #111827 !important;
    font-size: 42px !important;
    font-weight: 700 !important;
}

[data-testid="stMetricLabel"] {
    color: #6B7280 !important;
    font-weight: 600;
}

/* Input Box */
.stTextInput input {
    background: white !important;
    color: #111827 !important;
    border: 1px solid #D1D5DB !important;
    border-radius: 12px !important;
}

/* Buttons */
.stButton > button {
    background: #111827;
    color: white;
    border-radius: 12px;
    border: none;
    padding: 10px 24px;
    font-weight: 600;
}

.stButton > button:hover {
    background: #374151;
    color: white;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: white;
    border-right: 1px solid #E5E7EB;
}

section[data-testid="stSidebar"] * {
    color: #111827 !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    color: #111827 !important;
    font-weight: 600;
}

/* Expander */
.streamlit-expanderHeader {
    color: #111827 !important;
    font-weight: 600;
}

/* Headings */
h1, h2, h3 {
    color: #111827 !important;
}

/* Paragraphs */
p {
    color: #374151 !important;
}

</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def get_chunk_settings(num_pages):
    if num_pages <= 20:
        return 500, 100, 4
    elif num_pages <= 100:
        return 800, 150, 6
    else:
        return 1000, 200, 8


def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    loader = PyPDFLoader(temp_path)
    docs = loader.load()
    print(f"Pages loaded: {len(docs)}")

    for i in range(5):
        print(f"\nPAGE {i+1}")
        print(docs[i].page_content[:1000])
        print("="*100)

    num_pages = len(docs)
    chunk_size, chunk_overlap, k = get_chunk_settings(num_pages)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_documents(docs)

    embedding_model = load_embedding_model()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
         collection_name=f"doc_{uploaded_file.name.replace('.', '_')}"
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 12}
    )

    os.remove(temp_path)

    return retriever, num_pages, len(chunks), chunk_size, chunk_overlap, k


def generate_answer(query, retriever):
    retrieved_docs = retriever.invoke(query)
    print("\n\n========== RETRIEVED CHUNKS ==========\n")

    for i, doc in enumerate(retrieved_docs):
        print(f"\nChunk {i+1}\n")
        print(doc.page_content[:1000])
        print("\n" + "-"*80)

    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """
You are BuddyAI, a helpful document-based AI assistant.

Use ONLY the provided context to answer the user's question.
If the context contains relevant information,
answer using that information.

Only say
"I could not find the answer in the provided document."
when the retrieved context is completely unrelated.

Keep the answer clear, structured, and beginner-friendly.
"""
         ),
        ("human",
         """
Context:
{context}

Question:
{question}
"""
         )
    ])

    llm = ChatMistralAI(
        model="mistral-small-2603",
        temperature=0.2
    )

    final_prompt = prompt.format_prompt(
        context=context,
        question=query
    ).to_messages()

    response = llm.invoke(final_prompt)
    return response.content, retrieved_docs


with st.sidebar:
    st.title("🤖 BuddyAI")
    st.caption("Your intelligent document companion")

    uploaded_file = st.file_uploader(
        "Upload your PDF document",
        type=["pdf"]
    )

    st.markdown("---")

    st.subheader("⚙️ Pipeline")
    st.markdown("""
    1. Upload PDF  
    2. Load document  
    3. Split into chunks  
    4. Generate embeddings  
    5. Store in vector database  
    6. Retrieve relevant context  
    7. Generate answer with Mistral AI  
    """)

    st.markdown("---")

    st.subheader("🧠 Tech Stack")
    st.markdown("""
    - Streamlit  
    - LangChain  
    - Mistral AI  
    - ChromaDB  
    - HuggingFace Embeddings  
    - PyPDFLoader  
    """)

st.markdown("""
<div class="hero">
    <div class="hero-title">🤖 BuddyAI</div>
    <div class="hero-subtitle">
        Upload any PDF and ask questions from it using Retrieval-Augmented Generation.
    </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["💬 Chat with Document", "ℹ️ About BuddyAI"])

with tab1:
    if uploaded_file is None:
        st.info("Upload a PDF from the sidebar to begin.")
    else:
        with st.spinner("Reading, chunking, and embedding your document..."):
            retriever, num_pages, total_chunks, chunk_size, chunk_overlap, k = process_pdf(uploaded_file)

        st.success("Document processed successfully.")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Pages", num_pages)
        col2.metric("Chunks", total_chunks)
        col3.metric("Chunk Size", chunk_size)
        col4.metric("Retrieved Chunks", k)

        st.markdown("### Ask a question")

        query = st.text_input(
            "Type your question here",
            placeholder="Example: Summarize this document"
        )

        if st.button("Generate Answer"):
            if not query.strip():
                st.warning("Please enter a question first.")
            else:
                with st.spinner("BuddyAI is thinking..."):
                    answer, retrieved_docs = generate_answer(query, retriever)

                st.markdown("### 🤖 BuddyAI Answer")
                st.info(answer)
                

                with st.expander("View Retrieved Context"):
                    for i, doc in enumerate(retrieved_docs, start=1):
                        st.markdown(f"#### Chunk {i}")
                        st.write(doc.page_content)
                        st.markdown("---")

with tab2:
    st.markdown("""
<div class="info-card" style="color:#111827;">

<h2>About BuddyAI</h2>

<p>
<b>BuddyAI</b> is a professional AI-powered PDF question-answering assistant built using
Retrieval-Augmented Generation. It helps users upload documents and receive answers
grounded strictly in the uploaded content.
</p>

<h3>How BuddyAI Works</h3>

<p>
When a user uploads a PDF, BuddyAI extracts the text, splits it into meaningful chunks,
converts those chunks into embeddings, stores them in a vector database, and retrieves
the most relevant sections when a question is asked.
</p>

<p>
The retrieved context is then passed to Mistral AI, which generates a clear and
document-specific response. This reduces hallucination and makes the system more reliable
than a normal chatbot.
</p>

<h3>Key Features</h3>

<ul>
<li>PDF upload support</li>
<li>Automatic chunk-size tuning based on document length</li>
<li>Semantic search using vector embeddings</li>
<li>Mistral AI-powered answering</li>
<li>Retrieved context visibility</li>
<li>Clean professional Streamlit interface</li>
</ul>

<h3>Best Use Cases</h3>

<ul>
<li>Research paper analysis</li>
<li>Study notes question-answering</li>
<li>Book summarization</li>
<li>Report understanding</li>
<li>Document-based chatbot projects</li>
</ul>

</div>
""", unsafe_allow_html=True)