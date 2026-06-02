#load pdf
#split into chunks
#embeddings
#store in chroma 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

loader = PyPDFLoader("document_loaders/stevejobs.pdf")
docs = loader.load()

num_pages = len(docs)

if num_pages <= 20:
    chunk_size = 500
    chunk_overlap = 100
elif num_pages <= 100:
    chunk_size = 800
    chunk_overlap = 150
else:
    chunk_size = 1000
    chunk_overlap = 200

splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)

#creating chunks of the document then do embeddings and store in vector database(VDB).they use approximate nearest neighbor search(ANN) to find the relevant chunks and then generate answer from those chunks.Eg:IVF(Inverted File Index)
#Question answering will be done on the chunks of the document(similarity of embeddings) and then the answer will be generated.This is how RAG works.
chunks = splitter.split_documents(docs)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./chroma_db",
    collection_name="steve_jobs"
)
#vectorstore cannot be invoked
print("Chroma database created successfully!")
retriever=vectorstore.as_retriever()
docs=retriever.invoke("Explain the history behind the making of iphone")
for d in docs:
    print(d.page_content)
    print("-" * 80)