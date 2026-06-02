from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
load_dotenv()
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore=Chroma(persist_directory="./chroma_db",
                    embedding_function=embedding_model,
                    collection_name="steve_jobs"
)
retriever=vectorstore.as_retriever( search_type="similarity",
                                    search_kwargs={"k": 8}
                        )
llm =ChatMistralAI(model="mistral-small-2603")
#prompt template
prompt = ChatPromptTemplate.from_messages([("system",
                                             """You are a helpful AI assistant.messages
                                
                                             USE ONLY the provided context to answer the question.
                                             
                                             If the answer is not present in the context,
                                             say:"I could not find the answer in the provided document."
                                             """),
                                           ("human",
                                            """ Context:
                                            
                                            {context}
                                            
                                            
                                            Question:{question}
                                            """
                                            )
                                        ]
                         )
print("Rag system is ready to answer your questions!")
print("press 0 to exit")
while True:
    query = input("You:")
    if query == "0":
        print("Goodbye!")
        break
    docs = retriever.invoke(query)
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )
    final_prompt=prompt.invoke({
        "context":context,
        "question":query
    }
    )
    final_prompt = prompt.format_prompt(context=context, question=query).to_messages()
    response = llm.invoke(final_prompt)
    print(f"\n AI: {response.content}")


