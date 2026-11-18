import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
load_dotenv() 
loader=PyPDFLoader("RAG_and_LangChain_Guide.pdf")
documents=loader.load()
splitter =RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunks=splitter.split_documents(documents)
print(f" the total chunks are created {len(chunks)}")
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#store embeddinsg in vetro database
vectorstore=Chroma.from_documents(
    documents=chunks,embedding=embedding_model,collection_name="my_rag_collection",persist_directory="chroma_db"
)
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)
query=input("ask your question:")
retrived_documents=vectorstore.similarity_search(query,k=3)
context="\n\n".join([doc.page_content for doc in retrived_documents])
prompt=f""" answer the question using only provideed conetnt ceonetxt, if the answer is not presnt in the conetxt , just say that there is no related data
context:
{context}
query:
{query}
answer
"""
response=llm.invoke(prompt)
print("\nAnswer:")
print(response.content)