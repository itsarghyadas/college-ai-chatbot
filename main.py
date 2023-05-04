import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')
chatbot_data_path = os.getenv('CHATBOT_DATA_DIR')
chroma_dir = os.getenv('CHROMA_DIR')

try:
    loader = PyPDFDirectoryLoader(chatbot_data_path)
    documents = loader.load()
except Exception as e:
    st.error(f"Error loading documents: {e}")
    raise

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=50)
texts = text_splitter.split_documents(documents)
persist_directory = chroma_dir
embeddings = OpenAIEmbeddings()

if os.path.exists(persist_directory):
    try:
        print("Loading from Existing Embeddings")
        docsearch = Chroma(persist_directory=persist_directory,
                           embedding_function=embeddings)
        qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(verbose=False, temperature=0.2), chain_type="stuff",
                                         retriever=docsearch.as_retriever(search_kwargs={"k": 2}), return_source_documents=True)
    except Exception as e:
        st.error(f"Error searching from existing Embeddings, please wait: {e}")
        raise
else:
    try:
        print("Creating New Embeddings")
        docsearch = Chroma.from_documents(
            documents=texts, embedding=embeddings, persist_directory=persist_directory)
        docsearch.persist()
        qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(verbose=False, temperature=0.2), chain_type="stuff",
                                         retriever=docsearch.as_retriever(search_kwargs={"k": 2}), return_source_documents=True)
    except Exception as e:
        st.error(f"Error creating new embeddings, please wait: {e}")
        raise

st.title("Document Chatbot")
st.write("This is a chatbot that can answer questions about a document.")
query = st.text_input("Enter your query here")

if query:
    try:
        result = qa({"query": query})
        st.success(result["result"])
        st.json({"Source: ": os.path.basename(result["source_documents"][0].metadata["source"]),
                 "Page Number: ": result["source_documents"][0].metadata["page"]})
    except Exception as e:
        st.error(f"Error getting answer: {e}")
        raise
