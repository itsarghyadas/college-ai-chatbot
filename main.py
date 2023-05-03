import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import TensorflowHubEmbeddings
from langchain.document_loaders import PyPDFDirectoryLoader
import os
from dotenv import load_dotenv
load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')
chatbot_data_path = os.getenv('CHATBOT_DATA_DIR')

loader = PyPDFDirectoryLoader(chatbot_data_path)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
embeddings = TensorflowHubEmbeddings()
docsearch = Chroma.from_documents(texts, embeddings)
qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(verbose=False, temperature=0), chain_type="stuff",
                                 retriever=docsearch.as_retriever(search_kwargs={"k": 1}), return_source_documents=True)

st.title("Document Chatbot")
st.write("This is a chatbot that can answer questions about a document.")
query = st.text_input("Enter your query here")
if query:
    result = qa({"query": query})
    st.success(result["result"])
    st.json({"Source: ": os.path.basename(result["source_documents"][0].metadata["source"]),
            "Page Number: ": result["source_documents"][0].metadata["page"]})
