import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import os
from pdf_loader import CustomPyPDFDirectoryLoader
from dotenv import load_dotenv
from memory_profiler import memory_usage
from langchain.prompts import PromptTemplate

load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')
chatbot_data_path = os.getenv('CHATBOT_DATA_DIR')
chroma_dir = os.getenv('CHROMA_DIR')
persist_directory = chroma_dir


@st.cache_data(experimental_allow_widgets=True)
def my_func():
    print("Executing function my_func()")
    prompt_template = """Use the following pieces of context to answer the question at the end. Try to sense the meaning of the question. If the answer is not available in the context, respond with "No context available." Do not hallucinate or use any external information. Make the answer meaningful and human understandable but don't write to much try to be short and concise.

    {context}

    Question: {question}
    Answer:"""

    PROMPT = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])

    st.title("Document Chatbot")
    st.write("This is a chatbot that can answer questions about a document.")
    query = st.text_input("Enter your query here")
    print(f"Query: {query}")
    embeddings = OpenAIEmbeddings()
    if os.path.exists(persist_directory):
        try:
            print("Loading from Existing Embeddings")
            st.info('Loading from Existing Embeddings', icon="ℹ")
            docsearch = Chroma(persist_directory=persist_directory,
                               embedding_function=embeddings)
            if query:
                results = docsearch.similarity_search_with_score(query)
                for result in results:
                    if result[1] > 0.5:
                        st.error(
                            "No relevant documents found, please try again with a college related query")
                        return
            print("Processing it with OpenAI")
            chain_type_kwargs = {"prompt": PROMPT}
            qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(
            ), chain_type="stuff", retriever=docsearch.as_retriever(search_kwargs={"k": 2}), chain_type_kwargs=chain_type_kwargs, return_source_documents=True)
        except Exception as e:
            st.error(
                f"Error searching from existing Embeddings, please wait: {e}")
            raise
    else:
        try:
            loader = CustomPyPDFDirectoryLoader(chatbot_data_path)
            print("Loading Documents")
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=50)
            texts = text_splitter.split_documents(documents)
            print("Splitting Documents")
            st.info(
                'Loading and preparing all the documents. This may take a few moments...', icon="ℹ")
        except Exception as e:
            raise ValueError(
                "Error loading documents. Please check that the data path is correct and that the documents are in the correct format.")

        try:
            print("Creating New Embeddings")
            st.info('Creating new Embeddings!', icon="ℹ")
            docsearch = Chroma.from_documents(
                documents=texts, embedding=embeddings, persist_directory=persist_directory)
            docsearch.persist()
            if query:
                results = docsearch.similarity_search_with_score(query)
                for result in results:
                    if result[1] > 0.5:
                        st.error(
                            "No relevant documents found, please try again with a college related query")
                        return
            print("Processing it with OpenAI")
            chain_type_kwargs = {"prompt": PROMPT}
            qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(
            ), chain_type="stuff", retriever=docsearch.as_retriever(search_kwargs={"k": 2}), chain_type_kwargs=chain_type_kwargs, return_source_documents=True)
            st.info('New embeddings created successfully!', icon="✅")
        except Exception as e:
            raise ValueError(
                "Error creating new embeddings. Please check that the embedding data and directory paths are correct.")

    if query:
        try:
            result = qa({"query": query})
            if result["result"] == "No context available.":
                st.warning("No context available.")
            else:
                st.success(result["result"])
                st.write("Source Link:")
                st.write(result["source_documents"][0].metadata["sourcelink"])
                st.write("Source Internal Document:")
                st.json({"Source: ": os.path.basename(result["source_documents"][0].metadata["source"]),
                        "Page Number: ": result["source_documents"][0].metadata["page"], })
        except Exception as e:
            st.error(f"Error getting answer: {e}")
            raise


if __name__ == "__main__":
    my_func()
    # calculate the total memory usage
    mem_usage = memory_usage()
    total_mem_usage = round(sum(mem_usage), 2)
    print(f"Total memory usage: {total_mem_usage} MB")
    st.info(f"Total memory usage: {total_mem_usage} MB")
