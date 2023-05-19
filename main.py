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
import requests
import sqlite3
import time
from openai.error import OpenAIError

load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')
chatbot_data_path = os.getenv('CHATBOT_DATA_DIR')
chroma_dir = os.getenv('CHROMA_DIR')
persist_directory = chroma_dir
user_data_dir = os.getenv('USER_DATA_DIR')
user_db = os.getenv('USER_DB_NAME')
ip_url = os.getenv('IP_URL')


if not os.path.exists(user_data_dir):
    os.makedirs(user_data_dir)

db_path = os.path.join(user_data_dir, user_db)

REQUEST_THRESHOLD = 7
TIME_LIMIT_SEC = 30

conn = sqlite3.connect(db_path, check_same_thread=False)
c = conn.cursor()
create_table_query = '''
CREATE TABLE IF NOT EXISTS requests (
    ip TEXT NOT NULL,
    last_request_time INTEGER NOT NULL,
    request_count INTEGER DEFAULT 0
);
'''
c.execute(create_table_query)
conn.commit()


def process_request(ip):
    current_time = int(time.time())
    c.execute(
        'SELECT request_count, last_request_time FROM requests WHERE ip=?', (ip,))
    row = c.fetchone()
    if row is None:
        request_count = 1
        last_request_time = current_time
        c.execute('INSERT INTO requests VALUES (?, ?, ?)',
                  (ip, last_request_time, request_count))
        conn.commit()
    else:
        request_count, last_request_time = row
        if current_time - last_request_time > TIME_LIMIT_SEC:
            request_count = 1
            last_request_time = current_time
        else:
            request_count += 1
            if request_count > REQUEST_THRESHOLD:
                st.error(
                    f'Error: Too many requests, please wait {TIME_LIMIT_SEC} seconds before trying again.')
                return False
            last_request_time = max(last_request_time, current_time)

        c.execute('UPDATE requests SET request_count=?, last_request_time=? WHERE ip=?',
                  (request_count, last_request_time, ip))
        conn.commit()
    return True


@st.cache_data(experimental_allow_widgets=True, show_spinner=False, ttl=3600)
def get_client_ip():
    return requests.get(ip_url).text


@st.cache_data(experimental_allow_widgets=True, show_spinner=False, ttl=3600)
def my_func():
    try:
        client_ip = get_client_ip()
    except requests.RequestException as e:
        st.error(f"Error getting client IP: {e}")
        return

    prompt_template = """Use the following pieces of context to answer the question at the end. Try to sense the meaning of the question. The context is sometimes unstructured, so make a structured and rephrased version of the context in your mind, and try to find the answer of the question from that context. if you sense or find multiple questions in one question, try to break it down and find the answer one by one then respond with the complete answer. If the answer is not available in the context, respond with "No context available." Do not hallucinate or use any external information. Make the answer meaningful and in sentences and human understandable but don't write too much, try to be short and concise. Also sound human and polite.

    {context}

    Question: {question}
    Answer:"""

    PROMPT = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])

    st.title("Ask your question 🤖:")
    st.write("This can answer your questions related to Chandernagore College, Please be specific with your questions.")
    query = st.text_input("Enter your query here")
    embeddings = OpenAIEmbeddings()
    if os.path.exists(persist_directory):
        try:
            docsearch = Chroma(persist_directory=persist_directory,
                               embedding_function=embeddings)
            if query:
                results = docsearch.similarity_search_with_score(query)
                for result in results:
                    if result[1] > 0.5:
                        st.error(
                            "❌ No relevant documents found, please try again with a college related query")
                        return
            chain_type_kwargs = {"prompt": PROMPT}
            qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(
            ), chain_type="stuff", retriever=docsearch.as_retriever(search_kwargs={"k": 2}), chain_type_kwargs=chain_type_kwargs, return_source_documents=True)
        except OpenAIError as e:
            st.error(f"OpenAI API Error: {e}")
            return
        except Exception as e:
            st.error(
                f"Error searching from existing Embeddings, please wait: {e}")
            return
    else:
        try:
            loader = CustomPyPDFDirectoryLoader(chatbot_data_path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=50)
            texts = text_splitter.split_documents(documents)
            info_placeholder = st.empty()
            info_placeholder.info(
                'Loading and preparing all the documents. This may take a few moments...', icon="ℹ")
            time.sleep(1)
            info_placeholder.empty()
        except Exception as e:
            st.error(
                "Error loading documents. Please check that the data path is correct and that the documents are in the correct format.")
            return

        try:
            info_placeholder = st.empty()
            info_placeholder.info('Creating new Embeddings!', icon="ℹ")
            time.sleep(1)
            info_placeholder.empty()
            docsearch = Chroma.from_documents(
                documents=texts, embedding=embeddings, persist_directory=persist_directory)
            info_placeholder.info(
                'New embeddings created successfully!', icon="✅")
            time.sleep(1)
            info_placeholder.empty()
            docsearch.persist()
            if query:
                results = docsearch.similarity_search_with_score(query)
                for result in results:
                    if result[1] > 0.5:
                        st.error(
                            "❌ No relevant documents found, please try again with a college related query")
                        return
            chain_type_kwargs = {"prompt": PROMPT}
            qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(
            ), chain_type="stuff", retriever=docsearch.as_retriever(search_kwargs={"k": 2}), chain_type_kwargs=chain_type_kwargs, return_source_documents=True)
        except OpenAIError as e:
            st.error(f"OpenAI API Error: {e}")
            return
        except Exception as e:
            st.error(
                "Error creating new embeddings. Please check that the embedding data and directory paths are correct.")
            return

    if not query:
        st.error("❓ Please enter a query")
        return
    if query:
        if not process_request(client_ip):
            return
        try:
            result = qa({"query": query})
            if result["result"] == "No context available.":
                st.error("❌ No context available")
            else:
                st.success(result["result"])
        except OpenAIError as e:
            st.error(f"OpenAI API Error: {e}")
            return
        except Exception as e:
            st.error(f"Error getting answer: {e}")
            return


if __name__ == "__main__":
    try:
        my_func()
        mem_usage = memory_usage()
        total_mem_usage = round(sum(mem_usage), 2)
        st.info(f"Total memory usage: {total_mem_usage} MB")
    except Exception as e:
        st.error(f"An error occurred: {e}")
