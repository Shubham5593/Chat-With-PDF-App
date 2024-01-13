# Importing All required libraries
import os

import pdf2image
import pypdf
from pdfminer import psparser
from pdfminer.utils import open_filename
import pikepdf
import streamlit as st
from streamlit_chat import message
import string

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai


from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)


from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.memory import ChatMessageHistory,ConversationBufferWindowMemory

from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Saving Google gemini pro LLM API key in Environment
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# Defining Prompt Template & LLM
prompt_template = """
    Answer the question  delimated in triple bracket as detailed as possible from the provided context given below delimated in triple bracket & using previous chat history of user, 
    make sure to provide all the details.
    If the answer is not in provided context answer from your pretrained knowledge, don't provide the wrong answer.

    Context:``` \n {context} \n ```

    {chat_history}
    Question: ``` \n{question}\n ```

    Answer:
    """

model = ChatGoogleGenerativeAI(model="gemini-pro",
                            temperature=0.9)

history = ChatMessageHistory.construct()
memory = ConversationBufferWindowMemory(llm=model,
                                        memory_key="chat_history",
                                        input_key="question",
                                        chat_memory=history,
                                        k=6, 
                                        return_messages=True)

# Defining Pretrained Embedding for converting text into vector
embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")

# Defining Chain To wrap LLM, Prompt_template & Memory
prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "chat_history", "question"])
chain = load_qa_chain(model,  memory=memory, chain_type="stuff", prompt=prompt)


def get_load_and_split_pdf(pdf_path,embeddings):
    # Loading PDF data Here with Unstructured PDF loader
    loader = UnstructuredPDFLoader(pdf_path)
    data = loader.load()

    # Splitting Loaded PDF data into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=700)
    chunks = text_splitter.split_documents(data)
    chunks = list(chunks)

    vector_store = FAISS.from_documents(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

