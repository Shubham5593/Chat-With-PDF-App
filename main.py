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
from utils import *



if 'responses' not in st.session_state:
    st.session_state['responses'] = ["Welcome! First Upload PDF & start chat with me.."]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

st.header("Q&A Chatbot From PDF")

pdf_doc = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", type="pdf", accept_multiple_files=False)

if pdf_doc != None:
    if st.button("Submit & Process"):
        with st.spinner("Processing..."):
            if pdf_doc is not None:
                folder_path = "User_Uploaded_PDF"  # Replace with your desired folder path
                os.makedirs(folder_path, exist_ok=True)  # Create the folder if it doesn't exist
                file_path = os.path.join(folder_path, pdf_doc.name)

                # Save the uploaded file to the specified path
                with open(file_path, "wb") as f:
                    f.write(pdf_doc.getbuffer())

                # Fetching PDF path from Local
                pdf_name = pdf_doc.name
                pdf_path = os.path.join(r"D:\shubham\Assignments\QA_Chatbot_From_PDF\User_Uploaded_PDF",pdf_name)

                get_load_and_split_pdf(pdf_path,embeddings)

        st.success("File Processing Completed")
        
# container for chat history
response_container = st.container()
# container for text box    
textcontainer = st.container()

with textcontainer:
    querry = st.text_input("Query: ", key="input")
    if querry :
        # Loading Vectorstores from local
        new_db = FAISS.load_local("faiss_index", embeddings)
        # simillarity search for user querry to extract required chunk text
        docs = new_db.similarity_search(querry)

        # Invoking chain
        response = chain(
            {"input_documents":docs, "question": querry}
            ,return_only_outputs=True)
        final_text = response["output_text"]
        final_text = final_text.replace("*","-")

        st.session_state.requests.append(querry)
        st.session_state.responses.append(final_text)

with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i],key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')

