import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import tempfile
import os

st.title("🤖 Chat con tu PDF")

openai_api_key = st.text_input("🔑 Ingresa tu OpenAI API Key", type="password")

pdf_file = st.file_uploader("📄 Sube un archivo PDF", type="pdf")

if openai_api_key and pdf_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(pdf_file.read())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    pages = loader.load_and_split()

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(pages, embeddings)

    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(openai_api_key=openai_api_key),
        retriever=vectorstore.as_retriever()
    )

    pregunta = st.text_input("❓ Preguntá algo sobre el PDF")

    if pregunta:
        respuesta = qa.run(pregunta)
        st.markdown(f"**Respuesta:** {respuesta}")
