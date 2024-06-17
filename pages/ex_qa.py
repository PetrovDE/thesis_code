import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from models import LLM

if st.sidebar.button("Home"):
    st.switch_page("app.py")
st.title("Обобщение данных по документу")
st.sidebar.title("Загрузка документа")

upload_files = st.sidebar.file_uploader("Загрузить файлы", accept_multiple_files=True)

llm = LLM(temperature=0)
session = "ex_qa"
docs = []


def clear_cache():
    keys = list(st.session_state.keys())
    for key in keys:
        st.session_state.pop(key)


clear_cache()

if upload_files:
    text = []
    for file in upload_files:
        file_extentions = os.path.splitext(file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        loader = None
        if file_extentions == ".pdf":
            loader = PyPDFLoader(tmp_path)
        elif file_extentions == ".doc" or file_extentions == ".docx":
            loader = Docx2txtLoader(tmp_path)
        elif file_extentions == ".txt":
            loader = TextLoader(tmp_path)

        if loader:
            text.extend(loader.load())
            os.remove(tmp_path)

    docs = text

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            stream = llm.generate_answer(docs, prompt)
            st.markdown(stream)

        st.session_state.messages.append({"role": "assistant", "content": stream})
