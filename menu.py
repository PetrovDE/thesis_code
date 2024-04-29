import streamlit as st


def menu():
    st.sidebar.page_link("pages/ex_qa.py", label="Simple Question Answering")
    st.sidebar.page_link("pages/img_qa.py", label="Image Question Answering")
    st.sidebar.page_link("pages/doc_qa.py", label="Document Question Answering")
    st.sidebar.page_link("pages/doc_web_qa.py", label="Question Answering on Web pages")

