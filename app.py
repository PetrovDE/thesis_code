import streamlit as st
from langchain_community.chat_models import ChatOllama
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.memory import ConversationBufferMemory

import tempfile
import os


def session_init():
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Приветствую! Задавайте вопросы!"]
    if 'past' not in st.session_state:
        st.session_state['past'] = ["Приветствую!"]


def conversation_chat(query, chain, history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]


def create_conversation(vector_store):
    memory = ConversationBufferMemory(memory_key="chat_history", return_message=True)
    local_llm = 'llama3'
    llm = ChatOllama(model=local_llm, temperature=0.01)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type='stuff',
        retriever=vector_store.as_retriever(),
        memory=memory,
    )
    return chain


def chat_history(chain):
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='chat_form', clear_on_submit=True):
            user_input = st.text_input(
                "Вопрос:",
                placeholder="Напишите вопрос по документу",
                key='input'
            )
            submit_button = st.form_submit_button(label='Отправить')

            if submit_button and user_input:
                with st.spinner("Генерируем ответ ***"):
                    output = conversation_chat(user_input, chain, st.session_state['history'])

                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(
                    st.session_state['past'][i],
                    is_user=True,
                    key=str(i) + '_user',
                    avatar_style='thumbs',
                )
                message(
                    st.session_state['generated'][i],
                    is_user=True,
                    key=str(i),
                    avatar_style='big-ears',
                )


def main():
    # init text splitter
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )

    session_init()
    st.title("Сервис семантического анализа документов")
    st.sidebar.title("Обработка документов")

    upload_files = st.sidebar.file_uploader("Загрузить файлы", accept_multiple_files=True)

    if upload_files:
        text = []
        for file in upload_files:
            file_extention = os.path.splitext(file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name

            loader = None
            if file_extention == ".pdf":
                loader = PyPDFLoader(tmp_path)
            elif file_extention == ".doc" or file_extention == ".docx":
                loader = Docx2txtLoader(tmp_path)
            elif file_extention == ".txt":
                loader = TextLoader(tmp_path)

            if loader:
                text.extend(loader.load())
                os.remove(tmp_path)

        text_chunks = text_splitter.split_documents(text)
        vectorstore = FAISS.from_documents(
            documents=text_chunks,
            embedding=GPT4AllEmbeddings(),
        )
        chain = create_conversation(vectorstore)

        chat_history(chain)


if __name__ == "__main__":
    main()
