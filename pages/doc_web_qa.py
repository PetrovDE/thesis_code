import os
import tempfile
import streamlit as st
import bs4
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader


if st.sidebar.button("Home"):
    st.switch_page("app.py")
st.title("Ответы на вопросы по документу")
st.sidebar.title("Обработка документов")

upload_files = st.sidebar.file_uploader("Загрузить файлы", accept_multiple_files=True)


llm = ChatOllama(model="llama3")

docs = []

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

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OllamaEmbeddings())
retriever = vectorstore.as_retriever()

### Contextualize question ###
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

### Answer question ###
qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Generate answer in russian language and dont translate code. \
Use three sentences maximum and keep the answer concise.\

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

with st.form(key='chat_form', clear_on_submit=False):
    user_input = st.text_input(
        "Вопрос:",
        placeholder="Напишите вопрос по документу",
        key='input'
    )
    submit_button = st.form_submit_button(label='Отправить')

    if submit_button and user_input:
        with st.spinner("Генерируем ответ ***"):
            output = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": "abs1"}},
            )["answer"]
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        st.markdown(output)

