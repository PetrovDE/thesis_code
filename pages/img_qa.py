import os
import tempfile
import streamlit as st

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

if st.sidebar.button("Home"):
    st.switch_page("app.py")
st.title("Ответы на вопросы по изображению")
st.sidebar.title("Обработка изображений")

upload_files = st.sidebar.file_uploader("Загрузить файлы", accept_multiple_files=True)

llm = ChatOllama(model="llama3", temperature=0)

docs = []


def generate_answer(docs, prompt):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OllamaEmbeddings())
    retriever = vectorstore.as_retriever()

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

    qa_system_prompt = """You are an assistant for extractive summarization. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Generate answer in Russian language and dont translate code. \

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

    chat_history_for_chain = ChatMessageHistory()

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: chat_history_for_chain,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    answer = conversational_rag_chain.invoke(
        {"input": prompt},
        config={"configurable": {"session_id": "img_qa"}},
    )["answer"]
    return answer


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
            stream = generate_answer(docs, prompt)
            st.markdown(stream)

        st.session_state.messages.append({"role": "assistant", "content": stream})