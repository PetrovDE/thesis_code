from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser


class LLM:
    def __init__(self, model_type="Ollama", model="llama3", temperature=0.2):
        self.model_type = model_type
        self.model = model
        self.temperature = temperature

    def __llm(self):
        return ChatOllama(model=self.model, temperature=self.temperature)

    def generate_answer(self, docs, prompt, session="llm"):
        llm = self.__llm()
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

        qa_system_prompt = """You are an Russian assistant for question-answering and summarization tasks. \
        Use the following pieces of retrieved context to answer the question. \
        If you don't know the answer, just say that you don't know. \
        Generate answer in Russian language and dont translate code. \
        Use fifty sentences maximum and keep the answer concise.\

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
            config={"configurable": {"session_id": session}},
        )
        return answer['answer']
