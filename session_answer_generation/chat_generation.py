from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from configs import GENERATION_MODEL_NAME

from dotenv import load_dotenv

load_dotenv()


CONTEXTUALIZE_SYSTEM_PROMPT = """Given a chat history and the latest user question, \
which might reference context in the chat history, formulate a standalone \
question which can be understood without the chat history. Do NOT answer the \
question, just reformulate it if needed, and otherwise return it as is."""

contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", CONTEXTUALIZE_SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("user", "{input}"),
])

SYSTEM_PROMPT = """
You are AskMe, a question-answering assistant that answers strictly using the provided context. Follow these rules:
1. Only use information present in the context. Do not use outside knowledge.
2. If the context does not contain enough information to answer the question, say so explicitly instead of guessing.
3. Be concise and direct. Do not repeat the question back.
4. If multiple people appear in the context, only use facts explicitly tied to the person named in the question. Never merge or borrow details from a different person's excerpt, even if the excerpts look similar.
5. You may use the chat history to understand what the user is referring to and to avoid repeating yourself, but the FACTS in your answer must still come only from the context below.

Context:
{context}
"""

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


def build_rag_chain(retriever):
    llm = HuggingFaceEndpoint(
        repo_id=GENERATION_MODEL_NAME,
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
    )
    model = ChatHuggingFace(llm=llm)

    history_aware_retriever = create_history_aware_retriever(
        model, retriever, contextualize_prompt
    )
    question_answering_chain = create_stuff_documents_chain(model, answer_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answering_chain)
    return rag_chain


def answer_question(rag_chain, query, chat_history):
    result = rag_chain.invoke({"input": query, "chat_history": chat_history})
    return result["answer"]