import uuid

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from ShikkhaBondhu.configs import EMBEDDING_MODEL_NAME, GENERATION_TOP_K, CHROMA_COLLECTION_NAME


def build_session_retriever(documents):
    if not documents:
        return None, None

    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    session_collection_name = f"{CHROMA_COLLECTION_NAME}_{uuid.uuid4().hex}"

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        collection_name=session_collection_name,
        collection_metadata={"hnsw:space": "cosine"},
    )
    dense_retriever = vector_store.as_retriever(search_kwargs={"k": GENERATION_TOP_K})

    bm25_retriever = BM25Retriever.from_documents(documents=documents)
    bm25_retriever.k = GENERATION_TOP_K

    retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[0.4, 0.6],
    )
    return retriever, vector_store


def delete_session_collection(vector_store):
    if vector_store is None:
        return
    try:
        vector_store.delete_collection()
    except Exception as e:
        print(f"[retrieval] failed to delete old collection (non-fatal): {e}")