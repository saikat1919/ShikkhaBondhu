from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from configs import EMBEDDING_MODEL_NAME, GENERATION_TOP_K, CHROMA_COLLECTION_NAME
from dotenv import load_dotenv

load_dotenv()

def build_session_retriever(documents):
    if not documents:
        return None

    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        collection_name=CHROMA_COLLECTION_NAME,
        collection_metadata={"hnsw:space": "cosine"},
    )
    dense_retriever = vector_store.as_retriever(search_kwargs={"k": GENERATION_TOP_K})

    bm25_retriever = BM25Retriever.from_documents(documents=documents)
    bm25_retriever.k = GENERATION_TOP_K

    retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[0.4, 0.6],
    )
    return retriever