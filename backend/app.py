from flask import Flask, request, jsonify
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from rag_utils import retrieve, rag

rag_api = Flask("NovaCrest API")

# Load the embedding model and the saved FAISS index once, at startup
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
print(f"Vectors stored: {vectorstore.index.ntotal}")


def rag_settings(data):
    """Read the question and generation settings from the request, with defaults."""
    return {
        "query": data["query"],
        "k": data.get("k", 2),
        "model_name": data.get("model_name", "gpt-4o-mini"),
        "temperature": data.get("temperature", 0.0),
        "top_p": data.get("top_p", 1.0),
        "max_tokens": data.get("max_tokens", 512),
    }


@rag_api.get("/")
def home():
    return "Welcome to the NovaCrest RAG API!"


@rag_api.post("/v1/relevant_chunks")
def relevant_chunks():
    data = request.get_json()
    query, k = data["query"], data.get("k", 2)
    chunks = retrieve(query=query, k=k, vectorstore=vectorstore)
    return jsonify({"query": query, "k": k, "relevant_chunks": chunks})


@rag_api.post("/v1/answer_with_relevant_chunks")
def answer_with_relevant_chunks():
    settings = rag_settings(request.get_json())
    answer, chunks = rag(**settings, vectorstore=vectorstore)
    return jsonify({**settings, "relevant_chunks": chunks, "answer": answer})


@rag_api.post("/v1/answer")
def answer():
    settings = rag_settings(request.get_json())
    generated_answer, _ = rag(**settings, vectorstore=vectorstore)
    return jsonify({"answer": generated_answer})


if __name__ == "__main__":
    rag_api.run(debug=True)
