import os
import json
import pickle
import tempfile
import requests
import pdfplumber
import faiss
import numpy as np
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Load Gemini API key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# Initialize model and embedder
model = genai.GenerativeModel("gemini-1.5-flash-8b")
embedder = SentenceTransformer("all-MiniLM-L12-v2")

# Load static index and metadata
index = faiss.read_index("embeddings/chunks.index")
with open("embeddings/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

app = Flask(__name__)

# --- FAISS Retrieval for Known Docs ---
def get_top_chunks_static(query, k=3):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(query_embedding, k)
    return [metadata[i] for i in indices[0]]

# --- PDF Parsing + Embedding for Dynamic Docs ---
def extract_chunks_from_pdf_url(pdf_url, chunk_size=200):
    response = requests.get(pdf_url)
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name

    chunks = []
    with pdfplumber.open(tmp_path) as pdf:
        full_text = "\n".join([page.extract_text() or "" for page in pdf.pages])
    words = full_text.split()

    for i in range(0, len(words), chunk_size):
        chunk_text = " ".join(words[i:i + chunk_size])
        chunks.append({"chunk_text": chunk_text})

    return chunks

def get_top_chunks_dynamic(query, dynamic_chunks, k=3):
    texts = [chunk["chunk_text"] for chunk in dynamic_chunks]
    embeddings = embedder.encode(texts)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    query_embedding = embedder.encode([query])
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype(np.float32))

    distances, indices = index.search(query_embedding, k)
    return [dynamic_chunks[i] for i in indices[0]]

# --- Prompt Answering ---
def generate_answer(query, top_chunks):
    combined_context = "\n\n".join([c["chunk_text"] for c in top_chunks])
    prompt = f"""You are a helpful assistant designed to extract direct answers from insurance policy documents.
Given the policy excerpts below, answer the user's question clearly and concisely in **one sentence**. Do not include justifications, explanations, or formatting — just the answer.

--- POLICY EXCERPTS ---
{combined_context}

--- USER QUESTION ---
{query}

Answer:"""
    response = model.generate_content(prompt)
    return response.text.strip()

# --- Main Endpoint ---
@app.route("/hackrx/run", methods=["POST"])
def hackrx_run():
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return jsonify({"error": "Missing or invalid Authorization header"}), 401

    try:
        data = request.get_json()
        pdf_url = data.get("documents")  # optional
        questions = data.get("questions", [])

        if pdf_url:
            dynamic_chunks = extract_chunks_from_pdf_url(pdf_url)
            answers = [
                generate_answer(q, get_top_chunks_dynamic(q, dynamic_chunks))
                for q in questions
            ]
        else:
            answers = [
                generate_answer(q, get_top_chunks_static(q))
                for q in questions
            ]

        return jsonify({"answers": answers})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Server Run ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

