import os
import json
import pickle
import faiss
import numpy as np
import requests
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# Load model and index
embedder = SentenceTransformer("all-MiniLM-L12-v2")
index = faiss.read_index("embeddings/chunks.index")
with open("embeddings/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

model = genai.GenerativeModel("gemini-1.5-flash-8b")

app = Flask(__name__)

def get_top_chunks(query, k=3):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(query_embedding, k)
    return [metadata[i] for i in indices[0]]

def answer_question(query):
    top_chunks = get_top_chunks(query)
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

@app.route("/hackrx/run", methods=["POST"])
def hackrx_run():
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return jsonify({"error": "Missing or invalid Authorization header"}), 401

    try:
        data = request.get_json()
        questions = data.get("questions", [])
        answers = [answer_question(q) for q in questions]
        return jsonify({"answers": answers})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
