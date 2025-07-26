import faiss
import pickle
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# Load FAISS index and metadata
with open("embeddings/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

index = faiss.read_index("embeddings/chunks.index")
embedder = SentenceTransformer("all-MiniLM-L12-v2")

# Setup Gemini model
model = genai.GenerativeModel("gemini-1.5-flash-8b")

def get_top_chunks(query, k=3):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(query_embedding, k)
    return [metadata[i] for i in indices[0]]

def generate_answer(query):
    top_chunks = get_top_chunks(query)
    print("\n🔍 Top Relevant Chunks:")
    for i, chunk in enumerate(top_chunks, 1):
        print(f"\nChunk {i}:\n{chunk['chunk_text'][:300]}...\n")

    combined_context = "\n\n".join([c["chunk_text"] for c in top_chunks])
    prompt = f"""You are a helpful assistant designed to analyze insurance policy documents.
Based on the following text, answer the user's question in a clear and structured JSON format with the following fields:
- decision: (approved/rejected/depends)
- amount: (if any specific mentioned, otherwise "as per policy")
- justification: mention the specific clauses used.

--- POLICY EXCERPTS ---
{combined_context}

--- USER QUERY ---
{query}

Respond in JSON only."""

    response = model.generate_content(prompt)
    return response.text

if __name__ == "__main__":
    print("🔎 Gemini-Powered Document Query App")
    while True:
        query = input("Enter your question (or type 'exit'): ").strip()
        if query.lower() == "exit":
            break
        answer = generate_answer(query)
        print(f"\n📄 Answer:\n{answer}")

