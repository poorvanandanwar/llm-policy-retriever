import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

def load_index_and_metadata(index_path="embeddings/chunks.index", meta_path="embeddings/metadata.pkl"):
    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

def encode_query(query, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    return model.encode([query])[0] 

def search_index(query_embedding, index, metadata, top_k=5):
    D, I = index.search(np.array([query_embedding]), k=top_k)
    results = [metadata[i] for i in I[0]]
    return results

def main():
    query = "46M, knee surgery, Pune, 3-month-old policy"
    index, metadata = load_index_and_metadata()
    query_embedding = encode_query(query)
    top_chunks = search_index(query_embedding, index, metadata, top_k=5)

    print("\n Top Matching Chunks:")
    for i, chunk in enumerate(top_chunks):
        print(f"\n--- Result {i+1} ---")
        print(f"Clause ID: {chunk.get('clause_id')}")
        print(f"Source Doc: {chunk.get('source_doc')}")
        print(f"Page No: {chunk.get('page_no')}")
        print(f"Text:\n{chunk.get('chunk_text')[:500]}...\n")

if __name__ == "__main__":
    main()
