import os
import json
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

def load_chunks(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_and_save_embeddings(chunks, model_name="all-mpnet-base-v2", output_dir="embeddings"):
    texts = [chunk["chunk_text"] for chunk in chunks]
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True)


    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype(np.float32))

    os.makedirs(output_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(output_dir, "chunks.index"))
    with open(os.path.join(output_dir, "metadata.pkl"), "wb") as f:
        pickle.dump(chunks, f)

    print(f"Saved {len(chunks)} normalized embeddings to {output_dir}/")

if __name__ == "__main__":
    chunks = load_chunks("../../chunks/all_policy_chunks.json")
    generate_and_save_embeddings(chunks)
