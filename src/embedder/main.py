import os
import json
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

def load_chunks_from_folder(folder_path):
    all_chunks = []
    for file in os.listdir(folder_path):
        if file.endswith("_chunks.json"):
            file_path = os.path.join(folder_path, file)
            with open(file_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
                all_chunks.extend(chunks)
    return all_chunks

def generate_embeddings(texts, model_name="all-MiniLM-L12-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings

def save_index_and_metadata(embeddings, metadata, output_dir="embeddings"):
    os.makedirs(output_dir, exist_ok=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    faiss.write_index(index, os.path.join(output_dir, "chunks.index"))

    with open(os.path.join(output_dir, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

    print(f"✅ Saved {len(metadata)} chunks to '{output_dir}/'")

def main():
    chunk_folder = "chunks"
    output_dir = "embeddings"

    print("📂 Loading chunk files...")
    chunks = load_chunks_from_folder(chunk_folder)
    print(f"🔢 Total chunks loaded: {len(chunks)}")

    texts = [chunk["chunk_text"] for chunk in chunks]
    print("✨ Generating embeddings using 'all-MiniLM-L12-v2'...")
    embeddings = generate_embeddings(texts)

    print("💾 Saving FAISS index and metadata...")
    save_index_and_metadata(embeddings, chunks, output_dir)

if __name__ == "__main__":
    main()

