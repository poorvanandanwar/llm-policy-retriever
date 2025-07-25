import faiss
import pickle
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sentence_transformers import SentenceTransformer

with open("metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

index = faiss.read_index("chunks.index")

embedder = SentenceTransformer('all-MiniLM-L6-v2')
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

MAX_INPUT_TOKENS = 512
MAX_NEW_TOKENS = 300

def split_text(text, max_tokens=MAX_INPUT_TOKENS):
    tokens = tokenizer.encode(text, truncation=False)
    for i in range(0, len(tokens), max_tokens):
        yield tokenizer.decode(tokens[i:i + max_tokens], skip_special_tokens=True)

def get_top_chunks(query, k=3):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(query_embedding, k)
    return [metadata[i] for i in indices[0]]

def generate_answer(query):
    top_chunks = get_top_chunks(query)
    print("\n🔍 Top Relevant Chunks:")
    for i, chunk in enumerate(top_chunks, 1):
        print(f"\nChunk {i}:\n{chunk['chunk_text'][:300]}...\n")


    all_outputs = []

    for chunk in top_chunks:
        prompt = f"Answer the following question based on the text:\n\n{chunk['chunk_text']}\n\nQuestion: {query}"
        for sub_prompt in split_text(prompt):
            output = pipe(sub_prompt, max_new_tokens=MAX_NEW_TOKENS)[0]['generated_text']
            all_outputs.append(output)

  
    final_answer = "\n---\n".join(all_outputs)
    return final_answer

if __name__ == "__main__":
    print("Document Retrieval Chat App with FLAN-T5 and Chunk Splitting")
    while True:
        query = input("Enter your question (or type 'exit'): ").strip()
        if query.lower() == "exit":
            break
        answer = generate_answer(query)
        print(f"\n Answer:\n{answer}")
