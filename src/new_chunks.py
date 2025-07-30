import json

# Load existing chunks
with open("chunks/all_policy_chunks.json", "r", encoding="utf-8") as f:
    existing_chunks = json.load(f)

# Load new chunks from newly chunked file
with open("chunks/arogya_chunks.json", "r", encoding="utf-8") as f:
    new_chunks = json.load(f)

# Merge and save
all_chunks = existing_chunks + new_chunks

with open("all_policy_chunks.json", "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2, ensure_ascii=False)
