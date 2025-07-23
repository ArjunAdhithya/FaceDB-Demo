import os
from deepface import DeepFace
import chromadb
from chromadb.config import Settings

# Initialize ChromaDB client with cosine similarity
chroma_client = chromadb.PersistentClient(path="embeddings")

collection = chroma_client.get_or_create_collection(
    name="faces",
    metadata={"hnsw:space": "cosine"}
)

data_dir = "data"

added, skipped, failed = 0, 0, 0

# Get existing IDs to avoid duplicates
existing_ids = set(collection.get()["ids"])

for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.abspath(os.path.join(root, file))
            rel_id = os.path.relpath(img_path, data_dir)  # unique ID
            if rel_id in existing_ids:
                skipped += 1
                continue
            try:
                emb = DeepFace.represent(img_path, model_name="Facenet")
                if isinstance(emb, list) and emb:
                    embedding = emb[0]["embedding"]
                    person_name = os.path.basename(os.path.dirname(img_path))
                    collection.add(
                        ids=[rel_id],
                        embeddings=[embedding],
                        metadatas=[{"file_path": img_path, "person_name": person_name}]
                    )
                    added += 1
                    print(f"Added: {img_path}")
                else:
                    failed += 1
                    print(f"Face not detected: {img_path}")
            except Exception as e:
                failed += 1
                print(f"Error processing {img_path}: {e}")

print(f"\nIngestion Summary:")
print(f"  Added: {added}")
print(f"  Skipped (duplicates): {skipped}")
print(f"  Failed: {failed}")
