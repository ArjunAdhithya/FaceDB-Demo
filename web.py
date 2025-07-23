import os
import streamlit as st
import chromadb
from deepface import DeepFace
from PIL import Image

# Ensure embeddings folder exists
if not os.path.exists("embeddings"):
    os.makedirs("embeddings")

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="embeddings")
collection = chroma_client.get_or_create_collection("faces")

st.set_page_config(page_title="FaceSearch Demo", layout="wide")
st.title("FaceSearch Demo")

# Upload query image
st.header("Upload Image")
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# Controls for search
top_n = st.slider("Number of top matches", 1, 10, 5)
threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.3)

if uploaded_file:
    query_img_path = "query_temp.jpg"
    with open(query_img_path, "wb") as f:
        f.write(uploaded_file.read())

    query_img = Image.open(query_img_path)
    st.image(query_img, caption="Uploaded Image", width=300)

    try:
        query_embedding = DeepFace.represent(query_img_path, model_name="Facenet")[0]["embedding"]
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_n,
            include=["metadatas", "distances"]
        )

        st.header("Top Matches")
        if results["metadatas"][0]:
            match_cols = st.columns(5)
            shown = 0
            for idx, (metadata, score) in enumerate(zip(results["metadatas"][0], results["distances"][0])):
                if score <= threshold:  # Only show matches within the threshold
                    if metadata and "file_path" in metadata and os.path.exists(metadata["file_path"]):
                        img = Image.open(metadata["file_path"])
                        match_cols[shown % 5].image(
                            img, caption=f"Score: {score:.3f}", use_column_width=True
                        )
                        shown += 1
            if shown == 0:
                st.warning("No matches found within the threshold.")
        else:
            st.warning("No matches found in the database.")
    except Exception as e:
        st.error(f"Error processing uploaded image: {e}")

    os.remove(query_img_path)

# Show gallery of database images at the bottom
st.header("Database Gallery")
all_items = collection.get()
if all_items["metadatas"]:
    st.write("All stored face images:")
    cols = st.columns(5)
    for idx, metadata in enumerate(all_items["metadatas"]):
        if metadata and "file_path" in metadata and os.path.exists(metadata["file_path"]):
            img = Image.open(metadata["file_path"])
            cols[idx % 5].image(img, caption=os.path.basename(all_items["ids"][idx]), use_column_width=True)
else:
    st.info("No images found in the database")
