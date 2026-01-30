# app.py
import streamlit as st
import pickle
import numpy as np
import torch
from PIL import Image
import imagehash
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel
import google.generativeai as genai
import os
import pandas as pd

# ==============================
# CONFIG
# ==============================
import os
import streamlit as st
import google.generativeai as genai

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY not configured")
    st.stop()

genai.configure(api_key=api_key)


VECTOR_DB_PATH = "vector_database.pkl"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 5
CLIP_WEIGHT = 0.7
PHASH_WEIGHT = 0.3

# ==============================
# LOAD MODELS (CACHED)
# ==============================
@st.cache_resource
def load_resources():
    clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32"
    ).to(DEVICE)

    clip_processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32"
    )

    with open(VECTOR_DB_PATH, "rb") as f:
        vector_db = pickle.load(f)

    return clip_model, clip_processor, vector_db


clip_model, clip_processor, vector_db = load_resources()

# ==============================
# EMBEDDING + HASH
# ==============================
def get_clip_embedding(img: Image.Image):
    inputs = clip_processor(images=img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)
    emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
    return emb.cpu().numpy()

def phash_similarity(a, b):
    ha = imagehash.hex_to_hash(a)
    hb = imagehash.hex_to_hash(b)
    return 1 - (ha - hb) / ha.hash.size

# ==============================
# HYBRID SEARCH (TOP-K)
# ==============================
def hybrid_search(img, category, vector_db, top_k=5):
    items = vector_db.get(category, [])
    if not items:
        return []

    user_vec = get_clip_embedding(img)
    db_vecs = np.array([i["vector"] for i in items])

    clip_scores = cosine_similarity(user_vec, db_vecs)[0]
    clip_scores = (clip_scores + 1) / 2

    user_phash = str(imagehash.phash(img))
    phash_scores = np.array([
        phash_similarity(user_phash, i["phash"]) for i in items
    ])

    final_scores = CLIP_WEIGHT * clip_scores + PHASH_WEIGHT * phash_scores
    top_idx = np.argsort(final_scores)[::-1][:top_k]

    return [{
        "path": items[i]["path"],
        "final": float(final_scores[i]),
        "clip": float(clip_scores[i]),
        "phash": float(phash_scores[i])
    } for i in top_idx]

# ==============================
# STREAMLIT UI
# ==============================
st.set_page_config(layout="wide")
st.title("⚡ Retail Image Search (Paths Only, No SAM)")

uploaded_file = st.file_uploader(
    "Upload product image", type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    user_img = Image.open(uploaded_file).convert("RGB")
    st.image(user_img, caption="Input Image")

    if st.button("Identify & Find"):
        with st.spinner("Classifying & searching..."):

            # 1️⃣ GEMINI CATEGORY CLASSIFICATION
            gemini = genai.GenerativeModel("gemini-2.5-flash")

            prompt = f"""
            Identify the clothing category.
            Choices: {', '.join(vector_db.keys())}
            Return ONLY the word.
            """

            category = gemini.generate_content(
                [prompt, user_img]
            ).text.strip()

            if category not in vector_db:
                st.error(f"Unknown category returned: {category}")
                st.stop()

            st.success(f"Detected category: {category}")

            # 2️⃣ HYBRID SEARCH (NO SEGMENTATION)
            results = hybrid_search(
                user_img, category, vector_db, TOP_K
            )

            if not results:
                st.warning("No matches found.")
                st.stop()

            st.subheader("Top-5 Similar Results (Paths Only)")

            df = pd.DataFrame(results)
            st.dataframe(df)

            st.markdown("### Result Paths")
            for r in results:
                st.markdown(
                    f"- `{r['path']}`  \n"
                    f"  Final: **{r['final']:.3f}**, "
                    f"CLIP: {r['clip']:.3f}, "
                    f"pHash: {r['phash']:.3f}"
                )

            st.info(
                "✅ SAM removed | ✅ No DB images loaded | "
                "✅ GitHub & Streamlit Community Cloud friendly"
            )
