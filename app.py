# app.py
import os
import pickle
import numpy as np
import torch
import streamlit as st
import pandas as pd

from PIL import Image
import imagehash
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel

# Gemini (NEW SDK)
from google import genai
from google.genai import types

# ==============================
# CONFIG
# ==============================
VECTOR_DB_PATH = "vector_database.pkl"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 5
CLIP_WEIGHT = 0.7
PHASH_WEIGHT = 0.3

# ==============================
# GEMINI CLIENT (SECURE)
# ==============================
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY not set. Add it in Streamlit Secrets.")
    st.stop()

client = genai.Client(api_key=api_key)

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
import torch.nn.functional as F

import torch.nn.functional as F
import numpy as np

def get_clip_embedding(img: Image.Image):
    inputs = clip_processor(images=img, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)

    # ✅ STRICT, SAFE TYPE HANDLING
    if isinstance(emb, torch.Tensor):
        emb_t = emb.detach()
    elif isinstance(emb, np.ndarray):
        emb_t = torch.from_numpy(emb)
    else:
        raise TypeError(f"Unexpected embedding type: {type(emb)}")

    # ✅ L2 normalize
    emb_t = F.normalize(emb_t, p=2, dim=-1)

    return emb_t.cpu().numpy()



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
# st.title("⚡ Retail Image Search (Paths Only)")

uploaded_file = st.file_uploader(
    "Upload product image", type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    user_img = Image.open(uploaded_file).convert("RGB")
    st.image(user_img, caption="Uploaded Image")

    if st.button("Identify & Find"):
        with st.spinner("Classifying & searching..."):

            # 1️⃣ GEMINI CATEGORY CLASSIFICATION
            prompt = f"""
Identify the clothing category.
Choices: {', '.join(vector_db.keys())}
Return ONLY the word.
"""

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part(text=prompt),
                            types.Part(
                                inline_data=types.Blob(
                                    mime_type=uploaded_file.type,
                                    data=uploaded_file.getvalue()
                                )
                            )
                        ]
                    )
                ]
            )

            category = response.text.strip()

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
            st.dataframe(df, use_container_width=True)

            st.markdown("### Result Paths")
            for r in results:
                st.markdown(
                    f"- `{r['path']}`  \n"
                    f"  **Final:** {r['final']:.3f} | "
                    f"CLIP: {r['clip']:.3f} | "
                    f"pHash: {r['phash']:.3f}"
                )

            st.info(
                "✅ No SAM | ✅ No DB images | "
                "✅ Secure API | ✅ Streamlit Cloud ready"
            )
