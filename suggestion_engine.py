import json
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the model once
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    logging.info("Sentence embedding model loaded successfully.")
except Exception as e:
    logging.error("Failed to load sentence transformer model.", exc_info=True)
    model = None

# Load proposals once
try:
    with open("vectorized_proposals.json", "r", encoding="utf-8") as f:
        proposals = json.load(f)
    logging.info(f"Loaded {len(proposals)} proposals from vectorized_proposals.json")
except Exception as e:
    logging.error("Failed to load proposals file.", exc_info=True)
    proposals = []

def get_embedding(text):
    return model.encode(text, convert_to_numpy=True)

def get_top_project_suggestions(theme, tags, threshold=0.65):
    if not theme:
        logging.warning("Theme is missing in request.")
        return []

    query_text = theme + " " + ", ".join([t.strip() for t in tags.split(",") if t.strip()])
    logging.info(f"Vectorizing input: {query_text}")
    query_vec = get_embedding(query_text)

    results = []

    for prop in proposals:
        if "embedding" not in prop or "title" not in prop:
            continue

        try:
            proposal_embedding = np.array(prop["embedding"])
            similarity = cosine_similarity([query_vec], [proposal_embedding])[0][0]

            if similarity >= threshold:
                results.append((prop["title"], round(similarity, 4)))
                logging.info(f"Match found: '{prop['title']}' — Similarity: {similarity:.4f}")
            else:
                logging.debug(f"Irrelevant: '{prop['title']}' — Similarity: {similarity:.4f}")

        except Exception as e:
            logging.warning(f"Skipping proposal due to error: {e}")
            continue

    results.sort(key=lambda x: x[1], reverse=True)
    return results