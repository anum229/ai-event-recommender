from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import logging
import re

# Setup Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
logging.info("✅ Sentence embedding model loaded successfully.")

# Load vectorized proposals
with open('vectorized_proposals.json', 'r', encoding='utf-8') as f:
    proposals = json.load(f)
logging.info(f"✅ Loaded {len(proposals)} proposals from vectorized_proposals.json")

# Helper: Normalize input text
def normalize(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Main route
@app.route('/suggest-project-name', methods=['POST'])
def suggest_project_name():
    data = request.get_json()
    theme = data.get('theme', '').strip()
    tags = data.get('tags', '').strip()

    logging.info(f"📥 Received project suggestion request. Theme: '{theme}', Tags: '{tags}'")

    if not theme:
        return jsonify({'error': 'Theme is required'}), 400

    try:
        input_text = normalize(theme + " " + tags)
        input_vector = model.encode(input_text, convert_to_numpy=True).reshape(1, -1)

        scored = []
        threshold = 0.35

        for item in proposals:
            title = item.get("title")
            vector = item.get("embedding")

            if not vector:
                logging.warning(f"⚠️ Skipping proposal '{title}' due to missing embedding.")
                continue

            proposal_vector = np.array(vector).reshape(1, -1)
            similarity = cosine_similarity(input_vector, proposal_vector)[0][0]
            similarity = round(similarity, 3)

            if similarity >= threshold:
                logging.info(f"✅ Matched: '{title}' → Similarity: {similarity} ✅")
                scored.append((title, similarity))
            else:
                logging.info(f"❌ Skipped: '{title}' → Similarity: {similarity} ❌")

        if not scored:
            logging.info("❌ No relevant project suggestions found.")
            return jsonify({'suggested_project_names': []})

        # Sort and return top 3
        scored.sort(key=lambda x: x[1], reverse=True)
        suggestions = [{'projectTitle': title} for title, _ in scored[:3]]
        logging.info(f"🎯 Final Suggestions: {[s['projectTitle'] for s in suggestions]}")

        return jsonify({'suggested_project_names': suggestions})

    except Exception as e:
        logging.exception("🔥 Internal error during project suggestion.")
        return jsonify({'error': 'Internal server error'}), 500

# Run server
if __name__ == '__main__':
    logging.info("🚀 API server is starting on port 5001")
    app.run(port=5001)