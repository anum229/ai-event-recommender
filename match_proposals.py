import json
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load vectorized proposals
try:
    with open('vectorized_proposals.json', 'r', encoding='utf-8') as f:
        proposals = json.load(f)
    logging.info(f"Loaded {len(proposals)} proposals from vectorized_proposals.json")
except Exception as e:
    logging.error("Failed to load proposals:", exc_info=True)
    exit(1)

# Load embedding model
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    logging.info("Loaded embedding model successfully")
except Exception as e:
    logging.error("Failed to load sentence transformer model:", exc_info=True)
    exit(1)

# Step 1: Get input from user
theme = input("Enter your project theme: ").strip()
tags = input("Enter tags (comma-separated): ").strip()

# Step 2: Prepare user query
combined_input = theme + "\n" + ", ".join([t.strip() for t in tags.split(",") if t.strip()])
logging.info(f"User input vectorizing for theme + tags: {combined_input}")

# Step 3: Vectorize user input
user_embedding = model.encode(combined_input, convert_to_numpy=True)

# Step 4: Compare with existing embeddings
results = []
threshold = 0.65  # Adjustable threshold
logging.info(f"Using similarity threshold: {threshold}")

for proposal in proposals:
    title = proposal['title']
    embedding = np.array(proposal['embedding'])

    similarity = cosine_similarity([user_embedding], [embedding])[0][0]

    logging.debug(f"Compared with '{title}' — Similarity: {similarity:.4f}")
    if similarity >= threshold:
        results.append((title, round(similarity, 4)))

# Step 5: Sort by similarity
results.sort(key=lambda x: x[1], reverse=True)

# Step 6: Display results
if results:
    print(f"\n🔍 Found {len(results)} relevant proposal(s):")
    for i, (title, score) in enumerate(results, 1):
        print(f"{i}. {title} — Similarity: {score}")
else:
    print("\n❌ No relevant proposals found.")