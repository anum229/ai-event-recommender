import json
from sentence_transformers import SentenceTransformer
import numpy as np
import re

# Load proposals
with open('approved_proposals.json', 'r', encoding='utf-8') as f:
    proposals = json.load(f)

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

def normalize_text(text):
    """Basic text cleaning: lowercase, remove extra spaces and linebreaks"""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

vectorized_proposals = []
skipped = 0

for i, proposal in enumerate(proposals):
    title = proposal.get('title')
    text = proposal.get('text')

    if not title or not text:
        print(f"⚠️  Skipping proposal at index {i} due to missing fields:")
        skipped += 1
        continue

    combined_text = normalize_text(title + " " + text)
    embedding = model.encode(combined_text, convert_to_numpy=True).tolist()

    vectorized_proposals.append({
        "title": title,
        "embedding": embedding
    })
    print(f"✅ Vectorized: {title}")

# Save output
with open('vectorized_proposals.json', 'w', encoding='utf-8') as f:
    json.dump(vectorized_proposals, f, indent=2)

print(f"\n✅ Vectorization completed. Saved {len(vectorized_proposals)} proposals.")
print(f"⚠️ Skipped {skipped} proposals due to issues.")