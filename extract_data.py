import os
import requests
import pymongo
from PyPDF2 import PdfReader
import json

# MongoDB connection
client = pymongo.MongoClient("mongodb+srv://anum:anum123@cluster0.hvtbqi1.mongodb.net/smart_fyp_portal?retryWrites=true&w=majority&appName=Cluster0")
db = client["smart_fyp_portal"]
collection = db["proposals"]

# Output list
approved_proposals = []
min_text_length = 300  # Minimum characters required in extracted proposal text

# Fetch and process each approved proposal
for i, proposal in enumerate(collection.find({"fypStatus": "Approved"})):
    title = proposal.get("projectTitle")
    pdf_url = proposal.get("pdfUrl")

    if not title or not pdf_url:
        print(f"⚠️ Skipping proposal {i} due to missing title or pdfUrl.")
        continue

    try:
        # Download PDF
        response = requests.get(pdf_url)
        pdf_path = "temp_proposal.pdf"
        with open(pdf_path, "wb") as f:
            f.write(response.content)

        # Extract text from PDF
        reader = PdfReader(pdf_path)
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"

        # Validate extracted text
        if len(full_text.strip()) < min_text_length:
            print(f"⚠️ Skipping '{title}' — too little text extracted ({len(full_text.strip())} chars).")
            continue

        # Append to list
        approved_proposals.append({
            "title": title.strip(),
            "text": full_text.strip()
        })

        print(f"✅ Extracted: {title} ({len(full_text.strip())} characters)")

    except Exception as e:
        print(f"❌ Error processing '{title}': {e}")

# Clean up
if os.path.exists("temp_proposal.pdf"):
    os.remove("temp_proposal.pdf")

# Save output to JSON
with open("approved_proposals.json", "w", encoding="utf-8") as f:
    json.dump(approved_proposals, f, ensure_ascii=False, indent=2)

print(f"\n✨ Done. {len(approved_proposals)} valid proposals saved to approved_proposals.json")