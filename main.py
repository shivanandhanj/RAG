import os
import json
from dotenv import load_dotenv
from chroma_db.chroma_setup import create_chroma_db, load_chroma_collection
from gemini_api.generate_answer import split_text, get_answer

# Load environment variables from .env file
load_dotenv()

# Verify API key
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found. Please check your .env setup.")

# Load and process the JSON file
with open('recipe.json', 'r') as file:
    data = json.load(file)
    text_data = json.dumps(data, indent=21)

# Split text into chunks
chunked_text = split_text(text_data)
#print(f"Chunked Text: {chunked_text[:3]}...")  # Print first 3 chunks for verification

# Create and load Chroma DB
db, name = create_chroma_db(documents=chunked_text, path="D:/Full stack/chroma", name="rag_experiment")
db = load_chroma_collection(path="D:/Full stack/chroma", name="rag_experiment")
print("DB loaded:", db is not None)

# Run a query and generate an answer
query = "what is the Diet Type of  Salem Thatta Vadai?"
relevant_text = get_answer(query=query, db=db)
#print("Relevant Passage:", relevant_text)

# Final generated answer
print("Answer:", relevant_text)
