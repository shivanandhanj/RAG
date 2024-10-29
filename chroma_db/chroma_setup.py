# chroma_db/chroma_setup.py
import chromadb
from chromadb import Documents
from typing import List
from gemini_api.embedding_function import GeminiEmbeddingFunction

def create_chroma_db(documents: List, path: str, name: str):
    """
    Creates or retrieves a Chroma database collection using the provided documents, path, and collection name.
    """
    chroma_client = chromadb.PersistentClient(path=path)
    
    # Check if the collection already exists
    if name in [coll.name for coll in chroma_client.list_collections()]:
        db = chroma_client.get_collection(name=name, embedding_function=GeminiEmbeddingFunction())
    else:
        db = chroma_client.create_collection(name=name, embedding_function=GeminiEmbeddingFunction())
        for i, d in enumerate(documents):
            db.add(documents=d, ids=str(i))

    print(db)
    
    return db, name

def load_chroma_collection(path: str, name: str):
    chroma_client = chromadb.PersistentClient(path=path)
    db = chroma_client.get_collection(name=name, embedding_function=GeminiEmbeddingFunction())
    return db
