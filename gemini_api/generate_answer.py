# gemini_api/generate_answer.py
import os
import google.generativeai as genai
from chromadb import Documents

def split_text(text: str):
    # Using str.split() for efficient splitting
    # This approach assumes double newlines indicate separation
    chunks = text.split('\n\n')
    
    # Using a generator to yield non-empty chunks
    return (chunk for chunk in chunks if chunk.strip())  # Use strip() to remove leading/trailing whitespace


def get_relevant_passage(query, db, n_results):
    """
    Queries the ChromaDB for the most relevant passages based on a query.
    """
    # Perform the query with n_results
    print(f"Querying for: {query}")
    result = db.query(query_texts=[query], n_results=n_results)
    passages = result['documents']

    # Print out the retrieved passages for inspection
    print("Retrieved Passages:")
    # for i, passage in enumerate(passages, 1):
        #print(f"Passage {i}:\n{passage}\n{'-'*40}")

    # Flatten and return the retrieved passages if found
    return passages if passages else ["No relevant passages found."]

def make_rag_prompt(query: str, relevant_passage: str):
    escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = ("""You are a helpful and informative bot that answers questions using text from the reference passage included below. \
    Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
    However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
    strike a friendly and conversational tone. \
    If the passage is irrelevant to the answer, you may ignore it.
    QUESTION: '{query}'
    PASSAGE: '{relevant_passage}'

    ANSWER:
    """).format(query=query, relevant_passage=escaped)
   
    return prompt

def generate_answer(prompt: str):
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("Gemini API Key not provided. Please provide GEMINI_API_KEY as an environment variable")
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-pro')
    answer = model.generate_content(prompt)
   
    return answer.text

def get_answer(query, db):
    # Retrieve the top 3 relevant text chunks
    relevant_text = get_relevant_passage(query, db, n_results=3)

    # Flatten the list if it’s nested and join it into a single string
    flattened_text = "".join([item for sublist in relevant_text for item in sublist if isinstance(item, str)])

    # Generate prompt and get answer
    prompt = make_rag_prompt(query, flattened_text)

    answer = generate_answer(prompt)

    return answer

