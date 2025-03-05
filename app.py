from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import fitz  # PyMuPDF
from nomic import login, embed
from pinecone import Pinecone
import numpy as np
from groq import Groq
import requests
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize API Clients
client = Groq(api_key='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')
PINECONE_API_KEY = "pcsk_4B27To_tY2jeLoxqgm97GKUfwxMccU39ZsN3jcd2D8Lq7UjZhjwEyHerwKDc8hpeinqpe"
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("lang")

# Function to download and extract text from a PDF
def extract_text_from_pdf(pdf_url):
    response = requests.get(pdf_url)
    if response.status_code != 200:
        return None
    
    pdf_document = fitz.open(stream=response.content, filetype="pdf")
    extracted_text = "".join([page.get_text("text") for page in pdf_document])
    return extracted_text

# Function to process PDF and store embeddings
def process_pdf(pdf_url):
    extracted_text = extract_text_from_pdf(pdf_url)
    if not extracted_text:
        return "Error extracting text from PDF."
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_text(extracted_text)
    
    # Generate embeddings
    api_key = 'nk-LeXriqiihZl6pT8TT4QhSB8JQVhmJBAznO6Y-EaaDX4'
    login(api_key)
    output = embed.text(texts=text_chunks, model='nomic-embed-text-v1.5', task_type='search_document', dimensionality=256)
    embeddings = np.array(output['embeddings'])
    
    # Upsert embeddings into Pinecone
    vectors_to_upsert = [(str(i), embeddings[i], {"text": text_chunks[i]}) for i in range(len(embeddings))]
    index.upsert(vectors=vectors_to_upsert)
    return "PDF processed successfully."

# Function to query Pinecone and generate response
def query_rag_system(query):
    query_embedding = embed.text(texts=[query], model='nomic-embed-text-v1.5', task_type='search_query', dimensionality=256)['embeddings'][0]
    search_results = index.query(vector=query_embedding, top_k=2, include_metadata=True)
    
    retrieved_texts = " ".join([result['metadata']['text'] for result in search_results['matches']])
    system_prompt = f"""
    Instructions:
    - If the answer is not found in the provided PDF context, respond with 'I don't know.'
    - Use only the extracted context to generate responses; do not infer or fabricate information.
    - Provide concise and relevant answers based on the context.
    Context: {retrieved_texts}
    """
    
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": query}],
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=True,
    )
    
    response_text = "".join(chunk.choices[0].delta.content for chunk in completion if chunk.choices[0].delta.content)
    return response_text

# API Endpoints
@app.route('/pdf', methods=['POST'])
def upload_pdf():
    data = request.get_json()
    message = process_pdf(data['url'])
    return jsonify({"message": message})

@app.route('/question', methods=['POST'])
def ask_question():
    data = request.get_json()
    response = query_rag_system(data['ques'])
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
