from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from nomic import login,embed
from pinecone import Pinecone,ServerlessSpec
import numpy as np
from groq import Groq
import requests
from flask import Flask,jsonify,request
from flask_cors import CORS
app=Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
client = Groq(api_key='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')

# # Load the PDF
# pdfread = PdfReader("maeks.pdf")
PINECONE_API_KEY="pcsk_4B27To_tY2jeLoxqgm97GKUfwxMccU39ZsN3jcd2D8Lq7UjZhjwEyHerwKDc8hpeinqpe"
pc=Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("lang")

# # Extract text from PDF
# extracted_text = ""
# for page in pdfread.pages:
#     extracted_text += page.extract_text() + "\n"

# # Initialize text splitter
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=500,  # Define the chunk size
#     chunk_overlap=50  # Define overlap to maintain context
# )

# # Split text into chunks
# text_chunks = text_splitter.split_text(extracted_text)

# # Print the chunks
# # print(text_chunks)
# # for i, chunk in enumerate(text_chunks):
# #     print(f"Chunk {i+1}:\n{chunk}\n{'='*50}")
# api_key = 'nk-LeXriqiihZl6pT8TT4QhSB8JQVhmJBAznO6Y-EaaDX4'
# login(api_key)
# output = embed.text(
#     texts=text_chunks,
#     model='nomic-embed-text-v1.5',
#     task_type='search_document',
#     dimensionality=256
# )
# # print(output)

# embeddings = np.array(output['embeddings'])

# # Pair each embedding with a unique ID
# vectors_to_upsert = [
#     (str(i), embeddings[i], {"text": text_chunks[i]})
#     for i in range(len(embeddings))
# ]

# # Upsert vectors into the Pinecone index
# val=index.upsert(vectors=vectors_to_upsert)
def download_pdf(pdf_url, save_path="downloaded.pdf"):
    try:
        response = requests.get(pdf_url, stream=True)
        if response.status_code == 200:
            # Ensure the file is overwritten every time
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            print(f"PDF downloaded and saved as {save_path}")
            return save_path
        else:
            print("Failed to download PDF. HTTP Status Code:", response.status_code)
            return None
    except Exception as e:
        print("Error:", e)
        return None


# def download_pdf(pdf_url, save_path="downloaded.pdf"):
#     response = requests.get(pdf_url)
#     if response.status_code == 200:
#         with open(save_path, 'wb') as f:
#             f.write(response.content)
#         print("PDF downloaded successfully.")
#         return save_path
#     else:
#         print("Failed to download PDF.")
#         return None

def givepdf(mypdf):


# Load the PDF
    # local_pdf_path = download_pdf(mypdf)
    # if not local_pdf_path:
    #  return
    local_pdf_path = download_pdf(mypdf)
    if not local_pdf_path:
     return

    # Step 2: Load the downloaded PDF
    pdfread = PdfReader(local_pdf_path)

    # Step 2: Load the downloaded PDF
    # pdfread = pdf_document
    # PdfReader(local_pdf_path)
# pdfread = PdfReader("maeks.pdf")
    # PINECONE_API_KEY="pcsk_4B27To_tY2jeLoxqgm97GKUfwxMccU39ZsN3jcd2D8Lq7UjZhjwEyHerwKDc8hpeinqpe"
    # pc=Pinecone(api_key=PINECONE_API_KEY)
    # index = pc.Index("lang")

# Extract text from PDF
    extracted_text = ""
    for page in pdfread.pages:
        extracted_text += page.extract_text() + "\n"

# Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
           chunk_size=500,  # Define the chunk size
           chunk_overlap=50  # Define overlap to maintain context
            )

# Split text into chunks
    text_chunks = text_splitter.split_text(extracted_text)
    # index.delete(delete_all=True)
# https://cloud.appwrite.io/v1/storage/buckets/6786b6e90012a9d714fd/files/67b1d576000d6bcdc36c/view?project=6786b37a000e1a5e8c68&mode=admin

# Print the chunks
# print(text_chunks)
# for i, chunk in enumerate(text_chunks):
#     print(f"Chunk {i+1}:\n{chunk}\n{'='*50}")
    api_key = 'nk-LeXriqiihZl6pT8TT4QhSB8JQVhmJBAznO6Y-EaaDX4'
    login(api_key)
    output = embed.text(
      texts=text_chunks,
     model='nomic-embed-text-v1.5',
     task_type='search_document',
      dimensionality=256
          )
    embeddings = np.array(output['embeddings'])
# print(output)

     

# Pair each embedding with a unique ID
    vectors_to_upsert = [
      (str(i), embeddings[i], {"text": text_chunks[i]})
      for i in range(len(embeddings))
       ]

# Upsert vectors into the Pinecone index
    val=index.upsert(vectors=vectors_to_upsert)



def query_rag_system(query):
    # Generate embedding for the query
    api_key = 'nk-LeXriqiihZl6pT8TT4QhSB8JQVhmJBAznO6Y-EaaDX4'
    login(api_key)
    query_embedding = embed.text(
        texts=[query],
        model='nomic-embed-text-v1.5',
        task_type='search_query',
        dimensionality=256
    )['embeddings'][0]
    

    # Query Pinecone for relevant documents
    search_results = index.query(
        vector=query_embedding,
        top_k=2,  # Number of top results to retrieve
        include_metadata=True
    )

    # Extract retrieved texts
    retrieved_texts = " ".join([result['metadata']['text'] for result in search_results['matches']])

    # Construct system prompt
    system_prompt = f"""
    Instructions:
    - You are an AI assistant specialized in analyzing and discussing data and deep fake-related queries. Your role is to:

     1. Analyze user queries related to:
     - Data analysis and interpretation
      -   Deep fake detection and awareness
     - Digital media authenticity

     2. When responding:
     - Only use verified and provided data sources
      - Clearly state when information is based on available data
     - Highlight limitations or uncertainties in the analysis
     - Maintain ethical guidelines regarding deep fake discussions

   

     4. Ethical Guidelines:
     - Don't provide instructions for creating deep fakes
      - Promote responsible use of technology
     - Emphasize the importance of digital media literacy
     - Warn about potential misuse and consequences

      Please assist the user based on their query while adhering to these guidelines and using only the provided data sources.

     Remember to:
      ✓ Stay within ethical boundaries
      ✓ Be clear about data sources
      ✓ Maintain objectivity
      ✓ Promote responsible practices
     - Use only the extracted context to generate responses; do not infer or fabricate information.
     - Provide concise and relevant answers based on the context.

     give respone in 5 to 6 lines only and give user solutions about reporting to goverment if user sends normal question then replay as professtionally
    Context: {retrieved_texts}
    """

    # Generate response using Groq's LLM
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": query}],
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )

    # Print the response
    s=""
    for chunk in completion:
       content = chunk.choices[0].delta.content  # Extract content
       if content:  # Check if content is not None
        s += content
    return s

# Example query

@app.route('/pdf', methods=['POST'])
def set_data():
    d=request.get_json()
    # print(d)
    givepdf(d['url'])
    return jsonify({"message": f"document uploded  {d['url']}"})

@app.route('/question',methods=['POST'])
def ask_ques():
    d=request.get_json()
    ans=query_rag_system(d['ques'])
    print(ans)
    return jsonify({"responce":f"{ans}"})
if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000,debug=True)
# query = "give me questions"
# givepdf("https://cloud.appwrite.io/v1/storage/buckets/6786b6e90012a9d714fd/files/67b1d576000d6bcdc36c/view?project=6786b37a000e1a5e8c68&mode=admin")
# query_rag_system(query)


# query = "cgpa"
# query_embedding = embed.text(
#     texts=[query],
#     model='nomic-embed-text-v1.5',
#     task_type='search_query',
#     dimensionality=256
# )['embeddings'][0]
# search_results = index.query(
#     vector=query_embedding,
#     top_k=2,  # Number of top results to retrieve
#     include_metadata=True
# )

# for result in search_results['matches']:
#     print(f"Score: {result['score']}")
#     print(f"Text: {result['metadata']['text']}\n")
