from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from retrieve_result import retreival_result, result_after_retreival
from langchain.vectorstores import Pinecone as LangchainPinecone  # Using alias for LangChain Pinecone

from dotenv import load_dotenv
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = GROQ_API_KEY

embeddings = download_hugging_face_embeddings()

index_name = "medical-vector"

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = LangchainPinecone.from_existing_index(index_name, embeddings)

@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = str.lower(msg)
    print(input)
    docs = retreival_result(PINECONE_API_KEY,input,docsearch)
    # Process the documents and generate a response
    response = result_after_retreival(GROQ_API_KEY,input,docs)
    # Concatenate the response into a single string
    full_response = ''.join(response)
    print(full_response)
    # Handle general responses based on content
    if any(greeting in input for greeting in ["hi", "hello", "hey"]):
        return "Hello! How can I assist you today?"
    elif any(farewell in input for farewell in ["bye", "goodbye"]):
        return "Goodbye! Take care."
    elif "thanks" in input or "thank you" in input:
        return "You're welcome! Let me know if you have any other questions."
    elif full_response:  # If there is a relevant response
        return full_response
    else:  # Fallback response for unclear queries
        return "I'm sorry, I'm not sure about that."




if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8081, debug= True)
