from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import time
import os


load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


extracted_data = load_pdf_file(data='Data/')
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

index_name = "medical-vector"

# Initialize Pinecone with optional parameters
try:
    pc = Pinecone(
        api_key=os.environ.get("PINECONE_API_KEY"),
        proxy_url=None,  # Example optional parameter
        proxy_headers=None,  # Example optional parameter
        ssl_ca_certs=None,  # Example optional parameter
        ssl_verify=True,  # Example optional parameter, usually set to True
    )

    time.sleep(0.2)  # Optional sleep to ensure initialization completes

    # Check if the index exists
    indexes = pc.list_indexes()  # List of index names
    index_names = indexes.names()  # Get only the names of the indexes

    if index_name not in index_names:
        print(f'{index_name} does not exist')
        # Uncomment the following line to create the index of your choice
        # pc.create_index(
        #     name=index_name,
        #     dimension=384,
        #     metric="cosine",
        #     spec=ServerlessSpec(
        #         cloud="aws",
        #         region="us-east-1"
        #     )
        # )
    else:
        print(f'{index_name} exists.')

    # Connect to the existing index
    index = pc.Index(index_name)

except Exception as e:
    print(f"An error occurred while checking indexes: {e}")

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings, 
)
