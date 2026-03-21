from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

os.environ['HF_HOME'] = 'D:/huggingface_cache'
load_dotenv()

embeddings = HuggingFaceEmbeddings(model = "sentence-transformers/all-MiniLM-L6-v2")

document = [
    "delhi is the capital of india",
    "mumbia is the capital of maharashtra",
    "paris is the capital of france"
]

vector = embeddings.embed_documents(document)

print(str(vector))