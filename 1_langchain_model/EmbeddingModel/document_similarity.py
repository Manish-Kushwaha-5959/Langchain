from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

os.environ['HF_HOME'] = 'D:/huggingface_cache'

embedding = HuggingFaceEmbeddings(model = "sentence-transformers/all-MiniLM-L6-v2")

document = [
    "delhi is the capital of india",
    "mumbai is the capital of maharashtra",
    "kolkatta is the capital of west bengal",
    "paris the capital of france"
]

vector = embedding.embed_documents(document)

querry = "what is the capital of maharashtra"

querry_emb = embedding.embed_query(querry)

scores = cosine_similarity([querry_emb], vector)

print(querry)
print(document[np.argmax(scores)])