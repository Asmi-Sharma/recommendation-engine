from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
pinecone = Pinecone(api_key=PINECONE_API_KEY)
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
INDEX_NAME = 'recommendation-engine-synthetic'
index = pinecone.Index(INDEX_NAME)


def run_query_vectordb(query):
    embedding = model.encode(query).tolist()
    results = index.query(top_k=10, vector=embedding,
                          include_metadata=True, include_values=False)
    res = []
    # arrange the result in desired format
    for result in results['matches']:
        res.append(result['metadata'])
    return res
