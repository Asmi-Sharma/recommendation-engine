from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import pandas as pd
from tqdm.auto import tqdm
from dotenv import load_dotenv
import os

load_dotenv()

data = pd.read_csv('assets/csv/user_transactions.csv')
product_data = data.drop(['user_id', 'user_name'], axis=1)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pinecone = Pinecone(api_key=PINECONE_API_KEY)
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
INDEX_NAME = 'recommendation-engine-synthetic'
# check if index name already exist in pinecone database and if it does delete it
if INDEX_NAME in [index.name for index in pinecone.list_indexes()]:
    pinecone.delete_index(INDEX_NAME)

# now create a pinecone database with unique index
pinecone.create_index(name=INDEX_NAME,
                      dimension=model.get_sentence_embedding_dimension(),
                      metric='cosine',
                      spec=ServerlessSpec(cloud='aws', region='us-east-1'))

index = pinecone.Index(INDEX_NAME)

# upsert data in pinecone database
batch_size = 200  # to determine the size of data to pass at a time
vector_limit = 10000  # to determine the size of input values

df = product_data[:vector_limit]
records = df.to_dict(orient='records')

for i in tqdm(range(0, len(df), batch_size)):
    # find end of batch
    i_end = min(i+batch_size, len(df))
    # create IDs batch
    ids = [str(x) for x in range(i, i_end)]

    metadata = records[i: i_end]
    embedding = model.encode([record['product_name'] for record in records])
    index.upsert(vectors=zip(ids, embedding, metadata))
