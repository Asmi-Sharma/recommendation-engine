import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()
host = "127.0.0.1"
dbname = os.getenv("dbname")
user = os.getenv("db_user")
password = os.getenv("db_password")
port = "5432"

# Create connection string
connection_string = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"

engine = create_engine(connection_string)


def run_query_postgresql(user_id):
    query = """
        SELECT * FROM public.history_table
        WHERE user_id LIKE %s
        OR user_id LIKE %s
        OR user_id LIKE %s
        OR user_id = %s
    """
    # Use a tuple for parameters
    params = (
        user_id,
        f"{user_id}%",
        f"%{user_id}%",
        f"%,{user_id}"
    )

    # Execute the query
    user_result = pd.read_sql(query, con=engine, params=params)
    user_result = user_result.to_dict(orient='records')

    return user_result
