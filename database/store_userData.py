from sqlalchemy import create_engine
from dotenv import load_dotenv
import pandas as pd
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

data = pd.read_csv('assets/csv/user_transactions.csv')
data.to_sql('history_table', engine, if_exists="replace", index=False)
