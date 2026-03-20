import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

db_url = os.environ["SUPABASE_DB_URI"]
engine = create_engine(db_url)

with engine.connect() as conn:
    result = conn.execute(text("select 1"))
    print("Connected:", result.scalar())