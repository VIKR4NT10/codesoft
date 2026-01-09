import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

def load_mysql_table(database: str, table: str) -> pd.DataFrame:
    engine = create_engine(
        f"mysql+pymysql://{os.getenv('MYSQL_USER')}:{os.getenv('MYSQL_PASSWORD')}"
        f"@{os.getenv('MYSQL_HOST')}:{os.getenv('MYSQL_PORT')}/{database}"
    )

    query = f"SELECT * FROM {table}"
    return pd.read_sql(query, engine)
