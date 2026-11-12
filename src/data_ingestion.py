import pandas as pd
import re
from sqlalchemy import create_engine, text
from config.config import settings

import ipdb


class DataIngestion:
    def __init__(self):
        self.settings = settings

    def connect(self):
        engine = create_engine(self.settings.database_url, pool_pre_ping=True)
        return engine

    def load_data(self):
        schema = self.settings.DB_SCHEMA
        table = self.settings.DB_TABLE

        for identifier in (schema, table):
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", identifier):
                raise ValueError(f"Invalid SQL identifier {identifier}")

        # ipdb.set_trace()

        query = text(f'SELECT * FROM "{schema}"."{table}"')
        engine = self.connect()

        try:
            with engine.connect() as connection:
                df = pd.read_sql(query, connection)
        finally:
            engine.dispose()

        if df.empty:
            raise ValueError("No records found in postgresql")
        return df


    def run(self):
        print("Starting Data Ingestion...")
        print(f"Database: {self.settings.DB_NAME}")
        print(f"Table: {self.settings.DB_TABLE}")

        df = self.load_data()

        print("Data Ingestion is successful")
        print(f"Dataset shape: {df.shape}")
        print("\nSample")
        print(df.head())

        return df



    
if __name__ == "__main__":

    data_ingestion = DataIngestion()
    df = data_ingestion.run()