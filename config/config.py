from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import URL
from dotenv import load_dotenv

BASE_URL = Path(__file__).resolve().parent.parent
# print(f"the base url is : {BASE_URL}")

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_URL / ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "insurance_fraud_detection"
    DB_USER: str = "postgres"
    DB_PASSWORD: str
    DB_SCHEMA: str = "public"
    DB_TABLE:str = "healthcare_fraud_detection"

    @property
    def database_url(self):
        return URL.create(drivername="postgresql+psycopg", username=self.DB_USER, password=self.DB_PASSWORD, host=self.DB_HOST, port=self.DB_PORT, database=self.DB_NAME)


settings = Settings()
# print(settings.DB_PASSWORD)