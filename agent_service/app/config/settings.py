import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()                           # Load variables from the .env file
class Settings(BaseSettings):           # Declares the API KEY is a string and required
    APP_NAME: str = "AI Support Agent Service"
    GEMINI_API_KEY: str 

    FAISS_INDEX_PATH: str = "agent_service/faiss_index.bin"         # expected path for the FAISS index file.

    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

settings = Settings()