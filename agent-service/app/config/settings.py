import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()                              # Load key-value pairs from .env file (that contains LLM API keys)
class Settings(BaseSettings):              # Settings loaded from environment variables
    GEMINI_API_KEY: str
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')          # It won't raise an error if there are extra variables in the .env file

settings = Settings()