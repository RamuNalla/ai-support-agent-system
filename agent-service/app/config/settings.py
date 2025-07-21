import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

class Settings(BaseSettings):
    APP_NAME: str = "AI Support Agent Service"
    GEMINI_API_KEY: str 

    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

# Create a settings instance
settings = Settings()