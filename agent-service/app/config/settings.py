import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    APP_NAME: str = "AI Support Agent Service"
    HUGGINGFACE_MODEL_ID: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    HUGGINGFACEHUB_API_TOKEN: str

    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

settings = Settings()