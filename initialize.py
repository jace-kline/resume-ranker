from langchain_openai import OpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

def get_model(provider: str):
    provider = provider.upper()
    llm = None
    if provider == "OLLAMA":
        ollama_model = os.getenv("OLLAMA_MODEL", "gemma2:9b")
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        llm = OllamaLLM(model=ollama_model, base_url=ollama_base_url)
    elif provider == "ANTHROPIC":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        llm = ChatAnthropic(api_key=api_key, model_name="claude-3-5-sonnet-20240620", max_tokens=2048)
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        llm = OpenAI(api_key=api_key, max_tokens=2048)

    return llm

def initialize_llm_from_env():
    # Initialize LLM based on environment variables
    provider = os.getenv("MODEL_PROVIDER", "ANTHROPIC").upper()
    return get_model(provider)
