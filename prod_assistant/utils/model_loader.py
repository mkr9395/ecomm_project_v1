import os
import sys
import json
from dotenv import load_dotenv
from prod_assistant.utils.config_loader import load_config
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from prod_assistant.logger import GLOBAL_LOGGER as log
from prod_assistant.exception.custom_exception import ProductAssistantException
import asyncio


class ApiKeyManager: # Providing safe access via .get()
    """
    This class centralizes API key management by loading secrets from a JSON-based environment variable (used in ECS or secret managers), falling back to individual environment variables when needed, validating required keys at startup, and providing safe access while preventing secret leakage through logs.
    """
    REQUIRED_KEYS = ["GROQ_API_KEY", "GOOGLE_API_KEY"]
    
    def __init__(self):
        self.api_keys = {} # internal dictionary storing loaded secrets, Keeps secrets in memory only, not global
        
        # Attempt 1: Load JSON-based secret
        raw = os.getenv("API_KEYS")
        # In ECS / Secrets Manager, you often inject one secret like API_KEYS='{"GROQ_API_KEY":"abc123","GOOGLE_API_KEY":"xyz456"}
        
        if raw:
            try:
                parsed = json.loads(raw) # Parses the JSON string, convert it to python dictionary
                if not isinstance(parsed, dict): # validate it
                    raise ValueError("API_KEYS is not valid JSON object")
                self.api_keys = parsed # Logging source, not values (important security practice).
                log.info("Loaded API_KEYS from ECS secret")
            except Exception as e:
                log.warning("Failed to parse API_KEYS as JSON", error=str(e))
                
        # Attempt 2: Fallback to individual env vars
        for key in self.REQUIRED_KEYS:
            if not self.api_keys.get(key):
                env_val = os.getenv(key)
                if env_val:
                    self.api_keys[key] = env_val
                    log.info(f"loaded {key} from individual env var")
                    
        # Final validation (fail fast)
        missing = [k for k in self.REQUIRED_KEYS if not self.api_keys.get(k)]
        if missing:
            log.error("Missing required API keys", missing_keys=missing)
            raise ProductAssistantException("Missing API keys", sys) #Raise hard error
        
        log.info("API keys loaded", keys={k: v[:6] + "..." for k, v in self.api_keys.items()})
        
    def get(self, key: str) -> str: # Encapsulated access: Prevents direct dictionary access
        val = self.api_keys.get(key)
        if not val:
            raise KeyError(f"API key for {key} is missing")
        return val

class ModelLoader:
    
    """
    Loads embedding models and LLMs based on config and environment.
    """
    
    def __init__(self):
        """
        Reads environment variable ENV
        Defaults to "local" if not set
        Converts to lowercase
        Checks if it is NOT production
        """
        if os.getenv("ENV", "local").lower() != "production": 
            load_dotenv() # Loads variables from .env file into os.environ
            log.info("Running in LOCAL mode : .env loaded")
        else:
            log.info("Running in PRODUCTION mode")
            
        self.api_key_mgr = ApiKeyManager() # Loads API keys from: API_KEYS JSON secret (ECS / Secrets Manager)
        self.config = load_config()
        log.info("YAML config loaded", config_keys = list(self.config.keys()))
    
    def load_embeddings(self):
        """
        Load and return embedding model from Google Generative AI.
        """
        try:
            model_name = self.config["embedding_model"]["model_name"]
            log.info("Loading embedding model", model = model_name)
            
            # Patch: Ensure an event loop exists for gRPC aio
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                asyncio.set_event_loop(asyncio.new_event_loop())
                
            return GoogleGenerativeAIEmbeddings(
                model = model_name,
                google_api_key = self.api_key_mgr.get("GOOGLE_API_KEY") # type : ignore
            )
        except Exception as e:
            log.error("Error loading embedding model", error=str(e))
            raise ProductAssistantException("Failed to load embedding model", sys)
        
    def load_llm(self):
        """
        Load and return the configured LLM model.
        """
        llm_block = self.config["llm"]
        provider_key = os.getenv("LLM_PROVIDER", "google")
        
        if provider_key not in llm_block:
            log.error("LLM provider not found in config", provider = provider_key)
            raise ValueError(f"LLM provider '{provider_key}' not found in config")
        
        llm_config = llm_block[provider_key]
        provider = llm_config.get("provider")
        model_name = llm_config.get("model_name")
        temperature = llm_config.get("temperature", 0.2)
        max_tokens = llm_config.get("max_tokens", 2048)
        log.info("loading llm", provider = provider, model = model_name)
        
        if provider == "google":
            return ChatGoogleGenerativeAI(
                model = model_name,
                google_api_key = self.api_key_mgr.get("GOOGLE_API_KEY"),
                temperature = temperature,
                max_output_tokens = max_tokens
            )
        
        elif provider == "groq":
            return ChatGroq(
                model = model_name,
                api_key = self.api_key_mgr.get("GROQ_API_KEY"), # type: ignore
                temperature = temperature
            )
            
        # elif provider == "openai":
        #     return ChatOpenAI(
        #         model=model_name,
        #         api_key=self.api_key_mgr.get("OPENAI_API_KEY"),
        #         temperature=temperature,
        #         max_tokens=max_tokens
        #     )
        else:
            log.error("Unsopported LLM provider", provider = provider)
            raise ValueError(f"Unsupported LLM provider : {provider}")

if __name__ == "__main__":
    loader = ModelLoader()
    
    # TEst Embedding
    embeddings = loader.load_embeddings()
    print(f"Embedding Model Loaded : {embeddings}")
    result = embeddings.embed_query("hello, how are you?")
    print(f"Embedding results : {result}")
    
    # Test LLM
    llm = loader.load_llm()
    print(f"LLM loaded : {llm}")
    result = llm.invoke("hello, how are you?")
    print(f"LLM result : {result.content}")