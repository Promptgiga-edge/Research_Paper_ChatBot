import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class Config:
    """Configuration class for the Research Chatbot"""
    
    # GEMINI Configuration
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "")
    
    # SerpAPI Configuration for Google Scholar
    SERPAPI_KEY: str = os.getenv("SERPAPI_KEY", "")
    
    # Application Configuration
    MAX_RESULTS: int = int(os.getenv("MAX_RESULTS", "10"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "4000"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.3"))
    
    # Vector Database Configuration
    VECTOR_DB_PATH: str = os.getenv("VECTOR_DB_PATH", "./chroma_db")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    # Cache Configuration
    CACHE_DIR: str = os.getenv("CACHE_DIR", "./cache")
    CACHE_EXPIRY_HOURS: int = int(os.getenv("CACHE_EXPIRY_HOURS", "24"))
    
    # Streamlit Configuration
    APP_TITLE: str = "Research Paper Chatbot"
    APP_DESCRIPTION: str = "AI-powered research assistant for academic papers"
    
    def validate(self) -> bool:
        """Validate required configuration"""
        if not self.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is required")
        return True

# Global configuration instance
config = Config()
config.validate()