"""
Gemini Client for Research Agent
"""

import google.generativeai as genai
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
import logging
from typing import List, Dict, Any, Optional
import time

logger = logging.getLogger(__name__)

class GeminiClient:
    """Client for Google Gemini API"""
    
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash", temperature: float = 0.3, max_tokens: int = 4000):
        """
        Initialize Gemini client
        
        Args:
            api_key: Google API key for Gemini
            model: Gemini model name
            temperature: Temperature for generation
            max_tokens: Maximum tokens for generation
        """
        self.api_key = api_key
        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Initialize the model
        try:
            self.model = genai.GenerativeModel(model)
            logger.info(f"Initialized Gemini model: {model}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {e}")
            raise
    
    def test_connection(self) -> bool:
        """Test connection to Gemini API"""
        try:
            # Simple test generation
            response = self.model.generate_content(
                "Hello, this is a test message. Please respond with 'Connection successful'.",
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=50
                )
            )
            
            if response and response.text:
                logger.info("Gemini API connection test successful")
                return True
            else:
                logger.error("Gemini API connection test failed - no response")
                return False
                
        except Exception as e:
            logger.error(f"Gemini API connection test failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        try:
            # Get available models to find info about current model
            models = genai.list_models()
            
            for model in models:
                if model.name.endswith(self.model_name):
                    return {
                        'name': model.name,
                        'display_name': model.display_name,
                        'description': model.description,
                        'supported_generation_methods': model.supported_generation_methods
                    }
            
            # Return basic info if model not found in list
            return {
                'name': self.model_name,
                'display_name': self.model_name,
                'description': f"Gemini model: {self.model_name}",
                'supported_generation_methods': ['generateContent']
            }
            
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {
                'name': self.model_name,
                'display_name': self.model_name,
                'description': f"Gemini model: {self.model_name}",
                'supported_generation_methods': ['generateContent']
            }
    
    def invoke(self, messages: List) -> AIMessage:
        """
        Invoke Gemini with messages
        
        Args:
            messages: List of messages (compatible with LangChain format)
            
        Returns:
            AIMessage with the response
        """
        try:
            # Convert messages to Gemini format
            prompt_text = self._convert_messages_to_text(messages)
            
            # Generate response
            response = self.model.generate_content(
                prompt_text,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens
                )
            )
            
            # Handle different response scenarios
            if response.candidates:
                candidate = response.candidates[0]
                
                # Check if the response was blocked by safety filters
                if candidate.finish_reason == 2:  # SAFETY
                    logger.warning("Response blocked by safety filters")
                    return AIMessage(content="I apologize, but I cannot provide a response to this query due to safety guidelines. Please try rephrasing your question.")
                elif candidate.finish_reason == 3:  # RECITATION
                    logger.warning("Response blocked due to recitation concerns")
                    return AIMessage(content="I cannot provide this response due to potential copyright concerns. Please try a different approach to your question.")
                elif candidate.finish_reason == 4:  # OTHER
                    logger.warning("Response blocked for other reasons")
                    return AIMessage(content="I encountered an issue generating this response. Please try rephrasing your question.")
                
                # Try to get the text content
                if hasattr(candidate.content, 'parts') and candidate.content.parts:
                    text_parts = []
                    for part in candidate.content.parts:
                        if hasattr(part, 'text') and part.text:
                            text_parts.append(part.text)
                    
                    if text_parts:
                        return AIMessage(content=''.join(text_parts))
                
                # Fallback: try the direct text accessor
                try:
                    if response.text:
                        return AIMessage(content=response.text)
                except:
                    pass
            
            logger.error("Empty or invalid response from Gemini")
            return AIMessage(content="I apologize, but I couldn't generate a response. Please try again with a different question.")
                
        except Exception as e:
            logger.error(f"Error invoking Gemini: {e}")
            return AIMessage(content=f"Error generating response: {str(e)}")
    
    def _convert_messages_to_text(self, messages: List) -> str:
        """Convert LangChain messages to text format for Gemini"""
        prompt_parts = []
        
        for message in messages:
            if hasattr(message, 'content'):
                content = message.content
            else:
                content = str(message)
            
            if isinstance(message, SystemMessage):
                prompt_parts.append(f"System: {content}")
            elif isinstance(message, HumanMessage):
                prompt_parts.append(f"Human: {content}")
            elif isinstance(message, AIMessage):
                prompt_parts.append(f"Assistant: {content}")
            else:
                # Handle string messages or other formats
                prompt_parts.append(content)
        
        return "\n\n".join(prompt_parts)
    
    async def ainvoke(self, messages: List) -> AIMessage:
        """Async version of invoke"""
        # For now, just call the sync version
        # In a real implementation, you might want to use asyncio.to_thread
        return self.invoke(messages)


# Re-export ChatPromptTemplate for compatibility
__all__ = ['GeminiClient', 'ChatPromptTemplate']
