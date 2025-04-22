from openai import OpenAI
from google import genai
import os
import logging
import json
import importlib.util
from datetime import datetime
from abc import ABC, abstractmethod

# Configure logging
log_directory = os.getenv("LOG_DIR", "logs")
os.makedirs(log_directory, exist_ok=True)
log_file = os.path.join(
    log_directory, f"llm_calls_{datetime.now().strftime('%Y%m%d')}.log"
)

# Set up logger
logger = logging.getLogger("llm_logger")
logger.setLevel(logging.INFO)
logger.propagate = False  # Prevent propagation to root logger
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
logger.addHandler(file_handler)

# Simple cache configuration
cache_file = "llm_cache.json"


# Base class for all LLM providers
class LLMProvider(ABC):
    @abstractmethod
    def call(self, prompt: str) -> str:
        """Call the LLM provider with the given prompt and return the response"""
        pass

    @classmethod
    def is_available(cls) -> bool:
        """Check if the provider is available (dependencies installed, etc.)"""
        return True


# OpenAI provider implementation
class OpenAIProvider(LLMProvider):
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "your-api-key"),
        )
        self.model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    def call(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "text"},
            temperature=1,
            max_completion_tokens=32768,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            store=False,
        )
        return response.choices[0].message.content


# Google Gemini provider implementation
class GeminiProvider(LLMProvider):
    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", "your-api-key"))
        self.model = os.getenv("GEMINI_MODEL", "gemini-2.5-pro-exp-03-25")

    def call(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model, contents=[prompt]
        )
        return response.text

    @classmethod
    def is_available(cls) -> bool:
        return importlib.util.find_spec("google.genai") is not None


# Google Vertex AI Gemini provider implementation
class VertexGeminiProvider(LLMProvider):
    def __init__(self):
        self.client = genai.Client(
            vertexai=True,
            project=os.getenv("GEMINI_PROJECT_ID", "your-project-id"),
            location=os.getenv("GEMINI_LOCATION", "us-central1"),
        )
        self.model = os.getenv("GEMINI_MODEL", "gemini-2.5-pro-exp-03-25")

    def call(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model, contents=[prompt]
        )
        return response.text

    @classmethod
    def is_available(cls) -> bool:
        return (
            importlib.util.find_spec("google.genai") is not None
            and os.getenv("GEMINI_PROJECT_ID") is not None
        )


# Anthropic Claude provider implementation
class AnthropicProvider(LLMProvider):
    def __init__(self):
        from anthropic import Anthropic

        self.client = Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY", "your-api-key")
        )
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-3-7-sonnet-20250219")

    def call(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=21000,
            thinking={"type": "enabled", "budget_tokens": 20000},
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[1].text

    @classmethod
    def is_available(cls) -> bool:
        return importlib.util.find_spec("anthropic") is not None


# Factory function to get the appropriate LLM provider
def get_provider() -> LLMProvider:
    """
    Get the appropriate LLM provider based on environment variables.

    The provider is selected based on the LLM_PROVIDER environment variable.
    If not set, it will try to use the first available provider in the following order:
    - OpenAI
    - Google Gemini
    - Google Vertex AI Gemini
    - Anthropic Claude
    """
    provider_name = os.getenv("LLM_PROVIDER", "").lower()

    # Map of provider names to their implementations
    providers = {
        "openai": OpenAIProvider,
        "gemini": GeminiProvider,
        "vertex": VertexGeminiProvider,
        "claude": AnthropicProvider,
    }

    # If a specific provider is requested, try to use it
    if provider_name in providers:
        provider_cls = providers[provider_name]
        if provider_cls.is_available():
            return provider_cls()
        else:
            logger.warning(
                f"Requested provider {provider_name} is not available, falling back"
            )

    # Otherwise, try each provider in order of preference
    for name, provider_cls in providers.items():
        try:
            if provider_cls.is_available():
                logger.info(f"Using {name} LLM provider")
                return provider_cls()
        except Exception as e:
            logger.warning(f"Error initializing {name} provider: {e}")

    # If all else fails, use OpenAI as the default
    logger.warning("No available providers found, falling back to OpenAI")
    return OpenAIProvider()


# Main LLM calling function with caching
def call_llm(prompt: str, use_cache: bool = True) -> str:
    # Log the prompt
    logger.info(f"PROMPT: {prompt}")

    # Check cache if enabled
    if use_cache:
        # Load cache from disk
        cache = {}
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    cache = json.load(f)
            except:
                logger.warning("Failed to load cache, starting with empty cache")

        # Return from cache if exists
        if prompt in cache:
            logger.info(f"RESPONSE: {cache[prompt]}")
            return cache[prompt]

    # Get the appropriate provider
    try:
        provider = get_provider()
        response_text = provider.call(prompt)
    except Exception as e:
        logger.error(f"Error calling LLM: {e}")
        raise

    # Log the response
    logger.info(f"RESPONSE: {response_text}")

    # Update cache if enabled
    if use_cache:
        # Load cache again to avoid overwrites
        cache = {}
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    cache = json.load(f)
            except:
                pass

        # Add to cache and save
        cache[prompt] = response_text
        try:
            with open(cache_file, "w") as f:
                json.dump(cache, f)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    return response_text


if __name__ == "__main__":
    test_prompt = "Hello, how are you?"

    # First call - should hit the API
    print("Making call...")
    response1 = call_llm(test_prompt, use_cache=False)
    print(f"Response: {response1}")
