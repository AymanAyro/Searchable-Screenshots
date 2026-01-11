"""Embedding service for text vectorization via Ollama using LangChain."""

from typing import Optional
import hashlib
import time

from langchain_ollama import OllamaEmbeddings

from ..core.logging import get_logger
from ..core.retry import retry_with_backoff, CircuitBreaker


class EmbeddingService:
    """Generate text embeddings using Ollama via LangChain."""
    
    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "mxbai-embed-large",
        timeout: float = 60.0,
        cache_enabled: bool = True,
        max_retries: int = 3,
        retry_backoff_factor: float = 2.0,
        retry_initial_delay: float = 1.0,
    ):
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.cache_enabled = cache_enabled
        self.max_retries = max_retries
        self.retry_backoff_factor = retry_backoff_factor
        self.retry_initial_delay = retry_initial_delay
        
        self.logger = get_logger(__name__)
        self.circuit_breaker = CircuitBreaker(failure_threshold=5, timeout=60.0)
        self._last_success_time: Optional[float] = None
        
        # Initialize LangChain OllamaEmbeddings
        # Note: New version handles timeout internally via httpx client
        self._embeddings = OllamaEmbeddings(
            base_url=self.ollama_url,
            model=self.model,
        )
        self._dimension: Optional[int] = None
        # Embedding cache: text hash -> embedding vector
        self._cache: dict[str, list[float]] = {}
        self.logger.info(f"Initialized embedding service with model '{model}'")
    
    def _get_text_hash(self, text: str) -> str:
        """Generate hash for text to use as cache key."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def embed(self, text: str, max_retries: int = 3) -> Optional[list[float]]:
        """Generate embedding vector for text.
        
        Args:
            text: Text to embed
            max_retries: Number of retry attempts for transient failures (LangChain handles retries)
            
        Returns:
            Embedding vector as list of floats, or None if failed
        """
        # Validate input - check for empty or whitespace-only text
        if not text:
            return None
        
        cleaned_text = text.strip()
        if not cleaned_text:
            return None
        
        # Check cache if enabled
        if self.cache_enabled:
            text_hash = self._get_text_hash(cleaned_text)
            if text_hash in self._cache:
                return self._cache[text_hash]
        
        @retry_with_backoff(
            max_retries=self.max_retries,
            initial_delay=self.retry_initial_delay,
            backoff_factor=self.retry_backoff_factor,
            exceptions=(Exception,),
            logger=self.logger,
        )
        def _embed_text():
            return self._embeddings.embed_query(cleaned_text)
        
        try:
            # Use circuit breaker for API calls
            embedding = self.circuit_breaker.call(_embed_text)
            
            if embedding:
                self._dimension = len(embedding)
                # Cache the result if enabled
                if self.cache_enabled:
                    text_hash = self._get_text_hash(cleaned_text)
                    self._cache[text_hash] = embedding
                self._last_success_time = time.time()
                return embedding
            else:
                return None
        except Exception as e:
            self.logger.error(f"Embedding service failed: {e}")
            return None
    
    async def embed_async(self, text: str) -> Optional[list[float]]:
        """Async version of embed()."""
        if not text or not text.strip():
            return None
        
        cleaned_text = text.strip()
        
        # Check cache if enabled
        if self.cache_enabled:
            text_hash = self._get_text_hash(cleaned_text)
            if text_hash in self._cache:
                return self._cache[text_hash]
        
        import asyncio
        
        try:
            # Retry logic for async calls
            for attempt in range(self.max_retries + 1):
                try:
                    embedding = await self._embeddings.aembed_query(cleaned_text)
                    if embedding:
                        self._dimension = len(embedding)
                        # Cache the result if enabled
                        if self.cache_enabled:
                            text_hash = self._get_text_hash(cleaned_text)
                            self._cache[text_hash] = embedding
                        self._last_success_time = time.time()
                    return embedding
                except Exception as e:
                    if attempt < self.max_retries:
                        delay = self.retry_initial_delay * (self.retry_backoff_factor ** attempt)
                        self.logger.warning(f"Embedding API call failed (attempt {attempt + 1}/{self.max_retries + 1}), retrying in {delay:.2f}s: {e}")
                        await asyncio.sleep(delay)
                    else:
                        raise
        except Exception as e:
            self.logger.error(f"Embedding API error: {e}")
            return None
    
    def embed_batch(self, texts: list[str]) -> list[Optional[list[float]]]:
        """Generate embeddings for multiple texts.
        
        Uses LangChain's embed_documents which may be more efficient.
        """
        try:
            # Filter out empty texts
            valid_texts = [t.strip() for t in texts if t and t.strip()]
            if not valid_texts:
                return [None] * len(texts)
            
            # LangChain embed_documents returns list of embeddings
            embeddings = self._embeddings.embed_documents(valid_texts)
            
            # Map back to original list with None for empty texts
            result = []
            valid_idx = 0
            for text in texts:
                if text and text.strip():
                    if valid_idx < len(embeddings):
                        embedding = embeddings[valid_idx]
                        if embedding:
                            self._dimension = len(embedding)
                        result.append(embedding)
                        valid_idx += 1
                    else:
                        result.append(None)
                else:
                    result.append(None)
            
            return result
        except Exception as e:
            self.logger.error(f"Embedding batch failed: {e}")
            return [None] * len(texts)
    
    async def embed_batch_async(self, texts: list[str]) -> list[Optional[list[float]]]:
        """Async version of embed_batch().
        
        Uses LangChain's async embed_documents for better performance.
        """
        try:
            # Filter out empty texts
            valid_texts = [t.strip() for t in texts if t and t.strip()]
            if not valid_texts:
                return [None] * len(texts)
            
            # LangChain async embed_documents
            embeddings = await self._embeddings.aembed_documents(valid_texts)
            
            # Map back to original list with None for empty texts
            result = []
            valid_idx = 0
            for text in texts:
                if text and text.strip():
                    if valid_idx < len(embeddings):
                        embedding = embeddings[valid_idx]
                        if embedding:
                            self._dimension = len(embedding)
                        result.append(embedding)
                        valid_idx += 1
                    else:
                        result.append(None)
                else:
                    result.append(None)
            
            return result
        except Exception as e:
            self.logger.error(f"Embedding batch async failed: {e}")
            return [None] * len(texts)
    
    @property
    def dimension(self) -> Optional[int]:
        """Get the embedding dimension (available after first embedding)."""
        return self._dimension
    
    def is_available(self) -> bool:
        """Check if the embedding service is available."""
        if self.circuit_breaker.is_open:
            return False
        try:
            # Try to embed a test string
            test_embedding = self._embeddings.embed_query("test")
            return test_embedding is not None and len(test_embedding) > 0
        except Exception:
            return False
    
    def get_health_status(self) -> dict:
        """Get detailed health status of the embedding service.
        
        Returns:
            Dictionary with health status information
        """
        is_available = self.is_available()
        return {
            "available": is_available,
            "model": self.model,
            "ollama_url": self.ollama_url,
            "circuit_state": self.circuit_breaker._state.value,
            "last_success_time": self._last_success_time,
            "failure_count": self.circuit_breaker._failure_count,
            "cache_size": len(self._cache),
        }
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()
    
    def get_cache_size(self) -> int:
        """Get the number of cached embeddings."""
        return len(self._cache)
    
    def close(self) -> None:
        """Close the embeddings client (no-op for LangChain, but kept for compatibility)."""
        self.clear_cache()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
