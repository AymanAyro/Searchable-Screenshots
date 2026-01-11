"""Embedding service for text vectorization via Ollama using LangChain."""

from typing import Optional
import hashlib

from langchain_ollama import OllamaEmbeddings


class EmbeddingService:
    """Generate text embeddings using Ollama via LangChain."""
    
    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "mxbai-embed-large",
        timeout: float = 60.0,
        cache_enabled: bool = True,
    ):
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.cache_enabled = cache_enabled
        # Initialize LangChain OllamaEmbeddings
        # Note: New version handles timeout internally via httpx client
        self._embeddings = OllamaEmbeddings(
            base_url=self.ollama_url,
            model=self.model,
        )
        self._dimension: Optional[int] = None
        # Embedding cache: text hash -> embedding vector
        self._cache: dict[str, list[float]] = {}
    
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
        
        try:
            # LangChain embeddings.embed_query returns a list of floats
            embedding = self._embeddings.embed_query(cleaned_text)
            
            if embedding:
                self._dimension = len(embedding)
                # Cache the result if enabled
                if self.cache_enabled:
                    text_hash = self._get_text_hash(cleaned_text)
                    self._cache[text_hash] = embedding
                return embedding
            else:
                return None
        except Exception as e:
            print(f"Embedding service failed: {e}")
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
        
        try:
            # LangChain async embeddings
            embedding = await self._embeddings.aembed_query(cleaned_text)
            
            if embedding:
                self._dimension = len(embedding)
                # Cache the result if enabled
                if self.cache_enabled:
                    text_hash = self._get_text_hash(cleaned_text)
                    self._cache[text_hash] = embedding
            
            return embedding
        except Exception as e:
            print(f"Embedding API error: {e}")
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
            print(f"Embedding batch failed: {e}")
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
            print(f"Embedding batch async failed: {e}")
            return [None] * len(texts)
    
    @property
    def dimension(self) -> Optional[int]:
        """Get the embedding dimension (available after first embedding)."""
        return self._dimension
    
    def is_available(self) -> bool:
        """Check if the embedding service is available."""
        try:
            # Try to embed a test string
            test_embedding = self._embeddings.embed_query("test")
            return test_embedding is not None and len(test_embedding) > 0
        except Exception:
            return False
    
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
