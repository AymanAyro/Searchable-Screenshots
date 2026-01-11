"""Optional reranker service for improving search result quality.

Note: Reranker requires sentence-transformers which depends on torch.
If torch is not installed, reranker functionality will be disabled.
"""

from typing import Optional


class RerankerService:
    """Rerank search results using cross-encoder model.
    
    Requires sentence-transformers (which requires torch) to be installed.
    If not available, reranking will be disabled.
    """
    
    def __init__(self, model_name: str = "mixedbread-ai/mxbai-rerank-large-v1"):
        """Initialize the reranker.
        
        Args:
            model_name: HuggingFace model name for reranking
        """
        self.model_name = model_name
        self._model = None
        self._available = False
        self._check_availability()
    
    def _check_availability(self):
        """Check if sentence-transformers is available."""
        try:
            import sentence_transformers
            self._available = True
        except ImportError:
            self._available = False
            print("Warning: sentence-transformers not available. Reranker disabled.")
            print("To enable reranker, install: uv add sentence-transformers")
    
    def _load_model(self):
        """Lazy load the model to avoid startup overhead."""
        if not self._available:
            raise RuntimeError("Reranker not available: sentence-transformers not installed")
        
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except ImportError:
                raise RuntimeError("sentence-transformers not available. Install with: uv add sentence-transformers")
    
    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: Optional[int] = None,
    ) -> list[tuple[int, float]]:
        """Rerank documents based on relevance to query.
        
        Args:
            query: Search query
            documents: List of document texts to rerank
            top_k: Return only top K results (all if None)
            
        Returns:
            List of (original_index, score) tuples, sorted by score descending
        """
        if not documents:
            return []
        
        if not self._available:
            # Return original order if reranker not available
            return [(i, 1.0) for i in range(len(documents))]
        
        self._load_model()
        
        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]
        
        # Get scores
        scores = self._model.predict(pairs)
        
        # Create indexed scores and sort
        indexed_scores = [(i, float(score)) for i, score in enumerate(scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        if top_k is not None:
            indexed_scores = indexed_scores[:top_k]
        
        return indexed_scores
    
    def rerank_with_ids(
        self,
        query: str,
        items: list[tuple[int, str]],
        top_k: Optional[int] = None,
    ) -> list[tuple[int, float]]:
        """Rerank items that have IDs.
        
        Args:
            query: Search query
            items: List of (id, text) tuples
            top_k: Return only top K results
            
        Returns:
            List of (id, score) tuples, sorted by score descending
        """
        if not items:
            return []
        
        ids, texts = zip(*items)
        reranked = self.rerank(query, list(texts), top_k)
        
        return [(ids[idx], score) for idx, score in reranked]
    
    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._model is not None
    
    @property
    def is_available(self) -> bool:
        """Check if reranker is available (sentence-transformers installed)."""
        return self._available