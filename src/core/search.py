"""Unified search interface with query routing and hybrid search.

This module uses LangChain services (embeddings, vector stores) while maintaining
the same interface for backward compatibility.
"""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from collections import defaultdict
import hashlib
import time

from ..core.database import Database, Screenshot
from ..services.vector_store import VectorStore, VectorSearchResult
from ..services.embedding import EmbeddingService
from ..services.sparse_embedding import SparseEmbeddingService
from ..services.reranker import RerankerService


@dataclass
class SearchResult:
    """A search result with screenshot data and relevance score."""
    screenshot: Screenshot
    score: float
    search_type: str  # 'fts', 'vector', 'hybrid', 'hybrid+rerank'
    
    @property
    def file_path(self) -> Path:
        return Path(self.screenshot.file_path)
    
    @property
    def id(self) -> int:
        return self.screenshot.id


class SearchEngine:
    """Unified search engine with FTS5, vector, and hybrid search."""
    
    def __init__(
        self,
        db: Database,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        sparse_embedding: Optional[SparseEmbeddingService] = None,
        reranker: Optional[RerankerService] = None,
        use_reranker: bool = False,
        hybrid_weight: float = 0.5,
        hybrid_normalization: str = "rrf",
        use_query_expansion: bool = False,
    ):
        """Initialize the search engine.
        
        Args:
            db: Database instance
            vector_store: Qdrant vector store
            embedding_service: Dense embedding service
            sparse_embedding: Optional BM25 sparse embedding service
            reranker: Optional reranker service
            use_reranker: Whether to apply reranking
            hybrid_weight: Weight for hybrid search (0.0 = sparse only, 1.0 = dense only)
            hybrid_normalization: Normalization method ("rrf", "minmax", "sigmoid")
            use_query_expansion: Whether to use query expansion
        """
        self.db = db
        self.vector_store = vector_store
        self.embedding = embedding_service
        self.sparse_embedding = sparse_embedding
        self.reranker = reranker
        # Only enable reranker if it's requested AND available
        self.use_reranker = (
            use_reranker 
            and reranker is not None 
            and (hasattr(reranker, 'is_available') and reranker.is_available)
        )
        self.hybrid_weight = max(0.0, min(1.0, hybrid_weight))  # Clamp to [0, 1]
        self.hybrid_normalization = hybrid_normalization
        self.use_query_expansion = use_query_expansion
        # Query cache: query_hash -> (results, timestamp)
        self._query_cache: dict[str, tuple[list[SearchResult], float]] = {}
        self._cache_ttl = 300  # 5 minutes default TTL
    
    def search(
        self,
        query: str,
        limit: int = 20,
        use_cache: bool = True,
    ) -> list[SearchResult]:
        """Search for screenshots matching query.
        
        Uses query routing:
        - Quoted queries ("like this") → FTS5 exact match
        - Unquoted queries → Hybrid search (sparse + dense) if available, otherwise vector search
        
        Args:
            query: Search query
            limit: Maximum number of results
            use_cache: Whether to use query result cache
            
        Returns:
            List of search results ordered by relevance
        """
        query = query.strip()
        
        if not query:
            return []
        
        # Check cache if enabled
        if use_cache:
            cache_key = self._get_query_cache_key(query, limit)
            if cache_key in self._query_cache:
                cached_results, timestamp = self._query_cache[cache_key]
                # Check if cache is still valid
                if time.time() - timestamp < self._cache_ttl:
                    return cached_results
                else:
                    # Remove expired cache entry
                    del self._query_cache[cache_key]
        
        # Query routing based on quotes
        if self._is_exact_query(query):
            # Strip quotes and do FTS search
            exact_query = query[1:-1]
            results = self.fts_search(exact_query, limit)
        else:
            # Use hybrid search if sparse embedding is available
            if self.sparse_embedding and self.sparse_embedding.is_fitted:
                results = self.hybrid_search(query, limit)
            else:
                results = self.vector_search(query, limit)
        
        # Cache results if enabled
        if use_cache:
            cache_key = self._get_query_cache_key(query, limit)
            self._query_cache[cache_key] = (results, time.time())
        
        return results
    
    def _get_query_cache_key(self, query: str, limit: int) -> str:
        """Generate cache key for query."""
        key_str = f"{query}:{limit}"
        return hashlib.sha256(key_str.encode('utf-8')).hexdigest()
    
    def clear_cache(self) -> None:
        """Clear the query result cache."""
        self._query_cache.clear()
    
    def get_cache_size(self) -> int:
        """Get the number of cached queries."""
        return len(self._query_cache)
    
    def fts_search(
        self,
        query: str,
        limit: int = 20,
    ) -> list[SearchResult]:
        """Perform FTS5 full-text search.
        
        Args:
            query: Search query (will be used in MATCH)
            limit: Maximum number of results
            
        Returns:
            List of search results
        """
        # Escape special FTS5 characters and format query
        fts_query = self._format_fts_query(query)
        
        screenshots = self.db.fts_search(fts_query, limit)
        
        # FTS5 results are already ranked by BM25
        return [
            SearchResult(
                screenshot=s,
                score=1.0 - (i * 0.01),  # Approximate score based on rank
                search_type="fts",
            )
            for i, s in enumerate(screenshots)
        ]
    
    def vector_search(
        self,
        query: str,
        limit: int = 20,
    ) -> list[SearchResult]:
        """Perform vector similarity search.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of search results ordered by similarity
        """
        # Generate query embedding
        query_vector = self.embedding.embed(query)
        if query_vector is None:
            return []
        
        # Search vector store
        # Request more results if reranking
        search_limit = limit * 3 if self.use_reranker else limit
        vector_results = self.vector_store.search(query_vector, search_limit)
        
        # Fetch full screenshot data for results
        results = []
        for vr in vector_results:
            screenshot = self.db.get_by_id(vr.id)
            if screenshot:
                results.append(SearchResult(
                    screenshot=screenshot,
                    score=vr.score,
                    search_type="vector",
                ))
        
        # Apply reranking if enabled
        if self.use_reranker and results:
            results = self._rerank_results(query, results, limit)
        else:
            results = results[:limit]
        
        return results
    
    def hybrid_search(
        self,
        query: str,
        limit: int = 20,
    ) -> list[SearchResult]:
        """Perform hybrid search combining sparse BM25 and dense vector scores.
        
        The final score is computed as:
        final = (1 - hybrid_weight) * sparse_score + hybrid_weight * dense_score
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of search results ordered by combined score
        """
        # Generate query embedding for dense search
        query_vector = self.embedding.embed(query)
        if query_vector is None:
            # Fall back to sparse-only search
            return self._sparse_only_search(query, limit)
        
        # Get dense vector scores (request more for merging)
        search_limit = limit * 3
        vector_results = self.vector_store.search(query_vector, search_limit)
        
        # Expand query if enabled
        search_query = self._expand_query(query) if self.use_query_expansion else query
        
        # Detect query type and adjust hybrid weight if adaptive
        query_type = self._detect_query_type(query)
        effective_weight = self._get_adaptive_weight(query_type)
        
        # Get sparse BM25 scores
        sparse_scores_raw = {}
        if self.sparse_embedding and self.sparse_embedding.is_fitted:
            for doc_id, score in self.sparse_embedding.get_scores(search_query):
                sparse_scores_raw[doc_id] = score
        
        # Build dense scores map
        dense_scores_raw = {vr.id: vr.score for vr in vector_results}
        
        # Normalize scores based on selected method
        if self.hybrid_normalization == "rrf":
            # Use Reciprocal Rank Fusion
            combined_results = self._normalize_scores_rrf(
                dense_scores_raw, sparse_scores_raw, search_limit
            )
        else:
            # Use min-max normalization (original method)
            sparse_scores = self._normalize_minmax(sparse_scores_raw)
            dense_scores = self._normalize_minmax(dense_scores_raw)
            
            # Get all candidate doc IDs
            all_doc_ids = set(dense_scores.keys()) | set(sparse_scores.keys())
            
            # Combine scores (both now in [0, 1] range)
            combined_results = []
            for doc_id in all_doc_ids:
                dense_score = dense_scores.get(doc_id, 0.0)
                sparse_score = sparse_scores.get(doc_id, 0.0)
                
                # Weighted combination using effective_weight
                final_score = (1 - effective_weight) * sparse_score + effective_weight * dense_score
                
                combined_results.append((doc_id, final_score))
            
            # Sort by combined score descending
            combined_results.sort(key=lambda x: x[1], reverse=True)
        
        # Fetch screenshot data and build results
        results = []
        for doc_id, score in combined_results[:search_limit]:
            screenshot = self.db.get_by_id(doc_id)
            if screenshot:
                results.append(SearchResult(
                    screenshot=screenshot,
                    score=score,
                    search_type="hybrid",
                ))
        
        # Apply reranking if enabled
        if self.use_reranker and results:
            results = self._rerank_results(query, results, limit, search_type="hybrid+rerank")
        else:
            results = results[:limit]
        
        return results
    
    def _sparse_only_search(
        self,
        query: str,
        limit: int,
    ) -> list[SearchResult]:
        """Perform sparse-only search when dense embedding fails."""
        if not self.sparse_embedding or not self.sparse_embedding.is_fitted:
            return []
        
        results = []
        for doc_id, score in self.sparse_embedding.get_scores_normalized(query)[:limit]:
            screenshot = self.db.get_by_id(doc_id)
            if screenshot:
                results.append(SearchResult(
                    screenshot=screenshot,
                    score=score,
                    search_type="sparse",
                ))
        
        return results
    
    def _rerank_results(
        self,
        query: str,
        results: list[SearchResult],
        limit: int,
        search_type: str = "vector+rerank",
    ) -> list[SearchResult]:
        """Rerank results using cross-encoder."""
        if not self.reranker or not results:
            return results[:limit]
        
        # Prepare documents for reranking
        items = []
        for r in results:
            # Combine visual description and OCR text
            doc_text = ""
            if r.screenshot.visual_description:
                doc_text += r.screenshot.visual_description
            if r.screenshot.ocr_text:
                doc_text += "\n" + r.screenshot.ocr_text
            items.append((r, doc_text))
        
        # Rerank
        reranked = self.reranker.rerank(
            query,
            [doc for _, doc in items],
            top_k=limit,
        )
        
        # Rebuild results with new scores
        result_map = {i: r for i, (r, _) in enumerate(items)}
        reranked_results = []
        for idx, score in reranked:
            result = result_map[idx]
            reranked_results.append(SearchResult(
                screenshot=result.screenshot,
                score=score,
                search_type=search_type,
            ))
        
        return reranked_results
    
    def _is_exact_query(self, query: str) -> bool:
        """Check if query should use exact matching (quoted)."""
        return (
            len(query) >= 2 and
            query.startswith('"') and
            query.endswith('"')
        )
    
    def _normalize_scores_rrf(
        self,
        dense_scores: dict[int, float],
        sparse_scores: dict[int, float],
        k: int = 60,
    ) -> list[tuple[int, float]]:
        """Normalize scores using Reciprocal Rank Fusion (RRF).
        
        RRF combines rankings from different sources without requiring score normalization.
        Formula: RRF(d) = sum(1 / (k + rank(d, i))) for each ranking i
        
        Args:
            dense_scores: Dict of doc_id -> dense score
            sparse_scores: Dict of doc_id -> sparse score
            k: RRF constant (typically 60)
            
        Returns:
            List of (doc_id, rrf_score) tuples sorted by score descending
        """
        # Create rankings from scores
        dense_ranking = sorted(dense_scores.items(), key=lambda x: x[1], reverse=True)
        sparse_ranking = sorted(sparse_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Build rank maps (doc_id -> rank in each list)
        dense_ranks = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(dense_ranking)}
        sparse_ranks = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(sparse_ranking)}
        
        # Calculate RRF scores for all documents
        all_doc_ids = set(dense_ranks.keys()) | set(sparse_ranks.keys())
        rrf_scores = {}
        
        for doc_id in all_doc_ids:
            rrf_score = 0.0
            
            # Add contribution from dense ranking
            if doc_id in dense_ranks:
                rrf_score += 1.0 / (k + dense_ranks[doc_id])
            
            # Add contribution from sparse ranking
            if doc_id in sparse_ranks:
                rrf_score += 1.0 / (k + sparse_ranks[doc_id])
            
            rrf_scores[doc_id] = rrf_score
        
        # Sort by RRF score descending
        results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return results
    
    def _normalize_minmax(self, scores: dict[int, float]) -> dict[int, float]:
        """Normalize scores using min-max normalization to [0, 1] range."""
        if not scores:
            return {}
        
        values = list(scores.values())
        min_val = min(values)
        max_val = max(values)
        score_range = max_val - min_val
        
        if score_range == 0:
            # All scores are the same
            return {doc_id: 1.0 if score > 0 else 0.0 for doc_id, score in scores.items()}
        
        return {
            doc_id: (score - min_val) / score_range
            for doc_id, score in scores.items()
        }
    
    def _expand_query(self, query: str) -> str:
        """Expand query with synonyms and related terms.
        
        Args:
            query: Original search query
            
        Returns:
            Expanded query string
        """
        # Simple synonym expansion for common terms
        synonyms = {
            "error": ["error", "exception", "failure", "bug", "issue"],
            "screenshot": ["screenshot", "image", "picture", "photo"],
            "code": ["code", "program", "script", "source"],
            "terminal": ["terminal", "console", "command", "shell"],
            "window": ["window", "dialog", "popup", "panel"],
        }
        
        words = query.lower().split()
        expanded_words = set(words)
        
        for word in words:
            if word in synonyms:
                expanded_words.update(synonyms[word])
        
        return " ".join(expanded_words)
    
    def _detect_query_type(self, query: str) -> str:
        """Detect query type to choose optimal search strategy.
        
        Args:
            query: Search query
            
        Returns:
            "exact" for exact match queries, "semantic" for semantic queries
        """
        # Simple heuristic: queries with quotes or very short queries are exact
        if query.startswith('"') and query.endswith('"'):
            return "exact"
        
        # Very short queries (1-2 words) might be exact
        words = query.split()
        if len(words) <= 2 and all(len(w) > 3 for w in words):
            return "exact"
        
        return "semantic"
    
    def _get_adaptive_weight(self, query_type: str) -> float:
        """Get adaptive hybrid weight based on query type.
        
        Args:
            query_type: "exact" or "semantic"
            
        Returns:
            Adjusted hybrid weight
        """
        if query_type == "exact":
            # For exact queries, favor sparse (BM25) search
            return max(0.0, self.hybrid_weight - 0.2)
        else:
            # For semantic queries, favor dense (vector) search
            return min(1.0, self.hybrid_weight + 0.2)
    
    def _format_fts_query(self, query: str) -> str:
        """Format a query for FTS5 MATCH.
        
        FTS5 uses specific syntax, so we need to handle:
        - Multiple words → implicit AND
        - Special characters → escape or remove
        """
        # Simple approach: wrap each word in quotes for exact phrase matching
        # and join with OR for flexibility
        words = query.split()
        if len(words) == 1:
            # Single word: simple match
            return words[0]
        else:
            # Multiple words: phrase match
            return f'"{query}"'

