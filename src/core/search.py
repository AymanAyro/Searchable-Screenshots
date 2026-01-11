"""Unified search interface with query routing and hybrid search.

This module uses LangChain services (embeddings, vector stores) while maintaining
the same interface for backward compatibility.
"""

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta
import hashlib
import time
import re

from ..core.database import Database, Screenshot
from ..core.logging import get_logger
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
    matched_fields: list[str] = field(default_factory=list)  # Fields that matched
    highlights: list[str] = field(default_factory=list)  # Highlighted text snippets
    explanation: Optional[str] = None  # Why this result matched
    
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
        min_search_score: float = 0.0,
        time_boost_factor: float = 0.1,
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
        self.min_search_score = min_search_score
        self.time_boost_factor = time_boost_factor
        self.logger = get_logger(__name__)
    
    def search(
        self,
        query: str,
        limit: int = 20,
        app_name: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> list[SearchResult]:
        """Search for screenshots matching query.
        
        Uses query routing:
        - Quoted queries ("like this") → FTS5 exact match
        - Unquoted queries → Hybrid search (sparse + dense) if available, otherwise vector search
        
        Args:
            query: Search query
            limit: Maximum number of results
            app_name: Optional filter by app name
            date_from: Optional filter by date (from)
            date_to: Optional filter by date (to)
            
        Returns:
            List of search results ordered by relevance
        """
        # Preprocess query
        query = self._preprocess_query(query)
        
        if not query:
            self.logger.warning("Empty query after preprocessing")
            return []
        
        # Query routing based on quotes and query type
        if self._is_exact_query(query):
            # Strip quotes and do FTS search
            exact_query = query[1:-1]
            results = self.fts_search(exact_query, limit, app_name=app_name, date_from=date_from, date_to=date_to)
        elif self._is_simple_keyword_query(query):
            # For simple keyword queries (1-2 words), prioritize FTS for exact matches
            # but also do hybrid search and combine results
            fts_results = self.fts_search(query, limit * 2, app_name=app_name, date_from=date_from, date_to=date_to)
            
            # Also do hybrid search to catch semantic matches
            if self.sparse_embedding and self.sparse_embedding.is_fitted:
                hybrid_results = self.hybrid_search(query, limit * 2, app_name=app_name, date_from=date_from, date_to=date_to)
            else:
                hybrid_results = self.vector_search(query, limit * 2, app_name=app_name, date_from=date_from, date_to=date_to)
            
            # Combine and deduplicate, prioritizing FTS results
            combined = {}
            for result in fts_results:
                path_str = str(result.file_path)
                # Boost FTS results for simple queries
                result.score = min(1.0, result.score + 0.2)
                combined[path_str] = result
            
            for result in hybrid_results:
                path_str = str(result.file_path)
                if path_str not in combined:
                    combined[path_str] = result
                else:
                    # Keep the higher score
                    if result.score > combined[path_str].score:
                        combined[path_str] = result
            
            results = list(combined.values())
            results.sort(key=lambda x: x.score, reverse=True)
            results = results[:limit]
        else:
            # Use hybrid search if sparse embedding is available
            if self.sparse_embedding and self.sparse_embedding.is_fitted:
                results = self.hybrid_search(query, limit, app_name=app_name, date_from=date_from, date_to=date_to)
            else:
                results = self.vector_search(query, limit, app_name=app_name, date_from=date_from, date_to=date_to)
        
        # Apply filters and quality improvements
        results = self._apply_filters(results, app_name, date_from, date_to)
        results = self._deduplicate_results(results)
        results = self._apply_keyword_boost(query, results)  # Boost exact keyword matches
        results = self._apply_time_boost(results)
        results = self._filter_by_min_score(results)
        
        self.logger.info(f"Search for '{query}' returned {len(results)} results")
        return results
    
    def _preprocess_query(self, query: str) -> str:
        """Preprocess and normalize search query.
        
        Args:
            query: Raw search query
            
        Returns:
            Normalized query string
        """
        if not query:
            return ""
        
        # Trim whitespace
        query = query.strip()
        
        # Normalize whitespace (multiple spaces to single)
        query = re.sub(r'\s+', ' ', query)
        
        # Validate minimum length
        if len(query) < 1:
            return ""
        
        return query
    
    def fts_search(
        self,
        query: str,
        limit: int = 20,
        app_name: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
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
        
        screenshots = self.db.fts_search(fts_query, limit * 2)  # Get more for filtering
        
        # Filter by metadata if provided
        filtered_screenshots = []
        for s in screenshots:
            if app_name and s.app_name != app_name:
                continue
            if date_from and s.captured_at and s.captured_at < date_from:
                continue
            if date_to and s.captured_at and s.captured_at > date_to:
                continue
            filtered_screenshots.append(s)
        
        # FTS5 results are already ranked by BM25
        results = []
        for i, s in enumerate(filtered_screenshots[:limit]):
            # Extract highlights from matched text
            highlights = self._extract_highlights(query, s)
            matched_fields = self._get_matched_fields(query, s)
            
            results.append(SearchResult(
                screenshot=s,
                score=1.0 - (i * 0.01),  # Approximate score based on rank
                search_type="fts",
                matched_fields=matched_fields,
                highlights=highlights,
            ))
        
        return results
    
    def vector_search(
        self,
        query: str,
        limit: int = 20,
        app_name: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
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
                # Apply metadata filters
                if app_name and screenshot.app_name != app_name:
                    continue
                if date_from and screenshot.captured_at and screenshot.captured_at < date_from:
                    continue
                if date_to and screenshot.captured_at and screenshot.captured_at > date_to:
                    continue
                
                highlights = self._extract_highlights(query, screenshot)
                matched_fields = self._get_matched_fields(query, screenshot)
                
                results.append(SearchResult(
                    screenshot=screenshot,
                    score=vr.score,
                    search_type="vector",
                    matched_fields=matched_fields,
                    highlights=highlights,
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
        app_name: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
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
        
        # Debug logging for top dense results
        self.logger.debug(f"DENSE VECTOR - Query: '{query}', Found {len(vector_results)} results")
        for i, vr in enumerate(vector_results[:5]):
            screenshot = self.db.get_by_id(vr.id)
            if screenshot:
                self.logger.debug(
                    f"DENSE VECTOR - Rank {i+1}, Doc {vr.id}: score={vr.score:.4f}, "
                    f"file={Path(screenshot.file_path).name}"
                )
        
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
                # Debug logging for top sparse results
                if len(sparse_scores_raw) <= 5:
                    screenshot = self.db.get_by_id(doc_id)
                    if screenshot:
                        self.logger.debug(
                            f"SPARSE BM25 - Doc {doc_id}: score={score:.4f}, "
                            f"OCR_len={len(screenshot.ocr_text) if screenshot.ocr_text else 0}, "
                            f"Visual_len={len(screenshot.visual_description) if screenshot.visual_description else 0}"
                        )
        
        # Build dense scores map
        dense_scores_raw = {vr.id: vr.score for vr in vector_results}
        
        # Debug: Log score ranges before normalization
        if dense_scores_raw:
            dense_min, dense_max = min(dense_scores_raw.values()), max(dense_scores_raw.values())
            self.logger.debug(f"HYBRID - Dense scores range: {dense_min:.4f} to {dense_max:.4f}")
        if sparse_scores_raw:
            sparse_min, sparse_max = min(sparse_scores_raw.values()), max(sparse_scores_raw.values())
            self.logger.debug(f"HYBRID - Sparse scores range: {sparse_min:.4f} to {sparse_max:.4f}")
        
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
        
        # Debug: Log top combined results
        self.logger.debug(f"HYBRID - Top 5 combined results before filtering:")
        for i, (doc_id, score) in enumerate(combined_results[:5]):
            screenshot = self.db.get_by_id(doc_id)
            if screenshot:
                dense_score = dense_scores_raw.get(doc_id, 0.0)
                sparse_score = sparse_scores_raw.get(doc_id, 0.0)
                self.logger.debug(
                    f"HYBRID - Rank {i+1}, Doc {doc_id}: combined={score:.4f}, "
                    f"dense={dense_score:.4f}, sparse={sparse_score:.4f}, "
                    f"file={Path(screenshot.file_path).name}"
                )
        
        # Fetch screenshot data and build results
        results = []
        for doc_id, score in combined_results[:search_limit]:
            screenshot = self.db.get_by_id(doc_id)
            if screenshot:
                # Apply metadata filters
                if app_name and screenshot.app_name != app_name:
                    continue
                if date_from and screenshot.captured_at and screenshot.captured_at < date_from:
                    continue
                if date_to and screenshot.captured_at and screenshot.captured_at > date_to:
                    continue
                
                highlights = self._extract_highlights(query, screenshot)
                matched_fields = self._get_matched_fields(query, screenshot)
                
                results.append(SearchResult(
                    screenshot=screenshot,
                    score=score,
                    search_type="hybrid",
                    matched_fields=matched_fields,
                    highlights=highlights,
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
                highlights = self._extract_highlights(query, screenshot)
                matched_fields = self._get_matched_fields(query, screenshot)
                results.append(SearchResult(
                    screenshot=screenshot,
                    score=score,
                    search_type="sparse",
                    matched_fields=matched_fields,
                    highlights=highlights,
                ))
        
        return results
    
    def sparse_only_search(
        self,
        query: str,
        limit: int = 20,
    ) -> list[SearchResult]:
        """Public method to perform sparse-only search for diagnostics.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of search results from BM25 only
        """
        return self._sparse_only_search(query, limit)
    
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
    
    def _is_simple_keyword_query(self, query: str) -> bool:
        """Check if query is a simple keyword query (1-2 words, no special characters).
        
        Simple queries like "cat", "error", "terminal" should prioritize exact matches.
        
        Args:
            query: Search query
            
        Returns:
            True if query is simple keyword query
        """
        words = query.strip().split()
        # Simple query: 1-2 words, no quotes, no special operators
        if len(words) <= 2 and not query.startswith('"'):
            # Check if words are mostly alphanumeric (simple keywords)
            for word in words:
                # Allow alphanumeric and basic punctuation
                if not re.match(r'^[\w\-]+$', word):
                    return False
            return True
        return False
    
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
    
    def _extract_highlights(self, query: str, screenshot: Screenshot) -> list[str]:
        """Extract text snippets that match the query for highlighting.
        
        Args:
            query: Search query
            screenshot: Screenshot to extract highlights from
            
        Returns:
            List of highlighted text snippets
        """
        highlights = []
        query_lower = query.lower()
        query_words = query_lower.split()
        
        # Check OCR text
        if screenshot.ocr_text:
            text_lower = screenshot.ocr_text.lower()
            for word in query_words:
                if word in text_lower:
                    # Find context around the match
                    idx = text_lower.find(word)
                    start = max(0, idx - 30)
                    end = min(len(screenshot.ocr_text), idx + len(word) + 30)
                    snippet = screenshot.ocr_text[start:end].strip()
                    if snippet and snippet not in highlights:
                        highlights.append(snippet)
        
        # Check visual description
        if screenshot.visual_description:
            desc_lower = screenshot.visual_description.lower()
            for word in query_words:
                if word in desc_lower:
                    idx = desc_lower.find(word)
                    start = max(0, idx - 30)
                    end = min(len(screenshot.visual_description), idx + len(word) + 30)
                    snippet = screenshot.visual_description[start:end].strip()
                    if snippet and snippet not in highlights:
                        highlights.append(snippet)
        
        return highlights[:5]  # Limit to 5 highlights
    
    def _get_matched_fields(self, query: str, screenshot: Screenshot) -> list[str]:
        """Determine which fields matched the query.
        
        Args:
            query: Search query
            screenshot: Screenshot to check
            
        Returns:
            List of field names that matched
        """
        matched = []
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Check OCR text
        if screenshot.ocr_text:
            text_lower = screenshot.ocr_text.lower()
            if any(word in text_lower for word in query_words):
                matched.append("ocr_text")
        
        # Check visual description
        if screenshot.visual_description:
            desc_lower = screenshot.visual_description.lower()
            if any(word in desc_lower for word in query_words):
                matched.append("visual_description")
        
        # Check app name
        if screenshot.app_name:
            app_lower = screenshot.app_name.lower()
            if any(word in app_lower for word in query_words):
                matched.append("app_name")
        
        # Check window title
        if screenshot.window_title:
            title_lower = screenshot.window_title.lower()
            if any(word in title_lower for word in query_words):
                matched.append("window_title")
        
        return matched
    
    def _apply_filters(
        self,
        results: list[SearchResult],
        app_name: Optional[str],
        date_from: Optional[datetime],
        date_to: Optional[datetime],
    ) -> list[SearchResult]:
        """Apply metadata filters to results (already applied in search methods, but kept for consistency).
        
        Args:
            results: Search results
            app_name: Optional app name filter
            date_from: Optional date from filter
            date_to: Optional date to filter
            
        Returns:
            Filtered results
        """
        # Filters are already applied in individual search methods
        # This method is kept for consistency and potential future use
        return results
    
    def _deduplicate_results(self, results: list[SearchResult]) -> list[SearchResult]:
        """Remove duplicate results based on file path.
        
        Args:
            results: Search results
            
        Returns:
            Deduplicated results (keeping highest score for each file)
        """
        seen_paths = {}
        for result in results:
            path_str = str(result.file_path)
            if path_str not in seen_paths:
                seen_paths[path_str] = result
            else:
                # Keep the result with higher score
                if result.score > seen_paths[path_str].score:
                    seen_paths[path_str] = result
        
        # Return results sorted by score
        deduplicated = list(seen_paths.values())
        deduplicated.sort(key=lambda x: x.score, reverse=True)
        return deduplicated
    
    def _apply_time_boost(self, results: list[SearchResult]) -> list[SearchResult]:
        """Apply time-based relevance boost to recent screenshots.
        
        Args:
            results: Search results
            
        Returns:
            Results with time boost applied
        """
        if self.time_boost_factor == 0.0 or not results:
            return results
        
        now = datetime.now()
        boosted_results = []
        
        for result in results:
            boosted_score = result.score
            if result.screenshot.captured_at:
                # Calculate days since capture
                days_old = (now - result.screenshot.captured_at).days
                # Boost decreases with age (max boost for today, no boost after 30 days)
                if days_old <= 30:
                    boost = self.time_boost_factor * (1.0 - days_old / 30.0)
                    boosted_score = min(1.0, result.score + boost)
            
            # Create new result with boosted score
            boosted_result = SearchResult(
                screenshot=result.screenshot,
                score=boosted_score,
                search_type=result.search_type,
                matched_fields=result.matched_fields,
                highlights=result.highlights,
                explanation=result.explanation,
            )
            boosted_results.append(boosted_result)
        
        # Re-sort by boosted score
        boosted_results.sort(key=lambda x: x.score, reverse=True)
        return boosted_results
    
    def _apply_keyword_boost(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        """Apply significant boost to results with exact keyword matches.
        
        This ensures that results containing the exact query terms get prioritized,
        especially important for simple queries like "cat", "error", etc.
        
        Args:
            query: Original search query
            results: Search results
            
        Returns:
            Results with keyword boost applied
        """
        if not query or not results:
            return results
        
        query_lower = query.lower().strip()
        query_words = set(word.lower() for word in query_lower.split() if len(word) >= 2)
        
        # Also check for plural/singular forms and common variations
        expanded_words = set(query_words)
        for word in query_words:
            # Add plural/singular variations
            if word.endswith('s') and len(word) > 2:
                expanded_words.add(word[:-1])  # Remove 's' for singular
            else:
                expanded_words.add(word + 's')  # Add 's' for plural
            # Add common variations
            if word == 'cat':
                expanded_words.update(['kitten', 'kittens', 'feline', 'felines'])
            elif word == 'dog':
                expanded_words.update(['puppy', 'puppies', 'canine', 'canines'])
        
        if not expanded_words:
            return results
        
        boosted_results = []
        
        for result in results:
            boosted_score = result.score
            match_count = 0
            strong_match = False
            keyword_match = False
            
            # Check OCR text for exact matches
            if result.screenshot.ocr_text:
                ocr_lower = result.screenshot.ocr_text.lower()
                for word in expanded_words:
                    if word in ocr_lower:
                        match_count += 1
                        # Check for whole word match (more important)
                        if f" {word} " in f" {ocr_lower} " or ocr_lower.startswith(word + " ") or ocr_lower.endswith(" " + word):
                            strong_match = True
            
            # Check visual description for exact matches (including keywords section)
            if result.screenshot.visual_description:
                desc_lower = result.screenshot.visual_description.lower()
                
                # Extract keywords section if present (format: "## 4. Search Keywords")
                keywords_section = ""
                if "search keywords" in desc_lower or "keywords" in desc_lower:
                    # Try to extract the keywords line
                    lines = result.screenshot.visual_description.split('\n')
                    in_keywords = False
                    for line in lines:
                        if "keywords" in line.lower() or "search keywords" in line.lower():
                            in_keywords = True
                        elif in_keywords and line.strip():
                            keywords_section += " " + line.lower()
                            if line.strip().endswith('"') or len(keywords_section) > 500:
                                break
                
                # Check main description
                for word in expanded_words:
                    if word in desc_lower:
                        match_count += 1
                        # Check for whole word match
                        if f" {word} " in f" {desc_lower} " or desc_lower.startswith(word + " ") or desc_lower.endswith(" " + word):
                            strong_match = True
                
                # Check keywords section (higher weight)
                if keywords_section:
                    for word in expanded_words:
                        if word in keywords_section:
                            keyword_match = True
                            match_count += 2  # Keywords section matches count double
                            if f" {word} " in f" {keywords_section} " or keywords_section.startswith(word + " ") or keywords_section.endswith(" " + word):
                                strong_match = True
            
            # Apply boost based on matches
            if keyword_match and strong_match:
                # Maximum boost for keyword section whole-word matches
                boost = 0.5 + (match_count * 0.1)
                boosted_score = min(1.0, result.score + boost)
            elif strong_match:
                # Strong boost for whole word matches
                boost = 0.3 + (match_count * 0.1)
                boosted_score = min(1.0, result.score + boost)
            elif match_count > 0:
                # Moderate boost for partial matches
                boost = 0.15 + (match_count * 0.05)
                boosted_score = min(1.0, result.score + boost)
            
            # Create new result with boosted score
            boosted_result = SearchResult(
                screenshot=result.screenshot,
                score=boosted_score,
                search_type=result.search_type,
                matched_fields=result.matched_fields,
                highlights=result.highlights,
                explanation=result.explanation,
            )
            boosted_results.append(boosted_result)
        
        # Re-sort by boosted score
        boosted_results.sort(key=lambda x: x.score, reverse=True)
        return boosted_results
    
    def _filter_by_min_score(self, results: list[SearchResult]) -> list[SearchResult]:
        """Filter results by minimum score threshold.
        
        Args:
            results: Search results
            
        Returns:
            Filtered results
        """
        if self.min_search_score <= 0.0:
            return results
        
        return [r for r in results if r.score >= self.min_search_score]
    
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

