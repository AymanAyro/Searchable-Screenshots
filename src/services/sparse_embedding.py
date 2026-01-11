"""Sparse embedding service using BM25 for lexical search."""

import pickle
import re
from pathlib import Path
from typing import Optional
import numpy as np
from rank_bm25 import BM25Okapi

# Try to import stemming library, fall back if not available
try:
    from nltk.stem import PorterStemmer
    STEMMING_AVAILABLE = True
except ImportError:
    STEMMING_AVAILABLE = False
    PorterStemmer = None

# Common English stop words
STOP_WORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
    'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
    'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
    'had', 'what', 'said', 'each', 'which', 'their', 'time', 'if',
    'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some', 'her',
    'would', 'make', 'like', 'into', 'him', 'has', 'two', 'more',
    'very', 'after', 'words', 'long', 'than', 'first', 'been', 'call',
    'who', 'oil', 'sit', 'now', 'find', 'down', 'day', 'did', 'get',
    'come', 'made', 'may', 'part'
}


class SparseEmbeddingService:
    """Generate sparse embeddings using BM25 algorithm.
    
    BM25 provides lexical matching that complements dense semantic embeddings.
    The corpus is trained on document texts and scores are computed at query time.
    """
    
    def __init__(self, use_stemming: bool = True, remove_stop_words: bool = True):
        """Initialize sparse embedding service.
        
        Args:
            use_stemming: Whether to use Porter stemming (requires nltk)
            remove_stop_words: Whether to remove common stop words
        """
        self._bm25: Optional[BM25Okapi] = None
        self._corpus: list[str] = []
        self._doc_ids: list[int] = []
        self._tokenized_corpus: list[list[str]] = []
        self.use_stemming = use_stemming and STEMMING_AVAILABLE
        self.remove_stop_words = remove_stop_words
        
        if self.use_stemming:
            self._stemmer = PorterStemmer()
        else:
            self._stemmer = None
    
    def tokenize(self, text: str) -> list[str]:
        """Tokenize text for BM25 with optional stemming and stop word removal.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        if not text:
            return []
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Split into words
        tokens = text.split()
        
        # Remove stop words if enabled
        if self.remove_stop_words:
            tokens = [t for t in tokens if t not in STOP_WORDS]
        
        # Apply stemming if enabled
        if self.use_stemming and self._stemmer:
            tokens = [self._stemmer.stem(t) for t in tokens]
        
        return tokens
    
    def fit(self, documents: list[tuple[int, str]]) -> None:
        """Train BM25 on a corpus of documents.
        
        Args:
            documents: List of (doc_id, text) tuples
        """
        self._doc_ids = []
        self._corpus = []
        self._tokenized_corpus = []
        
        for doc_id, text in documents:
            if text:
                self._doc_ids.append(doc_id)
                self._corpus.append(text)
                self._tokenized_corpus.append(self.tokenize(text))
        
        if self._tokenized_corpus:
            self._bm25 = BM25Okapi(self._tokenized_corpus)
        else:
            self._bm25 = None
    
    def add_document(self, doc_id: int, text: str) -> None:
        """Add a single document to the corpus.
        
        Note: This rebuilds the BM25 index. For batch additions, use fit() or add_documents_batch().
        
        Args:
            doc_id: Unique document ID
            text: Document text content
        """
        if not text:
            return
        
        # Check if document already exists (update case)
        if doc_id in self._doc_ids:
            idx = self._doc_ids.index(doc_id)
            self._corpus[idx] = text
            self._tokenized_corpus[idx] = self.tokenize(text)
        else:
            self._doc_ids.append(doc_id)
            self._corpus.append(text)
            self._tokenized_corpus.append(self.tokenize(text))
        
        # Rebuild BM25 index
        if self._tokenized_corpus:
            self._bm25 = BM25Okapi(self._tokenized_corpus)
    
    def add_documents_batch(self, documents: list[tuple[int, str]]) -> None:
        """Add multiple documents to the corpus in a single batch operation.
        
        This is more efficient than calling add_document() multiple times
        as it only rebuilds the BM25 index once.
        
        Args:
            documents: List of (doc_id, text) tuples
        """
        if not documents:
            return
        
        # Process all documents
        for doc_id, text in documents:
            if not text:
                continue
            
            # Check if document already exists (update case)
            if doc_id in self._doc_ids:
                idx = self._doc_ids.index(doc_id)
                self._corpus[idx] = text
                self._tokenized_corpus[idx] = self.tokenize(text)
            else:
                self._doc_ids.append(doc_id)
                self._corpus.append(text)
                self._tokenized_corpus.append(self.tokenize(text))
        
        # Rebuild BM25 index once for all documents
        if self._tokenized_corpus:
            self._bm25 = BM25Okapi(self._tokenized_corpus)
    
    def remove_document(self, doc_id: int) -> None:
        """Remove a document from the corpus.
        
        Args:
            doc_id: Document ID to remove
        """
        if doc_id not in self._doc_ids:
            return
        
        idx = self._doc_ids.index(doc_id)
        self._doc_ids.pop(idx)
        self._corpus.pop(idx)
        self._tokenized_corpus.pop(idx)
        
        # Rebuild BM25 index
        if self._tokenized_corpus:
            self._bm25 = BM25Okapi(self._tokenized_corpus)
        else:
            self._bm25 = None
    
    def get_scores(self, query: str) -> list[tuple[int, float]]:
        """Get BM25 scores for all documents against a query.
        
        Args:
            query: Search query text
            
        Returns:
            List of (doc_id, score) tuples sorted by score descending
        """
        if not self._bm25 or not query:
            return []
        
        tokenized_query = self.tokenize(query)
        if not tokenized_query:
            return []
        
        scores = self._bm25.get_scores(tokenized_query)
        
        # Pair doc_ids with scores and sort by score descending
        results = list(zip(self._doc_ids, scores))
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def get_scores_normalized(self, query: str) -> list[tuple[int, float]]:
        """Get normalized BM25 scores (0-1 range) for combining with dense scores.
        
        Uses min-max normalization to scale scores to [0, 1].
        
        Args:
            query: Search query text
            
        Returns:
            List of (doc_id, normalized_score) tuples sorted by score descending
        """
        results = self.get_scores(query)
        if not results:
            return []
        
        scores = [score for _, score in results]
        min_score = min(scores)
        max_score = max(scores)
        
        # Avoid division by zero
        score_range = max_score - min_score
        if score_range == 0:
            # All scores are the same
            normalized = [(doc_id, 1.0 if score > 0 else 0.0) for doc_id, score in results]
        else:
            normalized = [
                (doc_id, (score - min_score) / score_range)
                for doc_id, score in results
            ]
        
        return normalized
    
    def save(self, path: Path) -> None:
        """Save the BM25 index to disk.
        
        Args:
            path: File path to save the index (pickle format)
        """
        data = {
            "doc_ids": self._doc_ids,
            "corpus": self._corpus,
            "tokenized_corpus": self._tokenized_corpus,
        }
        
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)
    
    def load(self, path: Path) -> bool:
        """Load the BM25 index from disk.
        
        Args:
            path: File path to load the index from
            
        Returns:
            True if loaded successfully, False if file doesn't exist
        """
        if not path.exists():
            return False
        
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            
            self._doc_ids = data["doc_ids"]
            self._corpus = data["corpus"]
            self._tokenized_corpus = data["tokenized_corpus"]
            
            if self._tokenized_corpus:
                self._bm25 = BM25Okapi(self._tokenized_corpus)
            else:
                self._bm25 = None
            
            return True
        except Exception as e:
            print(f"Failed to load BM25 index: {e}")
            return False
    
    def clear(self) -> None:
        """Clear all data and reset the service."""
        self._bm25 = None
        self._corpus = []
        self._doc_ids = []
        self._tokenized_corpus = []
    
    @property
    def document_count(self) -> int:
        """Get the number of documents in the corpus."""
        return len(self._doc_ids)
    
    @property
    def is_fitted(self) -> bool:
        """Check if the BM25 model has been fitted."""
        return self._bm25 is not None
