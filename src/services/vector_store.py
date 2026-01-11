"""Qdrant vector store wrapper for semantic search using LangChain."""

from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import threading

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    HnswConfigDiff,
)


@dataclass
class VectorSearchResult:
    """Result from vector similarity search."""
    id: int
    score: float
    file_path: Optional[str] = None


class VectorStore:
    """Qdrant vector store for semantic screenshot search using LangChain."""
    
    COLLECTION_NAME = "screenshots"
    DEFAULT_DIMENSION = 768  # nomic-embed-text dimension (updated from 1024)
    
    def __init__(self, path: Path, dimension: Optional[int] = None):
        """Initialize the vector store.
        
        Args:
            path: Directory path for persistent storage
            dimension: Embedding vector dimension (auto-detected if None)
        """
        self.path = path
        # Use provided dimension or default
        self.dimension = dimension or self.DEFAULT_DIMENSION
        path.mkdir(parents=True, exist_ok=True)
        
        # Thread lock for thread-safe operations
        self._lock = threading.Lock()
        self._closed = False
        
        # Initialize Qdrant client for collection management
        self.client = QdrantClient(path=str(path))
        self._ensure_collection()
        
        # Initialize LangChain Qdrant vector store for potential future use
        # We work with pre-computed vectors, so we use the client directly for now
        # but keep LangChain integration available
        from langchain_community.embeddings import FakeEmbeddings
        dummy_embeddings = FakeEmbeddings(size=dimension)
        
        self._vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.COLLECTION_NAME,
            embedding=dummy_embeddings,
        )
    
    def _ensure_collection(self) -> None:
        """Create the collection if it doesn't exist, or update if dimension mismatch."""
        with self._lock:
            try:
                collections = self.client.get_collections().collections
                collection_names = [c.name for c in collections]
            except Exception as e:
                # If client is closed, try to recreate it
                if "closed" in str(e).lower():
                    self.client = QdrantClient(path=str(self.path))
                    collections = self.client.get_collections().collections
                    collection_names = [c.name for c in collections]
                else:
                    raise
            
            if self.COLLECTION_NAME not in collection_names:
                # Create new collection with correct dimension and optimized HNSW
                # Optimize HNSW parameters for better search performance
                hnsw_config = HnswConfigDiff(
                    m=16,  # Number of bi-directional links (default: 16, higher = better recall, slower)
                    ef_construct=200,  # Size of dynamic candidate list (default: 100, higher = better quality)
                    full_scan_threshold=10000,  # Use full scan if collection is smaller
                )
                
                self.client.create_collection(
                    collection_name=self.COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=self.dimension,
                        distance=Distance.COSINE,
                        hnsw_config=hnsw_config,
                    ),
                )
            else:
                # Check if existing collection has correct dimension
                try:
                    collection_info = self.client.get_collection(self.COLLECTION_NAME)
                    existing_dim = collection_info.config.params.vectors.size
                    
                    if existing_dim != self.dimension:
                        # Dimension mismatch - need to recreate collection
                        print(f"Warning: Vector store dimension mismatch ({existing_dim} vs {self.dimension}).")
                        print("Recreating collection with correct dimension. Existing vectors will be lost.")
                        print("You may need to re-index your screenshots.")
                        
                        # Delete and recreate collection with optimized HNSW
                        self.client.delete_collection(self.COLLECTION_NAME)
                        hnsw_config = HnswConfigDiff(
                            m=16,
                            ef_construct=200,
                            full_scan_threshold=10000,
                        )
                        self.client.create_collection(
                            collection_name=self.COLLECTION_NAME,
                            vectors_config=VectorParams(
                                size=self.dimension,
                                distance=Distance.COSINE,
                                hnsw_config=hnsw_config,
                            ),
                        )
                except Exception as e:
                    print(f"Warning: Could not verify collection dimension: {e}")
                    # Continue anyway - will fail on add if dimension is wrong
    
    def add(
        self,
        id: int,
        vector: list[float],
        file_path: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Add a vector to the store.
        
        Args:
            id: Unique ID (should match SQLite screenshot ID)
            vector: Embedding vector
            file_path: Path to the screenshot file
            metadata: Additional metadata to store
        """
        with self._lock:
            # Combine metadata
            doc_metadata = metadata or {}
            if file_path:
                doc_metadata["file_path"] = file_path
            doc_metadata["id"] = id
            
            # Create a Document with the vector
            # LangChain Qdrant uses documents, but we need to add vectors directly
            # We'll use the underlying client to add vectors directly
            from qdrant_client.models import PointStruct
            
            point = PointStruct(
                id=id,
                vector=vector,
                payload=doc_metadata,
            )
            
            try:
                self.client.upsert(
                    collection_name=self.COLLECTION_NAME,
                    points=[point],
                )
            except Exception as e:
                # If client is closed, try to recreate it
                if "closed" in str(e).lower():
                    self.client = QdrantClient(path=str(self.path))
                    self.client.upsert(
                        collection_name=self.COLLECTION_NAME,
                        points=[point],
                    )
                else:
                    raise
    
    def add_batch(
        self,
        ids: list[int],
        vectors: list[list[float]],
        file_paths: Optional[list[str]] = None,
        metadata_list: Optional[list[dict]] = None,
    ) -> None:
        """Add multiple vectors to the store.
        
        Args:
            ids: List of unique IDs
            vectors: List of embedding vectors
            file_paths: Optional list of file paths
            metadata_list: Optional list of metadata dicts
        """
        with self._lock:
            from qdrant_client.models import PointStruct
            
            points = []
            for i, (id_, vector) in enumerate(zip(ids, vectors)):
                payload = {}
                if metadata_list and i < len(metadata_list):
                    payload = metadata_list[i] or {}
                if file_paths and i < len(file_paths):
                    payload["file_path"] = file_paths[i]
                payload["id"] = id_
                
                points.append(PointStruct(
                    id=id_,
                    vector=vector,
                    payload=payload,
                ))
            
            try:
                self.client.upsert(
                    collection_name=self.COLLECTION_NAME,
                    points=points,
                )
            except Exception as e:
                # If client is closed, try to recreate it
                if "closed" in str(e).lower():
                    self.client = QdrantClient(path=str(self.path))
                    self.client.upsert(
                        collection_name=self.COLLECTION_NAME,
                        points=points,
                    )
                else:
                    raise
    
    def search(
        self,
        query_vector: list[float],
        limit: int = 20,
        score_threshold: Optional[float] = None,
    ) -> list[VectorSearchResult]:
        """Search for similar vectors.
        
        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            score_threshold: Minimum similarity score (optional)
            
        Returns:
            List of search results ordered by similarity
        """
        with self._lock:
            # Use direct client query for vector search
            # (LangChain QdrantVectorStore works with Documents, but we use pre-computed vectors)
            try:
                results = self.client.query_points(
                    collection_name=self.COLLECTION_NAME,
                    query=query_vector,
                    limit=limit,
                    score_threshold=score_threshold,
                )
            except Exception as e:
                # If client is closed, try to recreate it
                if "closed" in str(e).lower():
                    self.client = QdrantClient(path=str(self.path))
                    results = self.client.query_points(
                        collection_name=self.COLLECTION_NAME,
                        query=query_vector,
                        limit=limit,
                        score_threshold=score_threshold,
                    )
                else:
                    raise
            
            return [
                VectorSearchResult(
                    id=int(r.id),
                    score=r.score,
                    file_path=r.payload.get("file_path") if r.payload else None,
                )
                for r in results.points
            ]
    
    def delete(self, id: int) -> None:
        """Delete a vector by ID."""
        with self._lock:
            try:
                self.client.delete(
                    collection_name=self.COLLECTION_NAME,
                    points_selector=[id],
                )
            except Exception as e:
                if "closed" in str(e).lower():
                    self.client = QdrantClient(path=str(self.path))
                    self.client.delete(
                        collection_name=self.COLLECTION_NAME,
                        points_selector=[id],
                    )
                else:
                    raise
    
    def delete_batch(self, ids: list[int]) -> None:
        """Delete multiple vectors by ID."""
        if ids:
            with self._lock:
                try:
                    self.client.delete(
                        collection_name=self.COLLECTION_NAME,
                        points_selector=ids,
                    )
                except Exception as e:
                    if "closed" in str(e).lower():
                        self.client = QdrantClient(path=str(self.path))
                        self.client.delete(
                            collection_name=self.COLLECTION_NAME,
                            points_selector=ids,
                        )
                    else:
                        raise
    
    def get_count(self) -> int:
        """Get the number of vectors in the store."""
        with self._lock:
            try:
                info = self.client.get_collection(self.COLLECTION_NAME)
                return info.points_count
            except Exception as e:
                if "closed" in str(e).lower():
                    self.client = QdrantClient(path=str(self.path))
                    info = self.client.get_collection(self.COLLECTION_NAME)
                    return info.points_count
                else:
                    raise
    
    def clear(self) -> None:
        """Delete and recreate the collection."""
        with self._lock:
            self.client.delete_collection(self.COLLECTION_NAME)
            self._ensure_collection()
    
    def close(self) -> None:
        """Close the client connection."""
        with self._lock:
            if not self._closed:
                self._closed = True
                try:
                    self.client.close()
                except Exception:
                    pass  # Ignore errors when closing
        # LangChain vector store doesn't need explicit closing
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
