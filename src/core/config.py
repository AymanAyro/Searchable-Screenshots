"""Configuration management for Searchable Screenshots."""

from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import Optional


@dataclass
class ScanFolder:
    """A folder to scan for screenshots."""
    path: str
    include_subfolders: bool = True
    
    def to_dict(self) -> dict:
        return {"path": self.path, "include_subfolders": self.include_subfolders}
    
    @classmethod
    def from_dict(cls, data: dict) -> "ScanFolder":
        return cls(path=data["path"], include_subfolders=data.get("include_subfolders", True))


@dataclass
class APIConfig:
    """Configuration for external APIs (Ollama)."""
    ollama_url: str = "http://localhost:11434"
    vision_model: str = "gemma3:4b"
    embed_model: str = "nomic-embed-text:latest"
    
    def to_dict(self) -> dict:
        return {
            "ollama_url": self.ollama_url,
            "vision_model": self.vision_model,
            "embed_model": self.embed_model,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "APIConfig":
        return cls(
            ollama_url=data.get("ollama_url", "http://localhost:11434"),
            vision_model=data.get("vision_model", "gemma3:4b"),
            embed_model=data.get("embed_model", "nomic-embed-text:latest"),
        )


@dataclass
class AppConfig:
    """Main application configuration."""
    scan_folders: list[ScanFolder] = field(default_factory=list)
    api: APIConfig = field(default_factory=APIConfig)
    use_reranker: bool = False
    hybrid_search_weight: float = 0.5  # 0.0 = sparse only, 1.0 = dense only
    parallel_processing: int = 1  # 1-20, concurrent images during indexing
    vision_prompt_style: str = "detailed"  # "fast" or "detailed"
    embedding_cache_enabled: bool = True
    query_cache_enabled: bool = True
    use_query_expansion: bool = False
    hybrid_normalization: str = "rrf"  # "rrf", "minmax", or "sigmoid"
    
    def to_dict(self) -> dict:
        return {
            "scan_folders": [f.to_dict() for f in self.scan_folders],
            "api": self.api.to_dict(),
            "use_reranker": self.use_reranker,
            "hybrid_search_weight": self.hybrid_search_weight,
            "parallel_processing": self.parallel_processing,
            "vision_prompt_style": self.vision_prompt_style,
            "embedding_cache_enabled": self.embedding_cache_enabled,
            "query_cache_enabled": self.query_cache_enabled,
            "use_query_expansion": self.use_query_expansion,
            "hybrid_normalization": self.hybrid_normalization,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "AppConfig":
        return cls(
            scan_folders=[ScanFolder.from_dict(f) for f in data.get("scan_folders", [])],
            api=APIConfig.from_dict(data.get("api", {})),
            use_reranker=data.get("use_reranker", False),
            hybrid_search_weight=data.get("hybrid_search_weight", 0.5),
            parallel_processing=data.get("parallel_processing", 1),
            vision_prompt_style=data.get("vision_prompt_style", "detailed"),
            embedding_cache_enabled=data.get("embedding_cache_enabled", True),
            query_cache_enabled=data.get("query_cache_enabled", True),
            use_query_expansion=data.get("use_query_expansion", False),
            hybrid_normalization=data.get("hybrid_normalization", "rrf"),
        )


class ConfigManager:
    """Manages loading and saving application configuration."""
    
    DEFAULT_CONFIG_DIR = Path.home() / ".config" / "searchable-screenshots"
    CONFIG_FILENAME = "config.json"
    
    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or self.DEFAULT_CONFIG_DIR
        self.config_path = self.config_dir / self.CONFIG_FILENAME
        self._config: Optional[AppConfig] = None
    
    @property
    def config(self) -> AppConfig:
        """Get the current configuration, loading from file if needed."""
        if self._config is None:
            self._config = self.load()
        return self._config
    
    def load(self) -> AppConfig:
        """Load configuration from file, creating default if doesn't exist."""
        if not self.config_path.exists():
            return AppConfig()
        
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return AppConfig.from_dict(data)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Failed to load config, using defaults: {e}")
            return AppConfig()
    
    def save(self, config: Optional[AppConfig] = None) -> None:
        """Save configuration to file."""
        if config is not None:
            self._config = config
        
        if self._config is None:
            return
        
        self.config_dir.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self._config.to_dict(), f, indent=2)
    
    def add_scan_folder(self, path: str, include_subfolders: bool = True) -> None:
        """Add a new scan folder to the configuration."""
        folder = ScanFolder(path=path, include_subfolders=include_subfolders)
        self.config.scan_folders.append(folder)
        self.save()
    
    def remove_scan_folder(self, path: str) -> bool:
        """Remove a scan folder from the configuration."""
        for i, folder in enumerate(self.config.scan_folders):
            if folder.path == path:
                self.config.scan_folders.pop(i)
                self.save()
                return True
        return False
    
    @property
    def data_dir(self) -> Path:
        """Get the data directory for databases and vector stores."""
        return self.config_dir / "data"
    
    @property
    def db_path(self) -> Path:
        """Get the SQLite database file path."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir / "screenshots.db"
    
    @property
    def vector_store_path(self) -> Path:
        """Get the Qdrant vector store directory path."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir / "vectors"
    
    @property
    def sparse_index_path(self) -> Path:
        """Get the BM25 sparse index file path."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir / "bm25_index.pkl"
