"""Test embedding similarity to understand why wrong documents rank high."""

import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.config import ConfigManager
from src.core.database import Database
from src.services.embedding import EmbeddingService
from src.core.processor import ScreenshotProcessor


def cosine_similarity(vec1, vec2):
    """Compute cosine similarity."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))


def main():
    db = Database(ConfigManager().db_path)
    emb = EmbeddingService()
    processor = ScreenshotProcessor(None, None, None, None, None, None)
    
    query = "cat"
    query_emb = emb.embed(query)
    
    print("=" * 80)
    print("EMBEDDING SIMILARITY ANALYSIS")
    print("=" * 80)
    print(f"\nQuery: '{query}'")
    print(f"Query embedding dimension: {len(query_emb)}\n")
    
    # Test Doc 1 (the one that shouldn't match)
    doc1 = db.get_by_id(1)
    if doc1:
        combined1 = processor._combine_for_embedding(doc1.ocr_text, doc1.visual_description)
        doc1_emb = emb.embed(combined1)
        sim1 = cosine_similarity(query_emb, doc1_emb)
        has_cat1 = "cat" in combined1.lower() or "kitten" in combined1.lower()
        
        print(f"Doc 1: {Path(doc1.file_path).name}")
        print(f"  Contains 'cat'/'kitten': {has_cat1}")
        print(f"  Embedding similarity: {sim1:.4f}")
        print(f"  Combined text length: {len(combined1)}")
        print(f"  Combined preview: {combined1[:200]}...\n")
    
    # Test Doc 14 (the cat image)
    doc14 = db.get_by_id(14)
    if doc14:
        combined14 = processor._combine_for_embedding(doc14.ocr_text, doc14.visual_description)
        doc14_emb = emb.embed(combined14)
        sim14 = cosine_similarity(query_emb, doc14_emb)
        has_cat14 = "cat" in combined14.lower() or "kitten" in combined14.lower()
        
        print(f"Doc 14: {Path(doc14.file_path).name}")
        print(f"  Contains 'cat'/'kitten': {has_cat14}")
        print(f"  Embedding similarity: {sim14:.4f}")
        print(f"  Combined text length: {len(combined14)}")
        print(f"  Combined preview: {combined14[:200]}...\n")
    
    # Compare
    if doc1 and doc14:
        print("=" * 80)
        print("COMPARISON")
        print("=" * 80)
        print(f"Doc 1 similarity: {sim1:.4f}")
        print(f"Doc 14 similarity: {sim14:.4f}")
        print(f"Difference: {sim14 - sim1:.4f}")
        if sim1 > sim14:
            print("\n[PROBLEM] Doc 1 has HIGHER similarity than Doc 14!")
            print("This explains why dense search might rank Doc 1 higher.")
        else:
            print("\n[OK] Doc 14 has higher similarity, as expected.")


if __name__ == "__main__":
    main()
