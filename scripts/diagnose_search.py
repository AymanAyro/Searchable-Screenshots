"""Diagnostic script to investigate search quality issues."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.config import ConfigManager
from src.core.database import Database
from src.core.search import SearchEngine
from src.services.embedding import EmbeddingService
from src.services.vector_store import VectorStore
from src.services.sparse_embedding import SparseEmbeddingService
import numpy as np


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot_product / (norm1 * norm2))


def main():
    """Run diagnostic tests."""
    print("=" * 80)
    print("SEARCH QUALITY DIAGNOSTIC")
    print("=" * 80)
    
    # Initialize services
    config_manager = ConfigManager()
    db = Database(config_manager.db_path)
    
    api_config = config_manager.config.api
    embedding = EmbeddingService(api_config.ollama_url, api_config.embed_model)
    
    # Get embedding dimension
    test_emb = embedding.embed("test")
    if not test_emb:
        print("ERROR: Could not generate test embedding. Is Ollama running?")
        return
    embedding_dim = len(test_emb)
    
    vector_store = VectorStore(config_manager.vector_store_path, dimension=embedding_dim)
    sparse_embedding = SparseEmbeddingService()
    
    # Load sparse index if exists
    if config_manager.sparse_index_path.exists():
        sparse_embedding.load(config_manager.sparse_index_path)
        print(f"Loaded sparse index with {sparse_embedding.document_count} documents")
    else:
        print("WARNING: Sparse index not found. Rebuilding...")
        # Rebuild sparse index from database
        all_screenshots = []
        # Get all screenshots (simplified - you might need to adjust this)
        # For now, we'll work with what we have
        
    search_engine = SearchEngine(
        db=db,
        vector_store=vector_store,
        embedding_service=embedding,
        sparse_embedding=sparse_embedding if sparse_embedding.is_fitted else None,
        hybrid_weight=0.5,  # Default
        hybrid_normalization="rrf",  # Default
    )
    
    # Create processor for text combination analysis
    from src.core.processor import ScreenshotProcessor
    processor = ScreenshotProcessor(None, None, None, None, None, None)
    
    query = "cat"
    print(f"\n{'=' * 80}")
    print(f"TESTING QUERY: '{query}'")
    print(f"{'=' * 80}\n")
    
    # Test 1: Sparse search only
    print("TEST 1: SPARSE (BM25) SEARCH ONLY")
    print("-" * 80)
    if sparse_embedding.is_fitted:
        sparse_results = search_engine.sparse_only_search(query, limit=10)
        print(f"Found {len(sparse_results)} results from BM25")
        for i, result in enumerate(sparse_results):
            filename = Path(result.screenshot.file_path).name
            ocr_preview = (result.screenshot.ocr_text[:50] + "...") if result.screenshot.ocr_text and len(result.screenshot.ocr_text) > 50 else (result.screenshot.ocr_text or "")
            print(f"  {i+1}. Doc {result.id}: score={result.score:.4f}, file={filename}")
            print(f"     OCR preview: {ocr_preview}")
            
            # Tokenize the combined text to see what BM25 sees
            combined = processor._combine_for_embedding(result.screenshot.ocr_text, result.screenshot.visual_description)
            if combined:
                tokens, debug = sparse_embedding.tokenize_with_debug(combined, result.id)
                print(f"     Tokens: {len(tokens)} total, sample: {tokens[:15]}")
                if 'found_keywords' in debug:
                    print(f"     [FOUND] Keywords: {debug['found_keywords']}")
    else:
        print("  Sparse embedding not fitted - skipping")
    
    # Test 2: Dense search only
    print(f"\nTEST 2: DENSE (VECTOR) SEARCH ONLY")
    print("-" * 80)
    vector_results_list = search_engine.vector_search(query, limit=10)
    print(f"Found {len(vector_results_list)} results from vector search")
    for i, result in enumerate(vector_results_list):
        filename = Path(result.screenshot.file_path).name
        print(f"  {i+1}. Doc {result.id}: score={result.score:.4f}, file={filename}")
        
        # Get the query embedding for comparison
        query_vector = embedding.embed(query)
        if query_vector:
            # Get the stored vector
            stored_vector = None
            try:
                points = vector_store.client.retrieve(
                    collection_name=vector_store.COLLECTION_NAME,
                    ids=[result.id]
                )
                if points:
                    stored_vector = points[0].vector
            except Exception as e:
                print(f"     Could not retrieve stored vector: {e}")
            
            if stored_vector:
                similarity = cosine_similarity(query_vector, stored_vector)
                print(f"     Cosine similarity: {similarity:.4f}")
                print(f"     Qdrant score: {result.score:.4f}")
    
    # Test 3: Hybrid search
    print(f"\nTEST 3: HYBRID SEARCH")
    print("-" * 80)
    results = search_engine.search(query, limit=10)
    print(f"Found {len(results)} results from hybrid search")
    for i, result in enumerate(results):
        filename = Path(result.screenshot.file_path).name
        print(f"  {i+1}. Doc {result.id}: score={result.score:.4f}, type={result.search_type}, file={filename}")
        if result.matched_fields:
            print(f"     Matched fields: {result.matched_fields}")
    
    # Test 4: Check tokenization for query
    print(f"\nTEST 4: TOKENIZATION ANALYSIS")
    print("-" * 80)
    query_tokens, query_debug = sparse_embedding.tokenize_with_debug(query) if sparse_embedding.is_fitted else ([], {})
    print(f"Query '{query}' tokenizes to: {query_tokens}")
    if query_debug:
        print(f"Tokenization debug: {query_debug}")
    
    # Test 5: Check specific documents
    print(f"\nTEST 5: CHECKING SPECIFIC DOCUMENTS")
    print("-" * 80)
    print("Searching for documents that might contain 'cat' or 'kitten'...")
    
    all_screenshots = []
    # Get a sample of screenshots to check
    try:
        for screenshot in db.fts_search("cat OR kitten OR cats", limit=20):
            all_screenshots.append(screenshot)
    except Exception as e:
        print(f"  FTS search failed: {e}")
        # Try alternative approach
        print("  Trying alternative method to get screenshots...")
    
    print(f"Found {len(all_screenshots)} screenshots via FTS with 'cat OR kitten OR cats'")
    for screenshot in all_screenshots[:5]:
        filename = Path(screenshot.file_path).name
        print(f"\n  Document: {filename}")
        print(f"  OCR text length: {len(screenshot.ocr_text) if screenshot.ocr_text else 0}")
        print(f"  Visual desc length: {len(screenshot.visual_description) if screenshot.visual_description else 0}")
        
        # Check what gets tokenized
        combined = processor._combine_for_embedding(screenshot.ocr_text, screenshot.visual_description)
        if combined:
            if sparse_embedding.is_fitted:
                tokens, debug = sparse_embedding.tokenize_with_debug(combined)
                print(f"  Combined text length: {len(combined)}")
                print(f"  Tokens generated: {len(tokens)}")
                print(f"  Sample tokens: {tokens[:20]}")
                if 'found_keywords' in debug:
                    print(f"  [FOUND] Keywords in tokens: {debug['found_keywords']}")
            
            # Check if "cat" appears in the text
            combined_lower = combined.lower()
            if "cat" in combined_lower or "kitten" in combined_lower:
                print(f"  [FOUND] Contains 'cat' or 'kitten' in combined text")
                # Show context
                if "cat" in combined_lower:
                    idx = combined_lower.find("cat")
                    context = combined[max(0, idx-50):idx+50]
                    print(f"  Context for 'cat': ...{context}...")
                if "kitten" in combined_lower:
                    idx = combined_lower.find("kitten")
                    context = combined[max(0, idx-50):idx+50]
                    print(f"  Context for 'kitten': ...{context}...")
            
            # Check visual description keywords section
            if screenshot.visual_description and "search keywords" in screenshot.visual_description.lower():
                print(f"  Visual description has keywords section")
                # Try to extract keywords
                lines = screenshot.visual_description.split('\n')
                in_keywords = False
                keywords_text = ""
                for line in lines:
                    if "keywords" in line.lower() or "search keywords" in line.lower():
                        in_keywords = True
                    elif in_keywords and line.strip():
                        keywords_text += " " + line
                        if len(keywords_text) > 200:
                            break
                if keywords_text:
                    print(f"  Keywords section: {keywords_text[:150]}...")
                    if "cat" in keywords_text.lower() or "kitten" in keywords_text.lower():
                        print(f"  [FOUND] Keywords section contains 'cat' or 'kitten'")
    
    # Test 6: Embedding similarity comparison
    print(f"\nTEST 6: EMBEDDING SIMILARITY COMPARISON")
    print("-" * 80)
    query_vector = embedding.embed(query)
    if query_vector:
        print(f"Query '{query}' embedding dimension: {len(query_vector)}")
        
        # Get embeddings for top results from both searches
        print("\nComparing query embedding with top results:")
        
        # Get top sparse result
        if sparse_embedding.is_fitted and sparse_results:
            top_sparse = sparse_results[0]
            sparse_combined = processor._combine_for_embedding(
                top_sparse.screenshot.ocr_text, 
                top_sparse.screenshot.visual_description
            )
            if sparse_combined:
                sparse_doc_emb = embedding.embed(sparse_combined)
                if sparse_doc_emb:
                    sparse_sim = cosine_similarity(query_vector, sparse_doc_emb)
                    print(f"  Top sparse result (Doc {top_sparse.id}): similarity={sparse_sim:.4f}, BM25_score={top_sparse.score:.4f}")
        
        # Get top dense result
        if vector_results_list:
            top_dense = vector_results_list[0]
            dense_combined = processor._combine_for_embedding(
                top_dense.screenshot.ocr_text,
                top_dense.screenshot.visual_description
            )
            if dense_combined:
                dense_doc_emb = embedding.embed(dense_combined)
                if dense_doc_emb:
                    dense_sim = cosine_similarity(query_vector, dense_doc_emb)
                    print(f"  Top dense result (Doc {top_dense.id}): similarity={dense_sim:.4f}, Vector_score={top_dense.score:.4f}")
        
        # Get top hybrid result
        if results:
            top_hybrid = results[0]
            hybrid_combined = processor._combine_for_embedding(
                top_hybrid.screenshot.ocr_text,
                top_hybrid.screenshot.visual_description
            )
            if hybrid_combined:
                hybrid_doc_emb = embedding.embed(hybrid_combined)
                if hybrid_doc_emb:
                    hybrid_sim = cosine_similarity(query_vector, hybrid_doc_emb)
                    print(f"  Top hybrid result (Doc {top_hybrid.id}): similarity={hybrid_sim:.4f}, Hybrid_score={top_hybrid.score:.4f}")
    
    # Summary Analysis
    print(f"\n{'=' * 80}")
    print("ROOT CAUSE ANALYSIS")
    print(f"{'=' * 80}")
    
    print("\nKey Findings:")
    print("-" * 80)
    
    # Analyze sparse vs dense results
    if sparse_embedding.is_fitted and sparse_results and vector_results_list:
        sparse_top = sparse_results[0]
        dense_top = vector_results_list[0]
        
        print(f"1. SPARSE (BM25) top result: Doc {sparse_top.id}, score={sparse_top.score:.4f}")
        print(f"2. DENSE (Vector) top result: Doc {dense_top.id}, score={dense_top.score:.4f}")
        
        if sparse_top.id != dense_top.id:
            print(f"   [WARNING] SPARSE and DENSE are returning DIFFERENT top results!")
            print(f"   This suggests the issue is in score combination or normalization.")
        
        # Check if sparse found the right document
        sparse_combined = processor._combine_for_embedding(
            sparse_top.screenshot.ocr_text,
            sparse_top.screenshot.visual_description
        )
        has_cat_in_sparse = sparse_combined and ("cat" in sparse_combined.lower() or "kitten" in sparse_combined.lower())
        
        dense_combined = processor._combine_for_embedding(
            dense_top.screenshot.ocr_text,
            dense_top.screenshot.visual_description
        )
        has_cat_in_dense = dense_combined and ("cat" in dense_combined.lower() or "kitten" in dense_combined.lower())
        
        print(f"\n3. Content Analysis:")
        print(f"   Sparse top result contains 'cat'/'kitten': {has_cat_in_sparse}")
        print(f"   Dense top result contains 'cat'/'kitten': {has_cat_in_dense}")
        
        if not has_cat_in_sparse:
            print(f"   [WARNING] SPARSE top result does NOT contain 'cat'/'kitten' - BM25 tokenization issue!")
        if not has_cat_in_dense:
            print(f"   [WARNING] DENSE top result does NOT contain 'cat'/'kitten' - Embedding similarity issue!")
    
    # Check hybrid result
    if results:
        hybrid_top = results[0]
        print(f"\n4. HYBRID top result: Doc {hybrid_top.id}, score={hybrid_top.score:.4f}, type={hybrid_top.search_type}")
        
        hybrid_combined = processor._combine_for_embedding(
            hybrid_top.screenshot.ocr_text,
            hybrid_top.screenshot.visual_description
        )
        has_cat_in_hybrid = hybrid_combined and ("cat" in hybrid_combined.lower() or "kitten" in hybrid_combined.lower())
        print(f"   Hybrid top result contains 'cat'/'kitten': {has_cat_in_hybrid}")
        
        if not has_cat_in_hybrid:
            print(f"   [WARNING] HYBRID top result does NOT contain 'cat'/'kitten' - Score combination issue!")
    
    print(f"\n{'=' * 80}")
    print("DIAGNOSTIC COMPLETE")
    print(f"{'=' * 80}")
    print("\nNext Steps:")
    print("1. Review the scores above to identify which search method is failing")
    print("2. Check the tokenization output to see if BM25 is properly tokenizing keywords")
    print("3. Compare embedding similarities to see if dense search is finding wrong connections")
    print("4. Review hybrid score combination to see if normalization is causing issues")
    print("\nTo see detailed logs during search, enable DEBUG logging level in config.")


if __name__ == "__main__":
    main()
