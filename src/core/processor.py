"""Screenshot processor that orchestrates the ingestion pipeline."""

from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Callable
import os
import asyncio
import hashlib

from ..core.database import Database, Screenshot, compute_file_hash
from ..core.config import ConfigManager, ScanFolder
from ..services.ocr import OCRService
from ..services.vision import VisionService, VisionAPIError
from ..services.embedding import EmbeddingService
from ..services.sparse_embedding import SparseEmbeddingService
from ..services.vector_store import VectorStore


@dataclass
class ProcessingStats:
    """Statistics from a processing run."""
    total_files: int = 0
    new_indexed: int = 0
    updated: int = 0
    skipped: int = 0
    failed: int = 0
    
    @property
    def processed(self) -> int:
        return self.new_indexed + self.updated


@dataclass
class ProcessingProgress:
    """Progress update during processing."""
    current_file: str
    current_index: int
    total_files: int
    status: str  # 'processing', 'skipped', 'failed', 'cancelled'
    error_message: Optional[str] = None


class ScreenshotProcessor:
    """Orchestrates the ingestion pipeline for screenshots."""
    
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
    
    def __init__(
        self,
        config: ConfigManager,
        db: Database,
        vector_store: VectorStore,
        ocr: OCRService,
        vision: VisionService,
        embedding: EmbeddingService,
        sparse_embedding: Optional[SparseEmbeddingService] = None,
    ):
        self.config = config
        self.db = db
        self.vector_store = vector_store
        self.ocr = ocr
        self.vision = vision
        self.embedding = embedding
        self.sparse_embedding = sparse_embedding
    
    def discover_images(self, folders: Optional[list[ScanFolder]] = None) -> list[Path]:
        """Discover all image files in configured folders.
        
        Args:
            folders: Specific folders to scan (uses config if None)
            
        Returns:
            List of image file paths
        """
        folders = folders or self.config.config.scan_folders
        images = []
        
        for folder in folders:
            folder_path = Path(folder.path)
            if not folder_path.exists():
                continue
            
            if folder.include_subfolders:
                for ext in self.IMAGE_EXTENSIONS:
                    images.extend(folder_path.rglob(f"*{ext}"))
            else:
                for ext in self.IMAGE_EXTENSIONS:
                    images.extend(folder_path.glob(f"*{ext}"))
        
        return sorted(set(images))
    
    def check_changes(self, images: list[Path]) -> tuple[list[Path], list[Path], list[Path]]:
        """Check which images are new, changed, or unchanged.
        
        Returns:
            Tuple of (new_images, changed_images, unchanged_images)
        """
        existing = self.db.get_all_paths_and_hashes()
        
        new_images = []
        changed_images = []
        unchanged_images = []
        
        for image_path in images:
            path_str = str(image_path)
            
            if path_str not in existing:
                new_images.append(image_path)
            else:
                current_hash = compute_file_hash(image_path)
                if current_hash != existing[path_str]:
                    changed_images.append(image_path)
                else:
                    unchanged_images.append(image_path)
        
        return new_images, changed_images, unchanged_images
    
    def process_single(
        self,
        image_path: Path,
        force: bool = False,
    ) -> Optional[int]:
        """Process a single image through the pipeline.
        
        All-or-nothing: only saves to database if ALL processing steps succeed.
        If any step fails, nothing is stored so the file can be retried later.
        
        Args:
            image_path: Path to the image file
            force: Force reprocessing even if unchanged
            
        Returns:
            Screenshot ID if successful, None if skipped
            
        Raises:
            Exception if processing fails (so caller knows to skip this file)
        """
        path_str = str(image_path)
        current_hash = compute_file_hash(image_path)
        
        # Check if already indexed
        existing = self.db.get_by_path(path_str)
        if existing and not force:
            if existing.file_hash == current_hash:
                return None  # Unchanged, skip
        
        # Step 1 & 2: Run OCR and Vision in parallel (they're independent)
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            ocr_future = executor.submit(self.ocr.extract_text, image_path)
            vision_future = executor.submit(self.vision.describe, image_path)
            
            # Wait for both to complete
            ocr_text = ocr_future.result()
            visual_desc = vision_future.result()
        
        # Step 3: Combine text for embedding
        combined_text = self._combine_for_embedding(ocr_text, visual_desc)
        
        # Debug logging for text combination
        if combined_text:
            ocr_len = len(ocr_text) if ocr_text else 0
            visual_len = len(visual_desc) if visual_desc else 0
            combined_len = len(combined_text)
            # Log first 200 chars of combined text for debugging
            preview = combined_text[:200].replace('\n', ' ') + ('...' if len(combined_text) > 200 else '')
            print(f"DEBUG EMBEDDING - {image_path.name}: OCR={ocr_len} chars, Visual={visual_len} chars, Combined={combined_len} chars")
            print(f"DEBUG EMBEDDING - Preview: {preview}")
        
        # Step 4: Generate embedding - REQUIRED for storage
        if not combined_text:
            raise ValueError(f"No text content to embed for {image_path}")
        
        # Generate embedding with retry logic
        embedding_vector = self.embedding.embed(combined_text)
        if embedding_vector is None:
            # Try once more before failing
            embedding_vector = self.embedding.embed(combined_text)
            if embedding_vector is None:
                raise RuntimeError(
                    f"Failed to generate embedding for {image_path}. "
                    f"Check that Ollama is running and the embedding model '{self.embedding.model}' is available."
                )
        
        # All processing succeeded - now save everything
        
        # Extract metadata
        app_name, window_title = self._extract_metadata(image_path)
        captured_at = self._get_capture_time(image_path)
        
        # Create database record
        screenshot = Screenshot(
            id=existing.id if existing else None,
            file_path=path_str,
            file_hash=current_hash,
            app_name=app_name,
            window_title=window_title,
            captured_at=captured_at,
            indexed_at=datetime.now(),
            ocr_text=ocr_text,
            visual_description=visual_desc,
        )
        
        if existing:
            self.db.update(screenshot)
            screenshot_id = existing.id
        else:
            screenshot_id = self.db.insert(screenshot)
        
        # Add to vector store
        self.vector_store.add(
            id=screenshot_id,
            vector=embedding_vector,
            file_path=path_str,
            metadata={"app_name": app_name, "window_title": window_title},
        )
        
        # Add to sparse embedding index
        if self.sparse_embedding:
            self.sparse_embedding.add_document(screenshot_id, combined_text)
        
        return screenshot_id
    
    def process_all(
        self,
        folders: Optional[list[ScanFolder]] = None,
        force: bool = False,
        progress_callback: Optional[Callable[[ProcessingProgress], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> ProcessingStats:
        """Process all images in configured folders.
        
        Args:
            folders: Specific folders to process (uses config if None)
            force: Force reprocessing of all images
            progress_callback: Called with progress updates
            cancel_check: Callable that returns True if processing should be cancelled
            
        Returns:
            Processing statistics
        """
        stats = ProcessingStats()
        images = self.discover_images(folders)
        stats.total_files = len(images)
        
        if not force:
            new_images, changed_images, unchanged_images = self.check_changes(images)
            stats.skipped = len(unchanged_images)
            to_process = new_images + changed_images
        else:
            to_process = images
        
        for i, image_path in enumerate(to_process):
            # Check for cancellation
            if cancel_check and cancel_check():
                if progress_callback:
                    progress_callback(ProcessingProgress(
                        current_file="",
                        current_index=i,
                        total_files=len(to_process),
                        status="cancelled",
                    ))
                break
            
            if progress_callback:
                progress_callback(ProcessingProgress(
                    current_file=str(image_path),
                    current_index=i + 1,
                    total_files=len(to_process),
                    status="processing",
                ))
            
            try:
                existing = self.db.get_by_path(str(image_path))
                result = self.process_single(image_path, force=force)
                
                if result is not None:
                    if existing:
                        stats.updated += 1
                    else:
                        stats.new_indexed += 1
                else:
                    stats.skipped += 1
            except VisionAPIError as e:
                # Vision API error - skip this image so it can be reindexed later
                print(f"Vision API error, skipping {image_path}: {e}")
                stats.failed += 1
                
                if progress_callback:
                    progress_callback(ProcessingProgress(
                        current_file=str(image_path),
                        current_index=i + 1,
                        total_files=len(to_process),
                        status="api_error",
                        error_message=str(e),
                    ))
            except Exception as e:
                print(f"Failed to process {image_path}: {e}")
                stats.failed += 1
                
                if progress_callback:
                    progress_callback(ProcessingProgress(
                        current_file=str(image_path),
                        current_index=i + 1,
                        total_files=len(to_process),
                        status="failed",
                        error_message=str(e),
                    ))
        
        return stats
    
    def _combine_for_embedding(
        self,
        ocr_text: Optional[str],
        visual_desc: Optional[str],
    ) -> Optional[str]:
        """Combine OCR and visual description for embedding with weighted combination.
        
        Prioritizes OCR text when substantial (>100 chars), visual description as supplement.
        """
        ocr_clean = ocr_text.strip() if ocr_text and ocr_text.strip() else None
        visual_clean = visual_desc.strip() if visual_desc and visual_desc.strip() else None
        
        # If OCR is substantial, prioritize it
        if ocr_clean and len(ocr_clean) > 100:
            # OCR is primary, visual is supplement
            parts = [f"Text content: {ocr_clean}"]
            if visual_clean:
                parts.append(f"Visual context: {visual_clean}")
        else:
            # Visual is primary or both are short
            parts = []
            if visual_clean:
                parts.append(f"Visual: {visual_clean}")
            if ocr_clean:
                parts.append(f"Text content: {ocr_clean}")
        
        combined = "\n\n".join(parts) if parts else None
        
        # Final validation - ensure we have meaningful content
        if combined and len(combined.strip()) < 5:
            return None
        
        return combined
    
    def _extract_metadata(self, image_path: Path) -> tuple[Optional[str], Optional[str]]:
        """Extract app name and window title from image path/metadata.
        
        For now, this is a simple implementation based on folder structure.
        Could be enhanced to read EXIF data or use OS-specific APIs.
        """
        # Simple heuristic: use parent folder as "app name"
        app_name = image_path.parent.name if image_path.parent.name else None
        window_title = None
        return app_name, window_title
    
    def _get_capture_time(self, image_path: Path) -> Optional[datetime]:
        """Get the capture time of an image."""
        try:
            # Use file modification time as a fallback
            mtime = os.path.getmtime(image_path)
            return datetime.fromtimestamp(mtime)
        except Exception:
            return None
    
    def _chunk_text(self, text: str, max_chunk_size: int = 2000, overlap: int = 200) -> list[str]:
        """Split long text into chunks with overlap for embedding.
        
        Args:
            text: Text to chunk
            max_chunk_size: Maximum characters per chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        if len(text) <= max_chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_chunk_size
            # Try to break at word boundary
            if end < len(text):
                # Look for space or newline near the end
                for i in range(end, max(start + max_chunk_size - 100, start), -1):
                    if text[i] in (' ', '\n', '\t'):
                        end = i + 1
                        break
            
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk.strip())
            
            # Move start position with overlap
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks if chunks else [text[:max_chunk_size]]
    
    # =========================================================================
    # Async Processing Methods (for parallel image processing)
    # =========================================================================
    
    async def process_single_async(
        self,
        image_path: Path,
        force: bool = False,
    ) -> Optional[int]:
        """Async version of process_single with parallel OCR+Vision.
        
        Runs OCR (CPU) and Vision (GPU) concurrently for each image,
        then generates embedding and saves to database.
        
        Args:
            image_path: Path to the image file
            force: Force reprocessing even if unchanged
            
        Returns:
            Screenshot ID if successful, None if skipped
            
        Raises:
            Exception if processing fails
        """
        path_str = str(image_path)
        current_hash = compute_file_hash(image_path)
        
        # Check if already indexed
        existing = self.db.get_by_path(path_str)
        if existing and not force:
            if existing.file_hash == current_hash:
                return None  # Unchanged, skip
        
        # Run OCR (CPU) and Vision (GPU) in parallel
        ocr_task = asyncio.to_thread(self.ocr.extract_text, image_path)
        vision_task = self.vision.describe_async(image_path)
        
        ocr_text, visual_desc = await asyncio.gather(ocr_task, vision_task)
        
        # Handle vision failure
        if visual_desc is None:
            raise VisionAPIError(f"Vision API returned None for {image_path}")
        
        # Combine text for embedding
        combined_text = self._combine_for_embedding(ocr_text, visual_desc)
        
        if not combined_text:
            raise ValueError(f"No text content to embed for {image_path}")
        
        # Generate embedding (async)
        embedding_vector = await self.embedding.embed_async(combined_text)
        if embedding_vector is None:
            raise RuntimeError(f"Failed to generate embedding for {image_path}")
        
        # All processing succeeded - now save everything (sync, fast)
        app_name, window_title = self._extract_metadata(image_path)
        captured_at = self._get_capture_time(image_path)
        
        screenshot = Screenshot(
            id=existing.id if existing else None,
            file_path=path_str,
            file_hash=current_hash,
            app_name=app_name,
            window_title=window_title,
            captured_at=captured_at,
            indexed_at=datetime.now(),
            ocr_text=ocr_text,
            visual_description=visual_desc,
        )
        
        if existing:
            self.db.update(screenshot)
            screenshot_id = existing.id
        else:
            screenshot_id = self.db.insert(screenshot)
        
        # Add to vector store
        self.vector_store.add(
            id=screenshot_id,
            vector=embedding_vector,
            file_path=path_str,
            metadata={"app_name": app_name, "window_title": window_title},
        )
        
        # Add to sparse embedding index
        if self.sparse_embedding:
            self.sparse_embedding.add_document(screenshot_id, combined_text)
        
        return screenshot_id
    
    async def process_all_async(
        self,
        folders: Optional[list[ScanFolder]] = None,
        force: bool = False,
        concurrency: int = 1,
        progress_callback: Optional[Callable[[ProcessingProgress], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> ProcessingStats:
        """Process all images with optimized batch processing.
        
        Uses phased batch processing for maximum efficiency:
        1. Pre-compute all file hashes in parallel
        2. Batch database lookup for all images
        3. Process images in parallel (OCR + Vision)
        4. Batch generate embeddings
        5. Batch write to database, vector store, and sparse embedding
        
        Args:
            folders: Specific folders to process (uses config if None)
            force: Force reprocessing of all images
            concurrency: Maximum number of images to process concurrently
            progress_callback: Called with progress updates
            cancel_check: Callable that returns True if processing should be cancelled
            
        Returns:
            Processing statistics
        """
        stats = ProcessingStats()
        images = self.discover_images(folders)
        stats.total_files = len(images)
        
        print(f"Discovered {len(images)} total images")
        
        if not force:
            new_images, changed_images, unchanged_images = self.check_changes(images)
            stats.skipped = len(unchanged_images)
            to_process = new_images + changed_images
            print(f"To process: {len(new_images)} new, {len(changed_images)} changed, {len(unchanged_images)} unchanged")
        else:
            to_process = images
            print(f"Force mode: processing all {len(to_process)} images")
        
        if not to_process:
            print("No images to process")
            return stats
        
        # Check for cancellation
        if cancel_check and cancel_check():
            if progress_callback:
                progress_callback(ProcessingProgress(
                    current_file="",
                    current_index=0,
                    total_files=len(to_process),
                    status="cancelled",
                ))
            return stats
        
        # Phase 1: Pre-compute all file hashes in parallel
        if progress_callback:
            progress_callback(ProcessingProgress(
                current_file="",
                current_index=0,
                total_files=len(to_process),
                status="processing",
            ))
        
        hash_tasks = [asyncio.to_thread(compute_file_hash, img) for img in to_process]
        file_hashes = await asyncio.gather(*hash_tasks)
        image_hash_map = {str(img): hash_val for img, hash_val in zip(to_process, file_hashes)}
        
        # Phase 2: Batch database lookup for all images
        image_paths = [str(img) for img in to_process]
        existing_map = await self.db.get_by_paths_batch_async(image_paths)
        
        # Filter out unchanged images if not forcing
        if not force:
            filtered_images = []
            filtered_hashes = []
            filtered_existing = {}
            for img, path_str in zip(to_process, image_paths):
                existing = existing_map.get(path_str)
                current_hash = image_hash_map[path_str]
                if existing and existing.file_hash == current_hash:
                    stats.skipped += 1
                else:
                    filtered_images.append(img)
                    filtered_hashes.append(current_hash)
                    if existing:
                        filtered_existing[path_str] = existing
            to_process = filtered_images
            file_hashes = filtered_hashes
            existing_map = filtered_existing
        
        if not to_process:
            return stats
        
        # Phase 3: Process images in parallel (OCR + Vision)
        semaphore = asyncio.Semaphore(concurrency)
        completed_count = 0
        completed_lock = asyncio.Lock()
        
        @dataclass
        class ProcessedImage:
            """Result from processing a single image."""
            image_path: Path
            path_str: str
            file_hash: str
            ocr_text: Optional[str]
            visual_desc: Optional[str]
            existing: Optional[Screenshot]
            error: Optional[Exception] = None
            cancelled: bool = False
        
        processed_images: list[ProcessedImage] = []
        
        async def process_one(image_path: Path, path_str: str, file_hash: str, existing: Optional[Screenshot], index: int):
            nonlocal completed_count
            
            async with semaphore:
                # Check cancellation
                if cancel_check and cancel_check():
                    return ProcessedImage(
                        image_path=image_path,
                        path_str=path_str,
                        file_hash=file_hash,
                        ocr_text=None,
                        visual_desc=None,
                        existing=existing,
                        cancelled=True,
                    )
                
                # Thread-safe increment of completed_count
                async with completed_lock:
                    completed_count += 1
                    current_count = completed_count
                
                if progress_callback:
                    progress_callback(ProcessingProgress(
                        current_file=path_str,
                        current_index=current_count,
                        total_files=len(to_process),
                        status="processing",
                    ))
                
                try:
                    # Run OCR (CPU) and Vision (GPU) in parallel
                    ocr_task = asyncio.to_thread(self.ocr.extract_text, image_path)
                    vision_task = self.vision.describe_async(image_path)
                    
                    ocr_text, visual_desc = await asyncio.gather(ocr_task, vision_task)
                    
                    # Handle vision failure
                    if visual_desc is None:
                        raise VisionAPIError(f"Vision API returned None for {image_path}")
                    
                    return ProcessedImage(
                        image_path=image_path,
                        path_str=path_str,
                        file_hash=file_hash,
                        ocr_text=ocr_text,
                        visual_desc=visual_desc,
                        existing=existing,
                    )
                    
                except VisionAPIError as e:
                    async with completed_lock:
                        stats.failed += 1
                    print(f"Vision API error, skipping {image_path}: {e}")
                    if progress_callback:
                        progress_callback(ProcessingProgress(
                            current_file=path_str,
                            current_index=current_count,
                            total_files=len(to_process),
                            status="api_error",
                            error_message=str(e),
                        ))
                    return ProcessedImage(
                        image_path=image_path,
                        path_str=path_str,
                        file_hash=file_hash,
                        ocr_text=None,
                        visual_desc=None,
                        existing=existing,
                        error=e,
                    )
                    
                except Exception as e:
                    async with completed_lock:
                        stats.failed += 1
                    print(f"Failed to process {image_path}: {e}")
                    import traceback
                    traceback.print_exc()
                    if progress_callback:
                        progress_callback(ProcessingProgress(
                            current_file=path_str,
                            current_index=current_count,
                            total_files=len(to_process),
                            status="failed",
                            error_message=str(e),
                        ))
                    return ProcessedImage(
                        image_path=image_path,
                        path_str=path_str,
                        file_hash=file_hash,
                        ocr_text=None,
                        visual_desc=None,
                        existing=existing,
                        error=e,
                    )
        
        # Process all images in parallel
        # Recreate image_paths after filtering
        image_paths_filtered = [str(img) for img in to_process]
        tasks = [
            process_one(img, path_str, file_hash, existing_map.get(path_str), i)
            for i, (img, path_str, file_hash) in enumerate(zip(to_process, image_paths_filtered, file_hashes))
        ]
        processed_images = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions that weren't caught
        valid_processed_images = []
        for i, result in enumerate(processed_images):
            if isinstance(result, Exception):
                # Unhandled exception - create error entry
                path_str = image_paths_filtered[i] if i < len(image_paths_filtered) else "unknown"
                print(f"Unhandled exception processing image {path_str}: {result}")
                import traceback
                traceback.print_exc()
                stats.failed += 1
                valid_processed_images.append(ProcessedImage(
                    image_path=to_process[i] if i < len(to_process) else Path("unknown"),
                    path_str=path_str,
                    file_hash=file_hashes[i] if i < len(file_hashes) else "",
                    ocr_text=None,
                    visual_desc=None,
                    existing=None,
                    error=result,
                ))
            elif result is None:
                # Should not happen, but handle it
                path_str = image_paths_filtered[i] if i < len(image_paths_filtered) else "unknown"
                print(f"Warning: process_one returned None for {path_str}")
                stats.failed += 1
            else:
                valid_processed_images.append(result)
        
        # Filter out failed/cancelled images - keep only successful ones
        # Note: We can process images with OCR=None as long as we have vision description
        # We can also process images with Vision=None as long as we have OCR text
        # Only fail if BOTH are None or if there's an error
        successful_images = []
        failed_during_processing = []
        
        for p in valid_processed_images:
            if p.cancelled:
                failed_during_processing.append((p.path_str, "cancelled"))
            elif p.error is not None:
                failed_during_processing.append((p.path_str, f"error: {p.error}"))
            elif p.ocr_text is None and p.visual_desc is None:
                # Both OCR and Vision failed - cannot process
                failed_during_processing.append((p.path_str, "Both OCR and Vision returned None"))
            elif p.ocr_text is None:
                # OCR failed but Vision succeeded - we can still process
                print(f"Warning: OCR returned None for {Path(p.path_str).name}, but Vision succeeded - will use Vision only")
                successful_images.append(p)
            elif p.visual_desc is None:
                # Vision failed but OCR succeeded - we can still process
                print(f"Warning: Vision returned None for {Path(p.path_str).name}, but OCR succeeded - will use OCR only")
                successful_images.append(p)
            else:
                # Both succeeded
                successful_images.append(p)
        
        # Log detailed failure information
        if failed_during_processing:
            print(f"\n=== FAILED IMAGES DURING OCR/VISION PROCESSING ===")
            import sys
            sys.stdout.flush()
            for path, reason in failed_during_processing:
                filename = Path(path).name if path else "unknown"
                print(f"  - {filename}: {reason}")
                sys.stdout.flush()
            print(f"Total failed: {len(failed_during_processing)}\n")
            sys.stdout.flush()
        else:
            print(f"\nNo images failed during OCR/Vision processing\n")
            import sys
            sys.stdout.flush()
        
        if not successful_images:
            return stats
        
        # Phase 4: Batch generate embeddings
        if progress_callback:
            progress_callback(ProcessingProgress(
                current_file="",
                current_index=len(successful_images),
                total_files=len(to_process),
                status="processing",
            ))
        
        # Combine texts for embedding and filter out images with no text content
        # Also chunk very long texts if needed
        valid_images = []
        combined_texts = []
        failed_text_combination = []
        
        for p in successful_images:
            combined = self._combine_for_embedding(p.ocr_text, p.visual_desc)
            if combined:
                # Chunk if text is very long (>2000 chars)
                if len(combined) > 2000:
                    chunks = self._chunk_text(combined, max_chunk_size=2000)
                    # Use first chunk for embedding (most important content)
                    combined = chunks[0] if chunks else combined
                valid_images.append(p)
                combined_texts.append(combined)
            else:
                stats.failed += 1
                ocr_len = len(p.ocr_text) if p.ocr_text else 0
                vision_len = len(p.visual_desc) if p.visual_desc else 0
                failed_text_combination.append((p.path_str, f"OCR: {ocr_len} chars, Vision: {vision_len} chars"))
        
        if failed_text_combination:
            print(f"\n=== FAILED IMAGES DURING TEXT COMBINATION ===")
            for path, reason in failed_text_combination:
                print(f"  - {Path(path).name}: {reason}")
            print(f"Total failed: {len(failed_text_combination)}\n")
        
        if not valid_images:
            return stats
        
        # Batch generate embeddings
        embedding_vectors = await self.embedding.embed_batch_async(combined_texts)
        
        # Filter out images with failed embeddings
        final_images = []
        final_embeddings = []
        failed_embeddings = []
        
        for p, emb in zip(valid_images, embedding_vectors):
            if emb is not None:
                final_images.append(p)
                final_embeddings.append(emb)
            else:
                stats.failed += 1
                failed_embeddings.append(p.path_str)
        
        if failed_embeddings:
            print(f"\n=== FAILED IMAGES DURING EMBEDDING GENERATION ===")
            for path in failed_embeddings:
                print(f"  - {Path(path).name}: embedding returned None")
            print(f"Total failed: {len(failed_embeddings)}\n")
        
        if not final_images:
            return stats
        
        # Phase 5: Batch write to database, vector store, and sparse embedding
        if progress_callback:
            progress_callback(ProcessingProgress(
                current_file="",
                current_index=len(final_images),
                total_files=len(to_process),
                status="processing",
            ))
        
        # Prepare database records
        screenshots_to_insert = []
        screenshots_to_update = []
        vector_store_data = []
        sparse_embedding_data = []
        
        # Extract metadata in parallel for all images
        metadata_tasks = [
            asyncio.to_thread(self._extract_metadata, p.image_path)
            for p in final_images
        ]
        capture_time_tasks = [
            asyncio.to_thread(self._get_capture_time, p.image_path)
            for p in final_images
        ]
        metadata_results = await asyncio.gather(*metadata_tasks)
        capture_time_results = await asyncio.gather(*capture_time_tasks)
        
        for p, emb, (app_name, window_title), captured_at in zip(
            final_images, final_embeddings, metadata_results, capture_time_results
        ):
            
            screenshot = Screenshot(
                id=p.existing.id if p.existing else None,
                file_path=p.path_str,
                file_hash=p.file_hash,
                app_name=app_name,
                window_title=window_title,
                captured_at=captured_at,
                indexed_at=datetime.now(),
                ocr_text=p.ocr_text,
                visual_description=p.visual_desc,
            )
            
            if p.existing:
                screenshots_to_update.append(screenshot)
            else:
                screenshots_to_insert.append(screenshot)
            
            # Prepare vector store data
            vector_store_data.append({
                'id': p.existing.id if p.existing else None,  # Will be set after insert
                'vector': emb,
                'file_path': p.path_str,
                'metadata': {"app_name": app_name, "window_title": window_title},
            })
            
            # Prepare sparse embedding data
            combined = self._combine_for_embedding(p.ocr_text, p.visual_desc)
            if combined:
                sparse_embedding_data.append({
                    'id': p.existing.id if p.existing else None,  # Will be set after insert
                    'text': combined,
                })
        
        # Batch insert new records
        if screenshots_to_insert:
            insert_tasks = [self.db.insert_async(s) for s in screenshots_to_insert]
            new_ids = await asyncio.gather(*insert_tasks)
            
            # Update vector store data with new IDs
            for i, new_id in enumerate(new_ids):
                idx = len(screenshots_to_update) + i
                if idx < len(vector_store_data):
                    vector_store_data[idx]['id'] = new_id
                if idx < len(sparse_embedding_data):
                    sparse_embedding_data[idx]['id'] = new_id
        
        # Batch update existing records
        if screenshots_to_update:
            update_tasks = [self.db.update_async(s) for s in screenshots_to_update]
            await asyncio.gather(*update_tasks)
        
        # Batch write to vector store
        if vector_store_data:
            ids = [d['id'] for d in vector_store_data if d['id'] is not None]
            vectors = [d['vector'] for d in vector_store_data if d['id'] is not None]
            file_paths = [d['file_path'] for d in vector_store_data if d['id'] is not None]
            metadata_list = [d['metadata'] for d in vector_store_data if d['id'] is not None]
            
            if ids:
                await asyncio.to_thread(
                    self.vector_store.add_batch,
                    ids, vectors, file_paths, metadata_list
                )
        
        # Batch write to sparse embedding
        if sparse_embedding_data and self.sparse_embedding:
            documents = [(d['id'], d['text']) for d in sparse_embedding_data if d['id'] is not None]
            if documents:
                await asyncio.to_thread(
                    self.sparse_embedding.add_documents_batch,
                    documents
                )
        
        # Update stats
        stats.new_indexed = len(screenshots_to_insert)
        stats.updated = len(screenshots_to_update)
        
        # Final summary with breakdown
        total_failed = len(failed_during_processing) + len(failed_text_combination) + len(failed_embeddings)
        print(f"\n=== FINAL PROCESSING SUMMARY ===")
        print(f"Total discovered: {stats.total_files}")
        print(f"Total processed: {len(to_process)}")
        print(f"Successfully indexed: {len(final_images)}")
        print(f"  - New: {stats.new_indexed}")
        print(f"  - Updated: {stats.updated}")
        print(f"  - Skipped (unchanged): {stats.skipped}")
        print(f"Failed: {total_failed}")
        print(f"  - During OCR/Vision: {len(failed_during_processing)}")
        print(f"  - During text combination: {len(failed_text_combination)}")
        print(f"  - During embedding: {len(failed_embeddings)}")
        print(f"Stats.failed counter: {stats.failed}")
        print("=" * 40)
        
        if progress_callback:
            progress_callback(ProcessingProgress(
                current_file="",
                current_index=len(to_process),
                total_files=len(to_process),
                status="done",
            ))
        
        return stats
