"""OCR service for text extraction from screenshots using OneOCR."""

import sys
import os
from pathlib import Path
from typing import Optional


class OCRService:
    """OCR text extraction using OneOCR (Windows native, performant)."""
    
    def __init__(self):
        self._ocr = None
        self._backend = "none"
        self._oneocr_available = False
        self._init_ocr()
    
    def _init_ocr(self) -> None:
        """Initialize the OCR backend."""
        if sys.platform != "win32":
            print("Warning: OneOCR is only available on Windows.")
            self._backend = "none"
            return
        
        try:
            import oneocr
            self._oneocr_available = True
        except ImportError:
            print("Warning: oneocr not installed. OCR will be disabled.")
            print("Install with: uv add oneocr")
            self._backend = "none"
            return
        
        # Try to initialize OneOCR (suppress DLL errors during initialization)
        import io
        from contextlib import redirect_stderr
        
        # Suppress stderr to hide DLL initialization errors
        stderr_capture = io.StringIO()
        try:
            with redirect_stderr(stderr_capture):
                self._ocr = oneocr.OcrEngine()
                self._backend = "oneocr"
        except Exception as e:
            error_msg = str(e)
            error_output = stderr_capture.getvalue()
            
            if "DLL" in error_msg or "dll" in error_msg.lower() or "DLL" in error_output:
                # Try to setup DLLs
                if self._setup_oneocr_dlls():
                    # Try again after setup
                    stderr_capture2 = io.StringIO()
                    try:
                        with redirect_stderr(stderr_capture2):
                            self._ocr = oneocr.OcrEngine()
                        self._backend = "oneocr"
                        print("OneOCR initialized successfully after DLL setup!")
                    except Exception as e2:
                        print(f"Warning: Failed to initialize OneOCR after DLL setup: {e2}")
                        self._backend = "none"
                else:
                    # Show setup instructions only once
                    print("\n" + "="*70)
                    print("OneOCR DLL Setup Required")
                    print("="*70)
                    print("OneOCR DLL files were not found automatically.")
                    print("\nTo setup OneOCR DLLs:")
                    print("  1. Run the find script: powershell -ExecutionPolicy Bypass -File scripts/find_oneocr_dlls.ps1")
                    print("  2. Or run setup: uv run python scripts/setup_oneocr_dlls.py")
                    print("  3. Or manually: uv run python scripts/setup_oneocr_dlls.py <dll> <model> <onnx>")
                    print("="*70 + "\n")
                    self._backend = "none"
            else:
                print(f"Warning: Failed to initialize OneOCR: {e}")
                self._backend = "none"
    
    def _setup_oneocr_dlls(self) -> bool:
        """Setup OneOCR DLL files by copying them from WindowsApps if needed."""
        try:
            from .ocr_setup import setup_oneocr_dlls
            return setup_oneocr_dlls()
        except Exception as e:
            # If setup module fails, try direct search
            return self._find_and_copy_dlls()
    
    def _find_and_copy_dlls(self) -> bool:
        """Find and copy OneOCR DLL files directly."""
        import shutil
        
        target_dir = Path.home() / ".config" / "oneocr"
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if DLLs already exist
        if (target_dir / "oneocr.dll").exists() and \
           (target_dir / "oneocr.onemodel").exists() and \
           (target_dir / "onnxruntime.dll").exists():
            return True
        
        # Search paths
        search_paths = [
            Path("C:/Program Files/WindowsApps"),
            Path("C:/Program Files (x86)/WindowsApps"),
            Path(os.path.expanduser("~/AppData/Local/Microsoft/WindowsApps")),
        ]
        
        oneocr_dll = None
        oneocr_model = None
        onnxruntime_dll = None
        
        for search_path in search_paths:
            if not search_path.exists():
                continue
            
            try:
                if oneocr_dll is None:
                    for dll_file in search_path.rglob("oneocr.dll"):
                        oneocr_dll = dll_file
                        break
                
                if oneocr_model is None:
                    for model_file in search_path.rglob("oneocr.onemodel"):
                        oneocr_model = model_file
                        break
                
                if onnxruntime_dll is None:
                    for onnx_file in search_path.rglob("onnxruntime.dll"):
                        onnxruntime_dll = onnx_file
                        break
                
                if oneocr_dll and oneocr_model and onnxruntime_dll:
                    break
            except (PermissionError, OSError):
                continue
        
        if oneocr_dll and oneocr_model and onnxruntime_dll:
            try:
                shutil.copy2(oneocr_dll, target_dir / "oneocr.dll")
                shutil.copy2(oneocr_model, target_dir / "oneocr.onemodel")
                shutil.copy2(onnxruntime_dll, target_dir / "onnxruntime.dll")
                return True
            except Exception as e:
                print(f"Error copying OneOCR DLLs: {e}")
                return False
        
        return False
    
    def extract_text(self, image_path: Path) -> Optional[str]:
        """Extract text from an image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Extracted text, or None if extraction failed
        """
        if not image_path.exists():
            return None
        
        if self._backend == "oneocr" and self._ocr is not None:
            return self._extract_with_oneocr(image_path)
        
        return None
    
    def _extract_with_oneocr(self, image_path: Path) -> Optional[str]:
        """Extract text using OneOCR (Windows native OCR)."""
        try:
            from PIL import Image
            
            img = Image.open(str(image_path))
            result = self._ocr.recognize_pil(img)
            
            # oneocr returns a dict with 'text' key
            if result and isinstance(result, dict):
                text = result.get('text', None)
                return text.strip() if text else None
            
            return None
        except Exception as e:
            print(f"OCR extraction failed for {image_path}: {e}")
            return None
    
    @property
    def backend_name(self) -> str:
        """Get the name of the active OCR backend."""
        return self._backend

