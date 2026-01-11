"""Helper script to locate and copy OneOCR DLL files."""

import os
import shutil
from pathlib import Path
from typing import Optional, Tuple


def find_oneocr_dlls() -> Optional[Tuple[Path, Path, Path]]:
    """Search for OneOCR DLL files in common Windows locations.
    
    Returns:
        Tuple of (oneocr.dll path, oneocr.onemodel path, onnxruntime.dll path) if found,
        None otherwise
    """
    import subprocess
    
    search_paths = [
        Path("C:/Program Files/WindowsApps"),
        Path("C:/Program Files (x86)/WindowsApps"),
        Path(os.path.expanduser("~/AppData/Local/Microsoft/WindowsApps")),
        Path("C:/Program Files"),
        Path("C:/Program Files (x86)"),
    ]
    
    oneocr_dll = None
    oneocr_model = None
    onnxruntime_dll = None
    
    # Try using PowerShell to search (more reliable for WindowsApps)
    if os.name == 'nt':
        try:
            # Use PowerShell to search WindowsApps (bypasses permission issues)
            ps_script = """
            Get-ChildItem -Path "C:\\Program Files\\WindowsApps" -Recurse -Filter "oneocr.dll" -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
            """
            result = subprocess.run(
                ["powershell", "-Command", ps_script],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0 and result.stdout.strip():
                dll_path = Path(result.stdout.strip())
                if dll_path.exists():
                    oneocr_dll = dll_path
                    # Assume other files are in the same directory
                    parent_dir = dll_path.parent
                    model_path = parent_dir / "oneocr.onemodel"
                    onnx_path = parent_dir / "onnxruntime.dll"
                    if model_path.exists():
                        oneocr_model = model_path
                    if onnx_path.exists():
                        onnxruntime_dll = onnx_path
        except Exception:
            pass  # Fall back to regular search
    
    # If PowerShell search didn't find all files, try regular search
    if not (oneocr_dll and oneocr_model and onnxruntime_dll):
        for search_path in search_paths:
            if not search_path.exists():
                continue
            
            try:
                # Search for oneocr.dll
                if oneocr_dll is None:
                    for dll_file in search_path.rglob("oneocr.dll"):
                        oneocr_dll = dll_file
                        # Check same directory for other files
                        parent_dir = dll_file.parent
                        if (parent_dir / "oneocr.onemodel").exists():
                            oneocr_model = parent_dir / "oneocr.onemodel"
                        if (parent_dir / "onnxruntime.dll").exists():
                            onnxruntime_dll = parent_dir / "onnxruntime.dll"
                        break
                
                # If we found DLL, try to find others in same directory
                if oneocr_dll and (oneocr_model is None or onnxruntime_dll is None):
                    parent_dir = oneocr_dll.parent
                    if oneocr_model is None:
                        model_path = parent_dir / "oneocr.onemodel"
                        if model_path.exists():
                            oneocr_model = model_path
                    if onnxruntime_dll is None:
                        onnx_path = parent_dir / "onnxruntime.dll"
                        if onnx_path.exists():
                            onnxruntime_dll = onnx_path
                
                # Search separately if not found in same directory
                if oneocr_model is None:
                    for model_file in search_path.rglob("oneocr.onemodel"):
                        oneocr_model = model_file
                        break
                
                if onnxruntime_dll is None:
                    for onnx_file in search_path.rglob("onnxruntime.dll"):
                        # Make sure it's not a different onnxruntime (check if near oneocr)
                        if "oneocr" in str(onnx_file.parent).lower() or oneocr_dll:
                            onnxruntime_dll = onnx_file
                            break
                
                # If we found all three, we're done
                if oneocr_dll and oneocr_model and onnxruntime_dll:
                    break
            except (PermissionError, OSError) as e:
                # Skip paths we can't access
                continue
    
    if oneocr_dll and oneocr_model and onnxruntime_dll:
        return (oneocr_dll, oneocr_model, onnxruntime_dll)
    
    return None


def setup_oneocr_dlls(target_dir: Optional[Path] = None, manual_paths: Optional[Tuple[Path, Path, Path]] = None) -> bool:
    """Find and copy OneOCR DLL files to the target directory.
    
    Args:
        target_dir: Directory to copy DLLs to (default: ~/.config/oneocr)
        manual_paths: Optional tuple of (oneocr.dll, oneocr.onemodel, onnxruntime.dll) paths
        
    Returns:
        True if setup was successful, False otherwise
    """
    if target_dir is None:
        target_dir = Path.home() / ".config" / "oneocr"
    
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Use manual paths if provided, otherwise find them
    if manual_paths:
        dll_paths = manual_paths
    else:
        dll_paths = find_oneocr_dlls()
        if dll_paths is None:
            return False
    
    oneocr_dll, oneocr_model, onnxruntime_dll = dll_paths
    
    try:
        # Copy files to target directory
        shutil.copy2(oneocr_dll, target_dir / "oneocr.dll")
        shutil.copy2(oneocr_model, target_dir / "oneocr.onemodel")
        shutil.copy2(onnxruntime_dll, target_dir / "onnxruntime.dll")
        
        return True
    except Exception as e:
        print(f"Error copying OneOCR DLLs: {e}")
        return False


if __name__ == "__main__":
    # Run setup if called directly
    if setup_oneocr_dlls():
        print("OneOCR DLLs successfully copied!")
    else:
        print("Failed to find or copy OneOCR DLLs.")
        print("Please ensure OneOCR is installed and the DLLs are accessible.")
