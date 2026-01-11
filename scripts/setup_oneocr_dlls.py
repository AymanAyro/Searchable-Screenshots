"""Standalone script to setup OneOCR DLL files.

This script searches for OneOCR DLL files in WindowsApps and copies them
to the required location (~/.config/oneocr/).

Usage:
    python scripts/setup_oneocr_dlls.py
    python scripts/setup_oneocr_dlls.py "C:/path/to/oneocr.dll" "C:/path/to/oneocr.onemodel" "C:/path/to/onnxruntime.dll"
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.ocr_setup import setup_oneocr_dlls, find_oneocr_dlls


def main():
    """Main function to setup OneOCR DLLs."""
    # Check for manual paths
    if len(sys.argv) == 4:
        dll_path = Path(sys.argv[1])
        model_path = Path(sys.argv[2])
        onnx_path = Path(sys.argv[3])
        
        if not dll_path.exists():
            print(f"[ERROR] oneocr.dll not found at: {dll_path}")
            return 1
        if not model_path.exists():
            print(f"[ERROR] oneocr.onemodel not found at: {model_path}")
            return 1
        if not onnx_path.exists():
            print(f"[ERROR] onnxruntime.dll not found at: {onnx_path}")
            return 1
        
        print(f"[INFO] Using manual paths:")
        print(f"  - oneocr.dll: {dll_path}")
        print(f"  - oneocr.onemodel: {model_path}")
        print(f"  - onnxruntime.dll: {onnx_path}")
        
        if setup_oneocr_dlls(manual_paths=(dll_path, model_path, onnx_path)):
            print("\n[SUCCESS] OneOCR DLLs successfully copied!")
            return 0
        else:
            print("\n[ERROR] Failed to copy DLL files.")
            return 1
    
    print("Searching for OneOCR DLL files...")
    
    dll_paths = find_oneocr_dlls()
    if dll_paths is None:
        print("\n[ERROR] Could not find OneOCR DLL files.")
        print("\nSearched in:")
        print("  - C:/Program Files/WindowsApps")
        print("  - C:/Program Files (x86)/WindowsApps")
        print("  - ~/AppData/Local/Microsoft/WindowsApps")
        print("\nTo manually specify DLL paths, run:")
        print("  python scripts/setup_oneocr_dlls.py <oneocr.dll> <oneocr.onemodel> <onnxruntime.dll>")
        print("\nOr search manually using PowerShell:")
        print('  Get-ChildItem -Path "C:\\Program Files\\WindowsApps" -Recurse -Filter "oneocr.dll"')
        return 1
    
    oneocr_dll, oneocr_model, onnxruntime_dll = dll_paths
    print(f"\n[SUCCESS] Found DLL files:")
    print(f"  - oneocr.dll: {oneocr_dll}")
    print(f"  - oneocr.onemodel: {oneocr_model}")
    print(f"  - onnxruntime.dll: {onnxruntime_dll}")
    
    print("\nCopying DLL files to ~/.config/oneocr/...")
    if setup_oneocr_dlls():
        print("\n[SUCCESS] OneOCR DLLs successfully copied!")
        print("You can now use OneOCR in the application.")
        return 0
    else:
        print("\n[ERROR] Failed to copy DLL files.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
