# Searchable Screenshots

A Python application that indexes and enables fast searching of your screenshots using OCR, vision models, and vector embeddings.

⚠️ Under active development. Use at your own risk.

## Features

- **Folder scanning** – Recursively scan selected folders for image files.
- **OCR extraction** – Pull text from screenshots using Tesseract OCR (robust, fast, cross-platform).
- **Vision description** – Generate concise image captions using Ollama vision models (e.g., Moondream).
- **Embeddings** – Create vector embeddings using LangChain and Ollama.
- **Hybrid search** – Combine BM25 keyword search and semantic vector search for powerful queries.
- **Configurable parallelism** – Adjust concurrency for indexing performance.
- **LangChain integration** – Built on LangChain framework for GenAI operations.

## Installation

This project uses `uv` package manager, install it at [https://docs.astral.sh/uv](https://docs.astral.sh/uv)

```bash
# Clone the repository
git clone https://github.com/Creative-Geek/Searchable-Screenshots.git
cd Searchable-Screenshots

# Install dependencies and create virtual environment
uv sync
```

### Prerequisites

1. **Ollama** must be running locally with the following models:
   - Vision model (default: `gemma3:4b`) for image descriptions
   - Embedding model (default: `nomic-embed-text:latest`) for vector embeddings

   Default models can be changed in Settings. To install other models:
   ```bash
   ollama pull gemma3:4b
   ollama pull nomic-embed-text:latest
   ```

2. **OneOCR** (for text extraction on Windows):
   - OneOCR is used for OCR text extraction
   - **First, ensure OneOCR is installed** (via Microsoft Store or Windows Apps)
   - The DLL files need to be copied from WindowsApps to `~/.config/oneocr/`
   
   **Setup Steps:**
   
   a) Find the DLL files (run PowerShell as Administrator for best results):
     ```powershell
     powershell -ExecutionPolicy Bypass -File scripts/find_oneocr_dlls.ps1
     ```
   
   b) If found, copy them automatically:
     ```powershell
     uv run python scripts/setup_oneocr_dlls.py
     ```
   
   c) Or manually specify paths:
     ```powershell
     uv run python scripts/setup_oneocr_dlls.py "C:\path\to\oneocr.dll" "C:\path\to\oneocr.onemodel" "C:\path\to\onnxruntime.dll"
     ```
   
   - Required files: `oneocr.dll`, `oneocr.onemodel`, `onnxruntime.dll`
   - **Note**: The application will run without OCR if DLLs aren't found, but OCR text extraction will be disabled.

### Optional: Reranker Support

For reranking search results (improves result quality), install sentence-transformers:
```bash
uv add sentence-transformers
```

Note: This will also install torch as a dependency. Reranker is optional and the system works without it.

## Usage

### Run the GUI

```bash
uv run main.py
```

The application will open a window where you can add folders, configure settings, and start indexing.

### Index from the command line

```bash
uv run src.core.processor --folder "C:\path\to\screenshots"
```

This will process the images and populate the local database.

## Configuration

Open **Settings** in the GUI to adjust:

- Ollama URL (default: `http://localhost:11434`)
- Vision model name (default: `gemma3:4b`)
- Embedding model name (default: `nomic-embed-text:latest`)
- Hybrid search weight (0.0 = sparse only, 1.0 = dense only, default = 0.5)
- Parallel processing count (default = 1)

## Testing

### Quick Test

1. **Start Ollama** (if not already running):
   ```bash
   ollama serve
   ```

2. **Run the application**:
   ```bash
   uv run main.py
   ```

3. **In the GUI**:
   - Click "Add Folder" and select a folder with screenshots
   - Click "Index Now" to process screenshots
   - Use the search bar to test search functionality
   - Try different search queries:
     - Exact match: `"specific text"` (quoted for FTS5 search)
     - Semantic search: `terminal error` (unquoted for hybrid search)

### Test Hybrid Search

Hybrid search combines:
- **Sparse (BM25)**: Keyword matching for exact terms
- **Dense (Vector)**: Semantic similarity for concepts

To test:
1. Index some screenshots with varied content
2. Search with:
   - Quoted queries (`"exact phrase"`) → Uses FTS5 only
   - Unquoted queries (`error message`) → Uses hybrid search
3. Adjust hybrid weight in Settings to see the difference:
   - Lower weight (0.0-0.3): More keyword-focused
   - Higher weight (0.7-1.0): More semantic/concept-focused

## Contributing

Feel free to open issues or submit pull requests. Please follow the existing code style and run the test suite before submitting.

## License

This project is licensed under the MIT License.
