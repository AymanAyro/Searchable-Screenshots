"""Vision service for visual description generation using Moondream via Ollama with LangChain."""

import base64
from pathlib import Path
from typing import Optional
from PIL import Image
import io
import httpx
import time
import asyncio

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from ..core.logging import get_logger
from ..core.retry import retry_with_backoff, CircuitBreaker


class VisionAPIError(Exception):
    """Raised when the Vision API encounters an error (network, server, etc)."""
    pass


class VisionService:
    """Generate visual descriptions of screenshots using Moondream via LangChain."""
    
    DEFAULT_PROMPT = (
         """
### SYSTEM ROLE
You are a Computer Vision Indexer. Your job is to extract searchable data from images.

### ANALYSIS STRATEGY
1.  **Classify:** Determine if the image is a Screenshot, Meme, Photograph, or Graphic.
2.  **Transcribe:** If text is present, extract the key phrases exactly.
3.  **Describe:** Write a description based on the classification.

### OUTPUT FORMAT
Strictly follow this Markdown structure:

## 1. Category & Type
*   **Type:** [Select one: UI/Screenshot, Meme, Photograph, Movie Scene, Text Graphic]
*   **Quality:** [e.g., High Res, Blurry, Grainy, Cropped, Low Light]

## 2. Content Recognition
*   **Main Subject:** [e.g., Person, Terminal Window, Airplane, Cartoon Character]
*   **Visible Text:** [Transcribe important text exactly. If Arabic or foreign language, note it. If none, write "None".]
*   **Key Objects:** [List physical items, e.g., "ThinkPad Keyboard", "Yellow Glove", "Notifications Badge"]

## 3. Semantic Description
[Write 2-3 sentences. **Crucial:**]
*   *If Meme:* Explain the joke or metaphor (e.g., "Comparing a massive plane to a tiny bike").
*   *If Screenshot:* Describe the app context (e.g., "Docker terminal output showing disk usage").
*   *If Photo:* Describe the action and mood.

## 4. Search Keywords
[Generate 20 keywords. Mix visual tags, text terms, and abstract concepts.]
*   *Format:* tag1, tag2, tag3...

### INPUT IMAGE
[Image]
        """
    )
    
    FAST_PROMPT = (
        """Describe this screenshot in 1-2 sentences. Include: what application or website is shown, 
main content, and any visible text. Be concise and searchable."""
    )
    
    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "gemma3:4b",
        timeout: float = 120.0,
        prompt_style: str = "detailed",
        max_retries: int = 3,
        retry_backoff_factor: float = 2.0,
        retry_initial_delay: float = 1.0,
    ):
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.prompt_style = prompt_style  # "fast" or "detailed"
        self.max_retries = max_retries
        self.retry_backoff_factor = retry_backoff_factor
        self.retry_initial_delay = retry_initial_delay
        
        self.logger = get_logger(__name__)
        self.circuit_breaker = CircuitBreaker(failure_threshold=5, timeout=60.0)
        self._last_success_time: Optional[float] = None
        
        # Validate model name format (Ollama models typically use format "name:tag")
        if not model or not model.strip():
            raise ValueError("Model name cannot be empty")
        
        # Initialize LangChain ChatOllama
        # Disable reasoning for faster image descriptions (reasoning models are much slower)
        # Note: If your model doesn't support this parameter, use a non-reasoning variant instead
        # (e.g., use "llama-3.2:3b" instead of "llama-3.2:3b-reasoning")
        try:
            self._llm = ChatOllama(
                base_url=self.ollama_url,
                model=self.model,
                timeout=timeout,
                temperature=0.3,
                num_ctx=8096,
                model_kwargs={
                    "reasoning": False,  # Disable reasoning for faster responses (if model supports it)
                },
            )
            self.logger.info(f"Initialized vision service with model '{model}'")
        except Exception as e:
            self.logger.error(f"Failed to initialize vision model '{model}': {e}")
            raise ValueError(
                f"Failed to initialize vision model '{model}'. "
                f"Please verify the model exists in Ollama (run 'ollama list' to see available models). "
                f"Error: {e}"
            ) from e
    
    def _detect_image_complexity(self, img: Image.Image) -> str:
        """Detect if image is text-heavy or visual-heavy.
        
        Args:
            img: PIL Image object
            
        Returns:
            "text_heavy" or "visual_heavy"
        """
        # Simple heuristic: check image size and color variance
        # Text-heavy images often have high contrast and many edges
        # For now, use a simple size-based heuristic
        width, height = img.size
        total_pixels = width * height
        
        # Large images with many pixels are likely more complex
        if total_pixels > 2_000_000:  # > ~1414x1414
            return "visual_heavy"
        else:
            return "text_heavy"
    
    def _adaptive_encode_image(
        self, 
        image_path: Path, 
        max_size: tuple[int, int] = (1920, 1920),
        adaptive: bool = True,
    ) -> Optional[str]:
        """Read and encode image with adaptive quality/resize based on complexity.
        
        Args:
            image_path: Path to the image file
            max_size: Maximum (width, height) to resize large images to
            adaptive: Whether to use adaptive quality/resize based on image complexity
        """
        try:
            with Image.open(image_path) as img:
                # Detect complexity if adaptive mode
                if adaptive:
                    complexity = self._detect_image_complexity(img)
                    
                    # Adjust quality and max_size based on complexity
                    if complexity == "text_heavy":
                        # Text-heavy: use lower quality (75-80) and smaller size for speed
                        quality = 75
                        adaptive_max_size = (1280, 1280)
                    else:
                        # Visual-heavy: use higher quality (85-90) and larger size
                        quality = 85
                        adaptive_max_size = max_size
                else:
                    quality = 85
                    adaptive_max_size = max_size
                
                # Resize if image is very large
                if img.size[0] > adaptive_max_size[0] or img.size[1] > adaptive_max_size[1]:
                    img.thumbnail(adaptive_max_size, Image.Resampling.LANCZOS)
                
                # Convert to RGB (handles RGBA, P, etc.)
                if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                    bg = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    bg.paste(img, mask=img.split()[3])
                    img = bg
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG", quality=quality)
                return base64.b64encode(buffered.getvalue()).decode("utf-8")
        except Exception as e:
            self.logger.error(f"Error encoding image {image_path}: {e}")
            return None
    
    def _encode_image(self, image_path: Path, max_size: tuple[int, int] = (1920, 1920)) -> Optional[str]:
        """Read and encode image, ensuring it's in a format Ollama accepts.
        
        Args:
            image_path: Path to the image file
            max_size: Maximum (width, height) to resize large images to (speeds up processing)
        """
        return self._adaptive_encode_image(image_path, max_size, adaptive=True)

    def describe(self, image_path: Path, prompt: Optional[str] = None) -> Optional[str]:
        """Generate a visual description of an image.
        
        Args:
            image_path: Path to the image file
            prompt: Optional custom prompt (uses DEFAULT_PROMPT or FAST_PROMPT based on prompt_style)
            
        Returns:
            Visual description text, or None if generation failed
        """
        if not image_path.exists():
            return None
        
        image_data = self._encode_image(image_path)
        if not image_data:
            return None
        
        # Use prompt style if no custom prompt provided
        if prompt is None:
            if self.prompt_style == "fast":
                prompt_text = self.FAST_PROMPT
            else:
                prompt_text = self.DEFAULT_PROMPT
        else:
            prompt_text = prompt
        
        @retry_with_backoff(
            max_retries=self.max_retries,
            initial_delay=self.retry_initial_delay,
            backoff_factor=self.retry_backoff_factor,
            exceptions=(Exception,),
            logger=self.logger,
        )
        def _invoke_llm():
            messages = [
                SystemMessage(content=prompt_text),
                HumanMessage(
                    content=[
                        {
                            "type": "text",
                            "text": "",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        }
                    ]
                )
            ]
            return self._llm.invoke(messages)
        
        try:
            # Use circuit breaker for API calls
            response = self.circuit_breaker.call(_invoke_llm)
            result = response.content.strip() if hasattr(response, 'content') else str(response).strip()
            self._last_success_time = time.time()
            self.logger.debug(f"Generated description for {image_path.name} ({len(result)} chars)")
            return result
        except Exception as e:
            # Check if it's a network/server error or model error
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ['connection', 'timeout', 'server', 'http', 'network']):
                raise VisionAPIError(f"Vision API error for {image_path}: {e}") from e
            # Check for model-related errors - try direct API call as fallback
            if 'model is required' in error_str or ('model' in error_str and '400' in error_str):
                # Fallback: Try direct Ollama API call
                try:
                    return self._describe_direct_api(image_path, prompt_text, image_data)
                except Exception as fallback_error:
                    raise VisionAPIError(
                        f"Model '{self.model}' error. LangChain failed: {e}. "
                        f"Direct API fallback also failed: {fallback_error}. "
                        f"Please verify the model exists and supports vision (run 'ollama show {self.model}')."
                    ) from fallback_error
            self.logger.error(f"Vision service failed for {image_path}: {e}")
            return None
    
    async def describe_async(self, image_path: Path, prompt: Optional[str] = None) -> Optional[str]:
        """Async version of describe()."""
        import asyncio
        
        if not image_path.exists():
            return None
        
        # Run image encoding in a thread to avoid blocking
        image_data = await asyncio.to_thread(self._encode_image, image_path)
        if not image_data:
            return None
        
        # Use prompt style if no custom prompt provided
        if prompt is None:
            if self.prompt_style == "fast":
                prompt_text = self.FAST_PROMPT
            else:
                prompt_text = self.DEFAULT_PROMPT
        else:
            prompt_text = prompt
        
        try:
            # Create messages with image
            messages = [
                SystemMessage(content=prompt_text),
                HumanMessage(
                    content=[
                        {
                            "type": "text",
                            "text": "",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        }
                    ]
                )
            ]
            
            # Invoke LangChain LLM asynchronously with retry
            for attempt in range(self.max_retries + 1):
                try:
                    response = await self._llm.ainvoke(messages)
                    result = response.content.strip() if hasattr(response, 'content') else str(response).strip()
                    self._last_success_time = time.time()
                    self.logger.debug(f"Generated description for {image_path.name} ({len(result)} chars)")
                    return result
                except Exception as e:
                    if attempt < self.max_retries:
                        delay = self.retry_initial_delay * (self.retry_backoff_factor ** attempt)
                        self.logger.warning(f"Vision API call failed (attempt {attempt + 1}/{self.max_retries + 1}), retrying in {delay:.2f}s: {e}")
                        await asyncio.sleep(delay)
                    else:
                        raise
        except Exception as e:
            # Check if it's a network/server error or model error
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ['connection', 'timeout', 'server', 'http', 'network']):
                raise VisionAPIError(f"Vision API error for {image_path}: {e}") from e
            # Check for model-related errors - try direct API call as fallback
            if 'model is required' in error_str or ('model' in error_str and '400' in error_str):
                # Fallback: Try direct Ollama API call
                try:
                    return await self._describe_direct_api_async(image_path, prompt_text, image_data)
                except Exception as fallback_error:
                    raise VisionAPIError(
                        f"Model '{self.model}' error. LangChain failed: {e}. "
                        f"Direct API fallback also failed: {fallback_error}. "
                        f"Please verify the model exists and supports vision (run 'ollama show {self.model}')."
                    ) from fallback_error
            self.logger.error(f"Vision API error for {image_path}: {e}")
            return None
    
    def _describe_direct_api(
        self,
        image_path: Path,
        prompt_text: str,
        image_data: str,
    ) -> Optional[str]:
        """Fallback: Direct Ollama API call when LangChain fails.
        
        This is used when LangChain's ChatOllama doesn't properly pass the model parameter.
        Uses raw image bytes (like Ollama web interface) instead of processed JPEG.
        """
        # Try raw image bytes first (like Ollama web interface does)
        try:
            raw_image_data = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        except Exception:
            # Fallback to processed image data if raw fails
            raw_image_data = image_data
        
        # Use a simple user prompt - incorporate system prompt into user message
        user_prompt = f"{prompt_text}\n\nPlease analyze the image and provide a detailed description."
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": user_prompt,
                    "images": [raw_image_data],
                },
            ],
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_ctx": 8096,
            },
        }
        
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.ollama_url}/api/chat",
                    json=payload,
                )
                response.raise_for_status()
                result = response.json()
                return result.get("message", {}).get("content", "").strip()
        except httpx.HTTPStatusError as e:
            # Try with processed image if raw fails
            if raw_image_data != image_data:
                payload["messages"][0]["images"] = [image_data]
                try:
                    with httpx.Client(timeout=self.timeout) as client:
                        response = client.post(
                            f"{self.ollama_url}/api/chat",
                            json=payload,
                        )
                        response.raise_for_status()
                        result = response.json()
                        return result.get("message", {}).get("content", "").strip()
                except Exception:
                    pass
            raise VisionAPIError(f"Direct API call failed for {image_path}: {e.response.text if hasattr(e, 'response') else str(e)}") from e
        except Exception as e:
            raise VisionAPIError(f"Direct API call failed for {image_path}: {e}") from e
    
    async def _describe_direct_api_async(
        self,
        image_path: Path,
        prompt_text: str,
        image_data: str,
    ) -> Optional[str]:
        """Async fallback: Direct Ollama API call when LangChain fails.
        Uses raw image bytes (like Ollama web interface) instead of processed JPEG.
        """
        # Try raw image bytes first (like Ollama web interface does)
        try:
            raw_image_data = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        except Exception:
            # Fallback to processed image data if raw fails
            raw_image_data = image_data
        
        # Use a simple user prompt - incorporate system prompt into user message
        user_prompt = f"{prompt_text}\n\nPlease analyze the image and provide a detailed description."
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": user_prompt,
                    "images": [raw_image_data],
                },
            ],
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_ctx": 8096,
            },
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/chat",
                    json=payload,
                )
                response.raise_for_status()
                result = response.json()
                return result.get("message", {}).get("content", "").strip()
        except httpx.HTTPStatusError as e:
            # Try with processed image if raw fails
            if raw_image_data != image_data:
                payload["messages"][0]["images"] = [image_data]
                try:
                    async with httpx.AsyncClient(timeout=self.timeout) as client:
                        response = await client.post(
                            f"{self.ollama_url}/api/chat",
                            json=payload,
                        )
                        response.raise_for_status()
                        result = response.json()
                        return result.get("message", {}).get("content", "").strip()
                except Exception:
                    pass
            raise VisionAPIError(f"Direct API call failed for {image_path}: {e.response.text if hasattr(e, 'response') else str(e)}") from e
        except Exception as e:
            raise VisionAPIError(f"Direct API call failed for {image_path}: {e}") from e
    
    def is_available(self) -> bool:
        """Check if the vision service is available."""
        if self.circuit_breaker.is_open:
            return False
        try:
            # Try to invoke with a simple test message
            test_messages = [SystemMessage(content="test"), HumanMessage(content="test")]
            self._llm.invoke(test_messages)
            return True
        except Exception:
            return False
    
    def get_health_status(self) -> dict:
        """Get detailed health status of the vision service.
        
        Returns:
            Dictionary with health status information
        """
        is_available = self.is_available()
        return {
            "available": is_available,
            "model": self.model,
            "ollama_url": self.ollama_url,
            "circuit_state": self.circuit_breaker._state.value,
            "last_success_time": self._last_success_time,
            "failure_count": self.circuit_breaker._failure_count,
        }
    
    def close(self) -> None:
        """Close the LLM client (no-op for LangChain, but kept for compatibility)."""
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
