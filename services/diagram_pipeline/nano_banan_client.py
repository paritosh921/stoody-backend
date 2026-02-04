"""
Nano Banan Pro Image Generation Client

Generates exam-quality diagram images from text instructions.

Constraints:
- Uses ONLY Nano Banan Pro for image generation
- No other image generation libraries allowed
"""

"""
Gemini Image Generation Client (Nano Banana Pro)

Uses Google Gemini's native image generation capabilities for exam-quality diagrams.

Models:
- gemini-2.5-flash-image: Fast, efficient (Nano Banana)
- gemini-3-pro-image-preview: Professional quality with advanced text rendering (Nano Banana Pro)

Constraints:
- Uses ONLY Gemini API for image generation
- No other image generation libraries allowed
"""

import os
import uuid
import base64
import logging
import json
import httpx
from typing import Optional
from datetime import datetime
from pathlib import Path

from .models import DiagramImage

logger = logging.getLogger(__name__)

# Gemini API Configuration
# Try NANO_BANAN_API_KEY first, then fall back to GEMINI_API_KEY
GEMINI_API_KEY = os.getenv("NANO_BANAN_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta"

# Model options for image generation (Nano Banana = Gemini native image gen):
# - gemini-2.5-flash-image: Nano Banana - fast, efficient (uses :generateContent)
# - gemini-3-pro-image-preview: Nano Banana Pro - professional quality (uses :generateContent)
# Note: Imagen models require Vertex AI, use Gemini models for standard API
GEMINI_IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")
GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash")

# Image storage configuration
IMAGES_DIR = Path(os.getenv("IMAGES_DIR", "images"))
DIAGRAMS_SUBDIR = "diagrams"


class GeminiImageClient:
    """
    Client for Google Gemini's native image generation (Nano Banana Pro).
    
    Generates exam-quality diagrams from text instructions using Gemini API.
    """
    
    def __init__(self):
        self.api_key = GEMINI_API_KEY
        self.api_url = GEMINI_API_URL
        self.model = GEMINI_IMAGE_MODEL
        self.vision_model = GEMINI_VISION_MODEL
        self.images_dir = IMAGES_DIR / DIAGRAMS_SUBDIR
        
        # Ensure diagrams directory exists
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.api_key:
            logger.warning("GEMINI_API_KEY not set - diagram generation will not work")
    
    async def generate_diagram(
        self,
        instructions: str,
        question_id: str,
        instructions_version: int = 1,
        aspect_ratio: str = "1:1",
        resolution: str = "1K",
    ) -> DiagramImage:
        """
        Generate a diagram image from text instructions using Gemini API.
        
        Args:
            instructions: The text instructions for the diagram
            question_id: ID of the question (for file naming)
            instructions_version: Version of the instructions
            aspect_ratio: Image aspect ratio ("1:1", "4:3", "16:9", etc.)
            resolution: Image resolution ("1K", "2K", "4K")
        
        Returns:
            DiagramImage object with path and metadata
        """
        import sys
        print(f"[NANO-BANAN] generate_diagram called for {question_id}", flush=True)
        print(f"[NANO-BANAN] API key: {self.api_key[:15]}..." if self.api_key else "[NANO-BANAN] No API key!", flush=True)
        print(f"[NANO-BANAN] Model: {self.model}", flush=True)
        sys.stdout.flush()
        
        if not self.api_key:
            print("[NANO-BANAN] ERROR: No API key configured!")
            raise ValueError("GEMINI_API_KEY not configured")
        
        # Generate unique filename
        image_id = str(uuid.uuid4())[:8]
        filename = f"diagram_{question_id}_{instructions_version}_{image_id}.png"
        filepath = self.images_dir / filename
        
        # Enhance the prompt for exam-quality diagrams
        enhanced_prompt = self._enhance_prompt(instructions)
        
        # Determine if using Imagen or Gemini model
        # Nano Banana models (gemini-2.5-flash-image, gemini-3-pro-image-preview) use generateContent
        is_imagen = self.model.startswith("imagen")
        is_nano_banana = "flash-image" in self.model or "pro-image" in self.model
        
        if is_imagen:
            # Imagen API uses :predict endpoint (requires Vertex AI, likely won't work here)
            url = f"{self.api_url}/models/{self.model}:predict"
            payload = {
                "instances": [
                    {"prompt": enhanced_prompt}
                ],
                "parameters": {
                    "sampleCount": 1,
                    "aspectRatio": aspect_ratio,
                }
            }
        else:
            # Nano Banana / Gemini native image generation uses :generateContent
            # Based on Google's official docs: https://ai.google.dev/gemini-api/docs/image-generation
            url = f"{self.api_url}/models/{self.model}:generateContent"
            payload = {
                "contents": [{
                    "parts": [
                        {"text": enhanced_prompt}
                    ]
                }]
            }
            # Note: For basic text-to-image, no generationConfig needed
            # The model will return image in response.candidates[0].content.parts[].inlineData
        
        headers = {
            "Content-Type": "application/json",
        }
        
        # Add API key as query parameter (Gemini style)
        params = {
            "key": self.api_key,
        }
        
        api_type = "Imagen" if is_imagen else ("Nano Banana" if is_nano_banana else "Gemini")
        print(f"[NANO-BANAN] About to make API request...", flush=True)
        print(f"[NANO-BANAN] URL: {url}", flush=True)
        print(f"[NANO-BANAN] Payload keys: {list(payload.keys())}", flush=True)
        logger.info(f"[{api_type.upper()}] Making request: model={self.model}")
        logger.info(f"[{api_type.upper()}] URL: {url}")
        logger.info(f"[{api_type.upper()}] API key present: {bool(self.api_key)}, key prefix: {self.api_key[:15]}..." if self.api_key else f"[{api_type.upper()}] No API key!")
        logger.debug(f"[IMAGEN-API] Payload: {json.dumps(payload)[:500]}")
        
        # Retry logic for rate limits (429 errors)
        max_retries = 3
        base_delay = 5.0  # Start with 5 seconds
        
        async with httpx.AsyncClient(timeout=180.0) as client:
            last_error = None
            for attempt in range(max_retries):
                response = await client.post(
                    url,
                    headers=headers,
                    params=params,
                    json=payload,
                )
                
                if response.status_code == 200:
                    print(f"[NANO-BANAN] SUCCESS! Got 200 response on attempt {attempt + 1}", flush=True)
                    logger.info(f"[{api_type.upper()}] Success! Status 200 received on attempt {attempt + 1}")
                    break
                elif response.status_code == 429:
                    # Rate limited - extract retry delay from response if available
                    error_data = response.json() if response.text else {}
                    retry_delay = base_delay * (2 ** attempt)  # Exponential backoff
                    
                    # Try to get suggested retry delay from API response
                    try:
                        details = error_data.get("error", {}).get("details", [])
                        for detail in details:
                            if detail.get("@type") == "type.googleapis.com/google.rpc.RetryInfo":
                                delay_str = detail.get("retryDelay", "5s")
                                retry_delay = float(delay_str.rstrip("s")) + 1.0  # Add 1s buffer
                                break
                    except:
                        pass
                    
                    logger.warning(f"Gemini API rate limited (attempt {attempt + 1}/{max_retries}), retrying in {retry_delay}s...")
                    last_error = f"Rate limited: {response.text[:200]}"
                    
                    if attempt < max_retries - 1:
                        import asyncio
                        await asyncio.sleep(retry_delay)
                        continue
                else:
                    error_text = response.text
                    logger.error(f"[{api_type.upper()}] Error {response.status_code}: {error_text}")
                    logger.error(f"[{api_type.upper()}] Full URL was: {url}")
                    logger.error(f"[{api_type.upper()}] Model was: {self.model}")
                    raise Exception(f"{api_type} API error: {response.status_code} - {error_text[:500]}")
            
            # Check if we exhausted retries on rate limit
            if response.status_code == 429:
                logger.error(f"Gemini API rate limit exceeded after {max_retries} retries")
                raise Exception(f"Gemini API rate limit exceeded. Please try again later or enable billing. {last_error}")
            
            data = response.json()
            
            # Extract the image based on model type
            image_b64 = None
            text_response = None
            
            if is_imagen:
                # Imagen response format: {"predictions": [{"bytesBase64Encoded": "..."}]}
                if "predictions" in data and len(data["predictions"]) > 0:
                    prediction = data["predictions"][0]
                    image_b64 = prediction.get("bytesBase64Encoded")
                    logger.info(f"[IMAGEN] Prediction received, image size: {len(image_b64) if image_b64 else 0}")
            else:
                # Nano Banana / Gemini response format: {"candidates": [{"content": {"parts": [...]}}]}
                logger.info(f"[{api_type.upper()}] Parsing response, keys: {list(data.keys())}")
                if "candidates" in data and len(data["candidates"]) > 0:
                    candidate = data["candidates"][0]
                    logger.info(f"[{api_type.upper()}] Candidate keys: {list(candidate.keys())}")
                    if "content" in candidate and "parts" in candidate["content"]:
                        parts = candidate["content"]["parts"]
                        logger.info(f"[{api_type.upper()}] Found {len(parts)} parts in response")
                        for i, part in enumerate(parts):
                            logger.info(f"[{api_type.upper()}] Part {i} keys: {list(part.keys())}")
                            if "text" in part:
                                text_response = part["text"]
                                logger.info(f"[{api_type.upper()}] Text response: {text_response[:200] if text_response else 'None'}")
                            elif "inlineData" in part:
                                image_b64 = part["inlineData"].get("data")
                                logger.info(f"[{api_type.upper()}] Image data received, size: {len(image_b64) if image_b64 else 0}")
                                break
                else:
                    logger.warning(f"[{api_type.upper()}] No candidates in response: {json.dumps(data)[:500]}")
            
            if not image_b64:
                error_msg = f"No image in response. Data: {json.dumps(data)[:500]}"
                logger.error(error_msg)
                raise Exception(error_msg)
            
            # Decode and save the image
            image_bytes = base64.b64decode(image_b64)
            
            with open(filepath, "wb") as f:
                f.write(image_bytes)
            
            # Get image dimensions
            width, height = self._get_image_dimensions(image_bytes)
            
            # Build the relative path for storage
            relative_path = f"{DIAGRAMS_SUBDIR}/{filename}"
            
            logger.info(f"Generated diagram: {filename} ({width}x{height})")
            
            return DiagramImage(
                id=image_id,
                path=relative_path,
                filename=filename,
                format="png",
                width=width,
                height=height,
                size_bytes=len(image_bytes),
                instructions_version=instructions_version,
                base64_data=image_b64,
            )
    
    def _enhance_prompt(self, instructions: str) -> str:
        """
        Enhance the instructions with exam-quality modifiers for Gemini.
        
        Ensures the generated image is suitable for exam papers.
        """
        # Concise, powerful prefix for Gemini/Nano Banana
        prefix = (
            "Generate a professional, high-contrast, black-and-white line drawing for an exam paper. "
            "Style: Technical illustration, vector art style, white background, black lines. "
            "NO SHADING, NO COLOR, NO GRADIENTS. "
            "Do NOT include any question text or problem descriptions. "
            "Make all labels and numeric values LARGE, clear, and legible. "
            "Place labels close to points with small leader lines if needed. "
            "Avoid overlapping labels. "
            "Draw EXACTLY what is described here: "
        )
        
        return f"{prefix}{instructions}"
    
    def _get_image_dimensions(self, image_bytes: bytes) -> tuple:
        """
        Get image dimensions from bytes.
        
        Returns (width, height) tuple, or (None, None) if unable to determine.
        """
        try:
            from PIL import Image
            import io
            
            img = Image.open(io.BytesIO(image_bytes))
            return img.size
        except Exception as e:
            logger.warning(f"Could not get image dimensions: {e}")
            return (None, None)
    
    async def get_image_base64(self, image_path: str) -> Optional[str]:
        """
        Get base64 encoded image data from a stored image.
        
        Args:
            image_path: Relative path to the image
        
        Returns:
            Base64 encoded image data, or None if not found
        """
        full_path = IMAGES_DIR / image_path
        
        if not full_path.exists():
            # Try parent directory
            full_path = IMAGES_DIR.parent / image_path
        
        if not full_path.exists():
            return None
        
        with open(full_path, "rb") as f:
            image_bytes = f.read()
        
        return base64.b64encode(image_bytes).decode("utf-8")
    
    def delete_image(self, image_path: str) -> bool:
        """
        Delete a stored diagram image.
        
        Args:
            image_path: Relative path to the image
        
        Returns:
            True if deleted, False if not found
        """
        full_path = IMAGES_DIR / image_path
        
        if not full_path.exists():
            full_path = IMAGES_DIR.parent / image_path
        
        if full_path.exists():
            full_path.unlink()
            return True
        
        return False

    async def describe_diagram(
        self,
        image_b64: str,
        question_text: str,
        instructions: str,
    ) -> str:
        """
        Describe a generated diagram using Gemini vision.

        Returns a concise textual description of the diagram.
        """
        if not self.api_key or not image_b64:
            return ""

        prompt = (
            "Describe the diagram precisely for exam validation. "
            "List shapes, labels, relative positions, and any numeric values shown. "
            "Do not invent elements that are not visible. If a value/label is not visible, say 'not visible'."
        )

        url = f"{self.api_url}/models/{self.vision_model}:generateContent"
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"inlineData": {"mimeType": "image/png", "data": image_b64}},
                ]
            }]
        }

        headers = {"Content-Type": "application/json"}
        params = {"key": self.api_key}

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, headers=headers, params=params, json=payload)
            if response.status_code != 200:
                raise Exception(f"Gemini vision error: {response.status_code} - {response.text[:200]}")

            data = response.json()
            if "candidates" in data and data["candidates"]:
                parts = data["candidates"][0].get("content", {}).get("parts", [])
                for part in parts:
                    if "text" in part:
                        return part["text"].strip()

        return ""


# Alias for backward compatibility
NanoBananClient = GeminiImageClient
