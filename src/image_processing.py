"""
image_processing.py -- Process Extracted Images
-----------------------------------------------
Performs OCR and vision API analysis on extracted images.

Features:
  - Tesseract OCR for text extraction from images
  - Azure Computer Vision API for image analysis
  - Image description generation
  - Text and object detection
  - Combine OCR + vision API results for comprehensive analysis
"""

import os
import logging
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from PIL import Image

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

try:
    from azure.ai.vision.imageanalysis import ImageAnalysisClient
    from azure.ai.vision.imageanalysis.models import VisualFeatures
    from azure.core.credentials import AzureKeyCredential
    AZURE_VISION_AVAILABLE = True
except ImportError:
    AZURE_VISION_AVAILABLE = False

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(os.path.join("..", ".env"))


class ImageProcessor:
    """Process images using OCR and vision APIs."""

    def __init__(self):
        """Initialize image processor with Azure credentials if available."""
        self.azure_endpoint = os.getenv("AZURE_VISION_ENDPOINT")
        self.azure_key = os.getenv("AZURE_VISION_API_KEY")
        self.azure_client = None
        
        if AZURE_VISION_AVAILABLE and self.azure_endpoint and self.azure_key:
            try:
                self.azure_client = ImageAnalysisClient(
                    endpoint=self.azure_endpoint,
                    credential=AzureKeyCredential(self.azure_key),
                )
                logger.info("Azure Computer Vision API initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Azure Vision API: {e}")
        elif not self.azure_endpoint or not self.azure_key:
            logger.info("Azure Vision credentials not configured. Falling back to OCR only.")

    def extract_text_with_ocr(self, image_path: str) -> Dict[str, Any]:
        """
        Extract text from image using Tesseract OCR.

        Args:
            image_path: Path to the image file

        Returns:
            Dict with:
            - text: Extracted text
            - confidence: Estimated confidence (0-100)
            - method: "tesseract_ocr"
            - status: "success" or "error"
        """
        result = {
            "text": "",
            "confidence": 0,
            "method": "tesseract_ocr",
            "status": "error",
            "error": None,
        }

        if not PYTESSERACT_AVAILABLE:
            result["error"] = "pytesseract not available"
            logger.warning("pytesseract not installed. Install with: pip install pytesseract")
            return result

        if not os.path.exists(image_path):
            result["error"] = f"Image not found: {image_path}"
            logger.warning(result["error"])
            return result

        try:
            # Open image with PIL
            image = Image.open(image_path)
            
            # Extract text using Tesseract
            text = pytesseract.image_to_string(image)
            
            # Get detailed info (optional - for confidence scoring)
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data.get("confidence", [0]) if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            result["text"] = text.strip()
            result["confidence"] = int(avg_confidence)
            result["status"] = "success"
            
            logger.info(f"OCR extraction successful: {len(text)} chars extracted")
            
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"OCR extraction failed: {e}")

        return result

    def analyze_image_with_azure(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze image using Azure Computer Vision API.

        Args:
            image_path: Path to the image file

        Returns:
            Dict with analysis results:
            - description: Image description
            - objects: Detected objects
            - text: Detected text (OCR via Azure)
            - tags: Detected tags
            - dominant_colors: Dominant colors
            - method: "azure_vision"
            - status: "success" or "error"
        """
        result = {
            "description": "",
            "objects": [],
            "text": "",
            "tags": [],
            "dominant_colors": [],
            "method": "azure_vision",
            "status": "error",
            "error": None,
        }

        if not AZURE_VISION_AVAILABLE:
            result["error"] = "Azure Vision SDK not installed"
            logger.warning("azure-ai-vision not available. Install with: pip install azure-ai-vision")
            return result

        if not self.azure_client:
            result["error"] = "Azure Vision credentials not configured"
            return result

        if not os.path.exists(image_path):
            result["error"] = f"Image not found: {image_path}"
            logger.warning(result["error"])
            return result

        try:
            # Read image file
            with open(image_path, "rb") as f:
                image_data = f.read()
            
            # Analyze image with multiple features
            analysis_result = self.azure_client.analyze_image_from_url(
                image_url=None,  # Using local file instead
                visual_features=[
                    VisualFeatures.DESCRIPTION,
                    VisualFeatures.OBJECTS,
                    VisualFeatures.READ,
                    VisualFeatures.TAGS,
                    VisualFeatures.ADULT,
                ],
            )
            
            # Extract description
            if analysis_result.description and analysis_result.description.captions:
                result["description"] = analysis_result.description.captions[0].text
            
            # Extract objects
            if analysis_result.objects:
                result["objects"] = [
                    {
                        "name": obj.tags[0].name if obj.tags else "unknown",
                        "confidence": obj.tags[0].confidence if obj.tags else 0,
                        "rect": {
                            "x": obj.rectangle.x,
                            "y": obj.rectangle.y,
                            "w": obj.rectangle.w,
                            "h": obj.rectangle.h,
                        },
                    }
                    for obj in analysis_result.objects
                ]
            
            # Extract text (OCR)
            if hasattr(analysis_result, 'read') and analysis_result.read:
                text_blocks = []
                for block in analysis_result.read.blocks:
                    for line in block.lines:
                        text_blocks.append(line.text)
                result["text"] = " ".join(text_blocks)
            
            # Extract tags
            if analysis_result.tags:
                result["tags"] = [
                    {"name": tag.name, "confidence": tag.confidence}
                    for tag in analysis_result.tags
                ]
            
            result["status"] = "success"
            logger.info(f"Azure Vision analysis successful")
            
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Azure Vision analysis failed: {e}")

        return result

    def process_image_complete(self, image_path: str) -> Dict[str, Any]:
        """
        Process image with all available methods (OCR + Azure Vision).

        Args:
            image_path: Path to the image file

        Returns:
            Combined analysis result
        """
        result = {
            "image_path": image_path,
            "image_filename": Path(image_path).name,
            "ocr_result": None,
            "azure_result": None,
            "combined_text": "",
        }

        # Try OCR first
        if PYTESSERACT_AVAILABLE:
            result["ocr_result"] = self.extract_text_with_ocr(image_path)
        
        # Try Azure Vision
        if self.azure_client:
            result["azure_result"] = self.analyze_image_with_azure(image_path)
        
        # Combine text results
        texts = []
        if result["ocr_result"] and result["ocr_result"].get("status") == "success":
            texts.append(result["ocr_result"].get("text", ""))
        if result["azure_result"] and result["azure_result"].get("status") == "success":
            if result["azure_result"].get("description"):
                texts.append(f"Description: {result['azure_result']['description']}")
            if result["azure_result"].get("text"):
                texts.append(result["azure_result"]["text"])
        
        result["combined_text"] = " ".join(filter(None, texts))
        
        logger.info(f"Completed image processing: {image_path}")
        return result

    def batch_process_images(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple images.

        Args:
            image_paths: List of image file paths

        Returns:
            List of processing results
        """
        results = []
        
        for idx, image_path in enumerate(image_paths, 1):
            logger.info(f"Processing image {idx}/{len(image_paths)}: {image_path}")
            result = self.process_image_complete(image_path)
            results.append(result)
        
        logger.info(f"Batch processing complete: {len(results)} images processed")
        return results


class ImageDescriptionCache:
    """Cache image descriptions to avoid reprocessing."""

    def __init__(self, cache_dir: str = "../data/image_cache"):
        """
        Initialize cache.

        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def get_cache_path(self, image_path: str) -> str:
        """Get cache file path for an image."""
        image_hash = Path(image_path).stem
        return os.path.join(self.cache_dir, f"{image_hash}_cache.json")

    def get(self, image_path: str) -> Optional[Dict[str, Any]]:
        """
        Get cached analysis for an image.

        Args:
            image_path: Path to the image

        Returns:
            Cached analysis or None if not found
        """
        cache_path = self.get_cache_path(image_path)
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        
        return None

    def set(self, image_path: str, analysis: Dict[str, Any]) -> None:
        """
        Cache analysis for an image.

        Args:
            image_path: Path to the image
            analysis: Analysis result to cache
        """
        cache_path = self.get_cache_path(image_path)
        
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(analysis, f, indent=2, ensure_ascii=True)
            logger.info(f"Cached analysis for {Path(image_path).name}")
        except Exception as e:
            logger.error(f"Failed to cache analysis: {e}")


def get_image_processor() -> ImageProcessor:
    """
    Get or create a global image processor instance.

    Returns:
        ImageProcessor instance
    """
    return ImageProcessor()


def get_image_cache(cache_dir: str = "../data/image_cache") -> ImageDescriptionCache:
    """
    Get or create a global image cache instance.

    Args:
        cache_dir: Directory to store cache files

    Returns:
        ImageDescriptionCache instance
    """
    return ImageDescriptionCache(cache_dir)
