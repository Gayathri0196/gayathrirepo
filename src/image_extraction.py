"""
image_extraction.py -- Extract Images from PDFs
-----------------------------------------------
Extracts images from PDF files using PyMuPDF and pdf2image.
Stores images with metadata (page number, coordinates, source).

Features:
  - Extract images from both regular and scanned PDFs
  - Generate descriptive filenames based on content
  - Track image metadata (source page, position, extraction method)
  - Save images locally for processing and retrieval
"""

import os
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
import fitz  # PyMuPDF

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

logger = logging.getLogger(__name__)


class ImageExtractor:
    """Extract and manage images from PDF documents."""

    def __init__(self, output_dir: str = "../data/extracted_images"):
        """
        Initialize the image extractor.

        Args:
            output_dir: Directory to save extracted images
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Image output directory: {self.output_dir}")

    def extract_images_from_pdf_fitz(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract images from PDF using PyMuPDF (fitz).

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of image metadata dicts with:
            - image_path: Path where image was saved
            - page_number: Page where image was found
            - image_index: Index of image on the page
            - coordinates: Image bounding box on page
            - size: Image dimensions (width, height)
            - extraction_method: "fitz"
            - source_pdf: Original PDF path
        """
        images = []
        
        if not os.path.exists(pdf_path):
            logger.warning(f"PDF not found: {pdf_path}")
            return images

        try:
            doc = fitz.open(pdf_path)
            pdf_name = Path(pdf_path).stem
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images(full=True)
                
                if not image_list:
                    continue
                
                logger.info(f"Found {len(image_list)} images on page {page_num + 1}")
                
                for img_index, (xref, *_) in enumerate(image_list):
                    try:
                        # Extract image from PDF
                        pix = fitz.Pixmap(doc, xref)
                        
                        # Convert to RGB if necessary
                        if pix.alpha:
                            pix = fitz.Pixmap(fitz.csRGB, pix)
                        
                        # Create filename
                        image_hash = hashlib.md5(
                            f"{pdf_name}_{page_num}_{img_index}".encode()
                        ).hexdigest()[:8]
                        filename = f"{pdf_name}_p{page_num + 1}_img{img_index}_{image_hash}.png"
                        filepath = os.path.join(self.output_dir, filename)
                        
                        # Save image
                        pix.save(filepath)
                        pix.n_colorspace = 1
                        
                        # Get image dimensions and coordinates
                        rect = page.get_image_bbox(image_list[img_index])
                        
                        image_metadata = {
                            "image_path": filepath,
                            "image_filename": filename,
                            "page_number": page_num + 1,
                            "image_index": img_index,
                            "coordinates": {
                                "x0": rect.x0,
                                "y0": rect.y0,
                                "x1": rect.x1,
                                "y1": rect.y1,
                                "width": rect.width,
                                "height": rect.height,
                            },
                            "size": (pix.width, pix.height),
                            "extraction_method": "fitz",
                            "source_pdf": pdf_path,
                        }
                        
                        images.append(image_metadata)
                        logger.info(f"Extracted image: {filename}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to extract image on page {page_num + 1}: {e}")
                        continue
            
            doc.close()
            
        except Exception as e:
            logger.error(f"Error extracting images from {pdf_path}: {e}")
        
        logger.info(f"Extracted {len(images)} images from {pdf_path}")
        return images

    def extract_images_from_pdf_pdf2image(
        self, pdf_path: str, dpi: int = 200
    ) -> List[Dict[str, Any]]:
        """
        Extract images from PDF using pdf2image (handles scanned PDFs better).

        Args:
            pdf_path: Path to the PDF file
            dpi: Resolution for PDF conversion

        Returns:
            List of image metadata dicts
        """
        if not PDF2IMAGE_AVAILABLE:
            logger.warning("pdf2image not available. Install with: pip install pdf2image")
            return []

        images = []
        
        if not os.path.exists(pdf_path):
            logger.warning(f"PDF not found: {pdf_path}")
            return images

        try:
            pdf_name = Path(pdf_path).stem
            pil_images = convert_from_path(pdf_path, dpi=dpi)
            
            for page_num, pil_image in enumerate(pil_images):
                try:
                    # Create filename
                    image_hash = hashlib.md5(
                        f"{pdf_name}_page_{page_num}".encode()
                    ).hexdigest()[:8]
                    filename = f"{pdf_name}_p{page_num + 1}_{image_hash}.png"
                    filepath = os.path.join(self.output_dir, filename)
                    
                    # Save image
                    pil_image.save(filepath, "PNG")
                    
                    image_metadata = {
                        "image_path": filepath,
                        "image_filename": filename,
                        "page_number": page_num + 1,
                        "image_index": 0,
                        "coordinates": {
                            "x0": 0,
                            "y0": 0,
                            "x1": pil_image.width,
                            "y1": pil_image.height,
                            "width": pil_image.width,
                            "height": pil_image.height,
                        },
                        "size": pil_image.size,
                        "extraction_method": "pdf2image",
                        "source_pdf": pdf_path,
                        "dpi": dpi,
                    }
                    
                    images.append(image_metadata)
                    logger.info(f"Extracted page as image: {filename}")
                    
                except Exception as e:
                    logger.warning(f"Failed to save page {page_num + 1} image: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Error extracting images with pdf2image from {pdf_path}: {e}")
        
        logger.info(f"Extracted {len(images)} page images from {pdf_path}")
        return images

    def extract_all_images(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract images using both methods and combine results.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Combined list of image metadata from both extraction methods
        """
        images = []
        
        # Try fitz first (faster for embedded images)
        fitz_images = self.extract_images_from_pdf_fitz(pdf_path)
        images.extend(fitz_images)
        
        # For scanned PDFs or if no images found, try pdf2image
        if not fitz_images and PDF2IMAGE_AVAILABLE:
            logger.info("No images found with fitz, trying pdf2image...")
            pdf2img_images = self.extract_images_from_pdf_pdf2image(pdf_path)
            images.extend(pdf2img_images)
        
        logger.info(f"Total images extracted: {len(images)}")
        return images

    def list_extracted_images(self) -> List[str]:
        """
        List all extracted images in the output directory.

        Returns:
            List of image file paths
        """
        if not os.path.exists(self.output_dir):
            return []
        
        images = []
        for filename in os.listdir(self.output_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                images.append(os.path.join(self.output_dir, filename))
        
        return sorted(images)


def get_image_extractor(output_dir: str = "../data/extracted_images") -> ImageExtractor:
    """
    Get or create a global image extractor instance.

    Args:
        output_dir: Directory to save extracted images

    Returns:
        ImageExtractor instance
    """
    return ImageExtractor(output_dir)
