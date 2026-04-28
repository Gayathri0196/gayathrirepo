# Image Extraction & Processing

This document describes the image extraction and processing capabilities added to the RAG Retrieval Agent.

## Overview

The system now automatically extracts images from PDFs, processes them using OCR and vision APIs, and integrates image context into the question-answering pipeline.

### Features

- **Image Extraction**: Extracts images from both regular and scanned PDFs
- **Optical Character Recognition (OCR)**: Extracts text from images using Tesseract
- **Vision API Analysis**: Analyzes images using Azure Computer Vision API
- **Image Caching**: Caches processed images to avoid reprocessing
- **Automatic Context Integration**: Enriches answers with image context
- **Both Source & Question PDFs**: Processes images from ingestion PDFs and question PDFs

## Architecture

### Components

#### 1. `src/image_extraction.py`
Handles image extraction from PDFs.

**Key Classes:**
- `ImageExtractor`: Main class for image extraction
  - `extract_images_from_pdf_fitz()`: Uses PyMuPDF for embedded image extraction
  - `extract_images_from_pdf_pdf2image()`: Uses pdf2image for scanned PDFs
  - `extract_all_images()`: Combines both methods

**Usage:**
```python
from image_extraction import get_image_extractor

extractor = get_image_extractor()
images = extractor.extract_all_images("path/to/pdf.pdf")
# Returns list of dicts with:
# - image_path, image_filename
# - page_number, image_index
# - coordinates, size
# - extraction_method, source_pdf
```

#### 2. `src/image_processing.py`
Processes extracted images with OCR and vision APIs.

**Key Classes:**
- `ImageProcessor`: Performs image analysis
  - `extract_text_with_ocr()`: Tesseract OCR extraction
  - `analyze_image_with_azure()`: Azure Computer Vision analysis
  - `process_image_complete()`: Combines OCR + vision API
  - `batch_process_images()`: Process multiple images

- `ImageDescriptionCache`: Caches analysis results

**Usage:**
```python
from image_processing import get_image_processor, get_image_cache

processor = get_image_processor()
cache = get_image_cache()

# Process an image
analysis = processor.process_image_complete("path/to/image.png")

# Returns dict with:
# - ocr_result: OCR extraction results
# - azure_result: Azure Vision analysis
# - combined_text: Merged analysis text
```

#### 3. Modified `src/ingestion.py`
Now extracts and processes images from source PDFs during ingestion.

**Changes:**
- Creates "Images & Diagrams" domain chunks from image analysis
- Includes image metadata and OCR confidence in chunk metadata
- Stores image paths for later reference

**Processing Flow:**
1. Extract text chunks (existing)
2. Extract images from PDF
3. Process each image (OCR + vision API)
4. Cache analysis results
5. Create chunks from image descriptions
6. Save all chunks to vector database

#### 4. Modified `src/batch_questions.py`
Now extracts and processes images from question PDFs.

**Changes:**
- `extract_questions_from_pdf()` now includes image extraction
- Updated `ExtractionBundle` to store extracted images
- Added `extracted_images` and `image_analyses` fields

#### 5. Modified `src/retriever.py`
Enhances retrieval with image context.

**New Function:**
- `enhance_with_image_context()`: Finds and adds related image context to retrieved documents

**Features:**
- Searches for images on the same pages as text results
- Adds image descriptions to document metadata
- Boosts document scores when image context is found

#### 6. Modified `src/vector_store.py`
Now handles image chunks in vector database.

**Changes:**
- Stores image metadata separately from text metadata
- Tracks image path, OCR confidence, and analysis type
- Supports filtering and searching by image domain

#### 7. Modified `src/qa_chain.py`
Integrates image context into LLM prompts.

**Changes:**
- `answer_with_context()` now includes image context
- Appends "IMAGE & DIAGRAM CONTEXT" section to prompts
- Provides image descriptions to LLM for better answers

## Configuration

### Environment Variables

Add these to your `.env` file:

```env
# Azure Computer Vision (optional - for image analysis)
AZURE_VISION_ENDPOINT=https://your-vision-resource.cognitiveservices.azure.com/
AZURE_VISION_API_KEY=your-vision-api-key-here

# Image processing options
ENABLE_OCR_FALLBACK=0  # Set to 1 to enable OCR for text extraction from scanned PDFs
```

### Dependencies

New Python packages required:

```
PyMuPDF            # PDF image extraction (embedded)
pdf2image          # PDF to image conversion (scanned PDFs)
Pillow             # Image processing
pytesseract        # Tesseract OCR
opencv-python      # Image processing
azure-ai-vision    # Azure Computer Vision API
```

Install with:
```bash
pip install -r requirements.txt
```

### Tesseract Setup (for OCR)

#### Windows
1. Download installer: https://github.com/UB-Mannheim/tesseract/wiki
2. Install to default location or set environment variable:
```powershell
$env:PYTESSERACT_PATH = "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

#### macOS
```bash
brew install tesseract
```

#### Linux
```bash
sudo apt-get install tesseract-ocr
```

## Usage

### Automatic Image Processing

Images are automatically processed during ingestion and batch question answering:

```bash
# Ingestion (now includes images)
cd src
python app.py ingest

# Batch processing (now extracts images from question PDF)
python app.py batch
```

### Manual Image Processing

```python
from image_extraction import get_image_extractor
from image_processing import get_image_processor, get_image_cache

# Extract images
extractor = get_image_extractor()
images = extractor.extract_all_images("path/to/document.pdf")

# Process images
processor = get_image_processor()
cache = get_image_cache()

for image_meta in images:
    cached = cache.get(image_meta["image_path"])
    if cached:
        analysis = cached
    else:
        analysis = processor.process_image_complete(image_meta["image_path"])
        cache.set(image_meta["image_path"], analysis)
    
    print(f"Page {image_meta['page_number']}: {analysis['combined_text'][:200]}")
```

## Data Structure

### Image Metadata

```python
{
    "image_path": "/path/to/image.png",
    "image_filename": "doc_p1_img0_abc123.png",
    "page_number": 1,
    "image_index": 0,
    "coordinates": {
        "x0": 100, "y0": 200,
        "x1": 300, "y1": 400,
        "width": 200, "height": 200
    },
    "size": (200, 200),
    "extraction_method": "fitz",  # or "pdf2image"
    "source_pdf": "/path/to/source.pdf",
}
```

### Image Analysis Result

```python
{
    "image_path": "/path/to/image.png",
    "image_filename": "doc_p1_img0_abc123.png",
    "ocr_result": {
        "text": "extracted text from image",
        "confidence": 85,
        "method": "tesseract_ocr",
        "status": "success"
    },
    "azure_result": {
        "description": "image description",
        "objects": [{"name": "diagram", "confidence": 0.95}],
        "text": "text from Azure OCR",
        "tags": [{"name": "chart", "confidence": 0.88}],
        "status": "success"
    },
    "combined_text": "merged OCR + Azure vision results"
}
```

### Image Chunks in Vector Database

```python
{
    "text": "combined OCR and vision text",
    "section_id": "img_1_0",
    "section_title": "Image from Page 1",
    "domain": "Images & Diagrams",
    "source_doc": "source_filename.pdf",
    "page_start": 1,
    "page_end": 1,
    "image_path": "/path/to/image.png",
    "image_metadata": {...},
    "analysis_metadata": {
        "ocr_confidence": 85,
        "has_azure_analysis": true
    }
}
```

## File Organization

```
data/
├── extracted_images/           # Extracted images from PDFs
│   ├── doc_p1_img0_abc123.png
│   ├── doc_p2_img0_def456.png
│   └── ...
├── image_cache/                # Cached image analysis
│   ├── doc_p1_img0_abc123_cache.json
│   └── ...
├── processed_docs/
│   └── chunks.json             # Includes image chunks
└── vectordb/
    └── Chroma database         # Includes image embeddings
```

## Limitations & Considerations

1. **Performance**: Image processing (especially Azure Vision) adds processing time during ingestion
2. **Cost**: Azure Vision API calls incur costs - consider batch processing times
3. **Quality**: OCR quality depends on image resolution and clarity
4. **Storage**: Extracted images are stored locally - can consume significant disk space
5. **API Limits**: Azure Vision has rate limits - large batches may take time

## Troubleshooting

### OCR Not Working
- Ensure Tesseract is installed and PATH is set correctly
- For Windows, verify installation path matches environment variable

### Azure Vision API Errors
- Verify `AZURE_VISION_ENDPOINT` and `AZURE_VISION_API_KEY` are set
- Check Azure subscription has Computer Vision API enabled
- Verify API region matches endpoint

### Images Not Extracted
- Check PDF has embedded images (pdf2image can extract page images)
- Set `ENABLE_OCR_FALLBACK=1` if PDF is scanned
- Check file permissions for write access to `data/extracted_images/`

### Slow Ingestion
- Image processing adds time - normal for large documents
- Consider processing images separately with `image_extraction.py` directly
- Cache files prevent reprocessing on subsequent runs

## Performance Tips

1. **Cache Images**: Processed images are automatically cached - rerunning ingestion is faster
2. **Selective OCR**: Only use OCR fallback when needed (`ENABLE_OCR_FALLBACK=1`)
3. **Batch Processing**: Process multiple images in parallel if possible
4. **Regular Cleanup**: Remove old cache files to free disk space:
   ```bash
   rm -rf data/image_cache/
   rm -rf data/extracted_images/
   ```

## API Cost Estimation

### Azure Vision API Pricing (approximate)
- Read API (OCR): ~$1-3 per 1000 images
- Describe Images: ~$1-3 per 1000 images
- Analyze Images: ~$1-3 per 1000 images

Set up cost alerts in Azure portal to monitor spending.

## Future Enhancements

- [ ] Support for embedded vector images (multi-modal embeddings)
- [ ] Google Vision API support
- [ ] AWS Rekognition support
- [ ] Parallel image processing
- [ ] Image cropping and region-specific OCR
- [ ] Table structure recognition
- [ ] Formula/equation extraction
