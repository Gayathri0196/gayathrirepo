# Quick Start: Image Processing Features

## Installation

### 1. Update Dependencies
```powershell
cd C:\Users\M.Devi\Desktop\gayathrirepo
pip install -r requirements.txt --upgrade
```

### 2. Install Tesseract (Optional, for OCR)

**Windows:**
1. Download from: https://github.com/UB-Mannheim/tesseract/wiki
2. Run installer (default path recommended)
3. Test installation:
   ```powershell
   tesseract --version
   ```

### 3. Configure Azure Vision (Optional, for advanced analysis)

Update your `.env` file:
```env
AZURE_VISION_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
AZURE_VISION_API_KEY=your-key-here
```

## First Run

### 1. Prepare PDFs with Images
- Add PDFs with images to `input_files/`
- Supported: Regular PDFs with embedded images, scanned PDFs

### 2. Run Ingestion
```powershell
cd src
python app.py ingest
```

**What happens:**
- ✅ Text extraction (existing)
- ✅ Image extraction to `../data/extracted_images/`
- ✅ OCR processing of images
- ✅ Azure Vision analysis (if configured)
- ✅ Image caching to `../data/image_cache/`
- ✅ Image chunks stored in vector database

### 3. Test with Chat
```powershell
python app.py chat
```

Try asking: "What diagrams or images are in the document?"

### 4. Batch Processing
```powershell
python app.py batch
```

Check results in `output_files/` - now includes image context!

## What Gets Created

```
data/
├── extracted_images/
│   ├── lms_test_plan_sample_p1_img0_abc123.png
│   ├── lms_test_plan_sample_p2_img0_def456.png
│   └── ...
├── image_cache/
│   ├── lms_test_plan_sample_p1_img0_abc123_cache.json
│   └── ...
└── processed_docs/
    └── chunks.json
        (now includes image chunks with OCR text)
```

## How to Use

### In Chat Mode
```
You: What information is shown in the diagrams on page 2?

Agent: The diagrams show...
[Image context automatically included]
```

### In Batch Mode
- Questions are answered with image context included
- Image descriptions provided to LLM
- Better understanding of visual content

## Troubleshooting

### Images not extracted?
- ✅ Check PDF has embedded images
- ✅ For scanned PDFs, ensure good image quality
- ✅ Check write permissions on `data/extracted_images/`

### OCR errors?
- ✅ Install Tesseract properly
- ✅ Verify `tesseract` command works in terminal
- ✅ Images need sufficient resolution (>100 DPI)

### Azure Vision errors?
- ✅ Check credentials in `.env`
- ✅ Verify API is enabled in Azure portal
- ✅ Check rate limits haven't been exceeded

### Slow ingestion?
- ✅ Normal - first run processes all images
- ✅ Second run uses cache (much faster)
- ✅ Can take 5-15 seconds per image

## Configuration Options

### Disable Image Processing
Comment out in `src/ingestion.py`:
```python
# image_metadata_list = image_extractor.extract_all_images(pdf_path)
```

### Enable OCR Fallback
```env
ENABLE_OCR_FALLBACK=1
```

### Clean Cache (start fresh)
```powershell
Remove-Item data/image_cache -Recurse -Force
Remove-Item data/extracted_images -Recurse -Force
python app.py ingest  # Reprocess all
```

## Features

| Feature | Status | Notes |
|---------|--------|-------|
| Extract embedded images | ✅ | PyMuPDF |
| Extract page images | ✅ | pdf2image (scanned PDFs) |
| Tesseract OCR | ✅ | Requires installation |
| Azure Vision API | ✅ | Optional, requires credentials |
| Image caching | ✅ | Automatic |
| Vector DB integration | ✅ | Image chunks searchable |
| Chat mode enhancement | ✅ | Auto includes image context |
| Batch mode enhancement | ✅ | Auto includes image context |

## Next Steps

1. **Test with your PDFs**: Place PDFs with images in `input_files/`
2. **Run ingestion**: `python app.py ingest`
3. **Ask image-related questions**: "What is shown in the diagrams?"
4. **Check extracted images**: View `data/extracted_images/` 
5. **Review performance**: Check logs for processing times

## Support

For issues or questions:
1. Check logs in `logs/app.log`
2. Review `IMAGE_PROCESSING.md` for detailed documentation
3. Verify all dependencies are installed
4. Check environment variables are set correctly

## Example Use Cases

1. **Technical Diagrams**: Understand architecture, flow diagrams
2. **Flowcharts**: Extract process flows from images
3. **Tables**: Extract table content via OCR
4. **Screenshots**: Understand system interface content
5. **Charts**: Analyze data visualizations

---

**Happy image processing! 🎉**
