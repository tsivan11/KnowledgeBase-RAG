# Document Type Testing Checklist

## Test Domain Setup
Create a new test domain: `kb/chunking_test/`

## 📋 Documents to Upload (One of Each Type)

### ✅ Text Documents
- [ ] **PDF** - Multi-page document with tables
  - Expected: ~1 chunk per page, tables preserved
  - Sample: Any research paper or report (3-5 pages)
  
- [ ] **DOCX** - Word document with paragraphs
  - Expected: Chunks respect paragraph boundaries
  - Sample: Any Word document with multiple paragraphs
  
- [ ] **TXT** - Plain text file
  - Expected: Sentence-aware splitting, no mid-sentence breaks
  - Sample: Any text file with multiple paragraphs
  
- [ ] **MD** - Markdown with headings
  - Expected: Chunks respect heading structure
  - Sample: Any README or documentation file
  
- [ ] **HTML** - Web page
  - Expected: Chunks by semantic sections
  - Sample: Any saved web page

### ✅ Data Files
- [ ] **CSV** - Comma-separated values
  - Expected: Each row = 1 chunk (or kept together if small)
  - Sample: Any spreadsheet saved as CSV
  
- [ ] **XLSX** - Excel spreadsheet
  - Expected: Each sheet processed, rows kept together
  - Sample: Any Excel file with 1-2 sheets

### ✅ Presentations
- [ ] **PPTX** - PowerPoint presentation
  - Expected: Each slide = separate chunk (or kept if small)
  - Sample: Any PowerPoint with 3-5 slides

### ✅ Images
- [ ] **JPG/PNG** - Image files
  - Expected: Single chunk per image (OCR/Vision output)
  - Sample: Any image with text or diagrams

### ✅ Audio (Optional - requires API key)
- [ ] **MP3/WAV** - Audio file
  - Expected: Transcript as sentence-aware chunks
  - Sample: Any short audio recording (<25MB)

## 🎯 Quick Test Commands

1. Create test domain folder:
```bash
mkdir kb/chunking_test
```

2. Copy sample files to the folder

3. Ingest documents:
```bash
python src/ingest_pdfs.py --domain chunking_test
```

4. Chunk the pages:
```bash
python src/chunk_pages.py --domain chunking_test
```

5. Analyze results:
```bash
python src/test_chunking.py --domain chunking_test
python src/verify_chunks.py --domain chunking_test
```

## 📊 What to Verify

For each document type, check:
- ✅ No mid-sentence breaks (except images/CSV)
- ✅ Chunk sizes are reasonable (not too small/large)
- ✅ Structure is preserved (pages, sections)
- ✅ No duplicate content (except in overlap)
- ✅ All content is captured (nothing lost)
