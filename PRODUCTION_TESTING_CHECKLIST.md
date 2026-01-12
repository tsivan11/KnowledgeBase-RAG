# 🚀 PRODUCTION TESTING CHECKLIST
## Complete Document Type Testing Guide

**Purpose**: Verify all 23 document types work correctly before production release  
**Test Domain**: Use `chunking_test` or create a new test domain  
**Goal**: Ensure robust, production-ready chunking for all supported formats

---

## 📋 COMPLETE TESTING MATRIX

### ✅ **Category 1: TEXT DOCUMENTS**

#### 1. PDF Files (`.pdf`)
**Upload**: 
- ✓ Multi-page PDF (3-5 pages)
- ✓ PDF with tables
- ✓ Scanned PDF (tests OCR fallback)

**What to Look For**:
- ✅ Chunks roughly align with pages (1-2 chunks per page typical)
- ✅ Tables are preserved and formatted (rows with `|` separators)
- ✅ No text is lost between pages
- ✅ Page numbers in metadata (`"page": 1, 2, 3...`)
- ✅ Chunk sizes reasonable (500-3000 chars per chunk)

**Potential Issues**:
- ⚠️ Scanned PDFs require Tesseract OCR installed
- ⚠️ Complex tables might have formatting issues
- ⚠️ Very large pages (>4000 chars) will be split

**Verification Command**:
```bash
python src/verify_chunks.py --domain chunking_test --samples --type pdf --count 5
```

---

#### 2. Word Documents (`.docx`)
**Upload**:
- ✓ DOCX with multiple paragraphs
- ✓ DOCX with embedded images (if Vision API enabled)

**What to Look For**:
- ✅ Paragraphs kept together (not split mid-paragraph)
- ✅ No awkward breaks
- ✅ If Vision API enabled: image descriptions appear as "=== EMBEDDED IMAGES ==="
- ✅ Chunk sizes: 500-2500 chars typical

**Potential Issues**:
- ⚠️ Without Vision API, images are ignored
- ⚠️ Very long paragraphs (>2000 chars) will be split at sentence boundaries
- ⚠️ Complex formatting (tables in Word) might lose structure

**Verification**:
```bash
python src/verify_chunks.py --domain chunking_test --samples --type docx
```

---

#### 3. Plain Text (`.txt`)
**Upload**:
- ✓ Text file with multiple paragraphs
- ✓ Long-form text (>5000 chars)

**What to Look For**:
- ✅ **CRITICAL**: No mid-sentence breaks (every chunk ends with `.`, `!`, or `?`)
- ✅ Paragraphs combined intelligently when small
- ✅ Sentence-aware splitting active
- ✅ Chunk sizes: 300-2000 chars

**Potential Issues**:
- ⚠️ Single-sentence paragraphs <100 chars might be filtered
- ⚠️ Very long sentences (>2000 chars) will be split at commas/semicolons

**Verification**:
Look at chunk endings:
```bash
python src/verify_chunks.py --domain chunking_test --samples --type txt --count 5
# Check that each chunk ends cleanly, not mid-sentence
```

---

#### 4. Markdown (`.md`)
**Upload**:
- ✓ Markdown with `#` headings and `##` subheadings
- ✓ README-style documentation

**What to Look For**:
- ✅ **CRITICAL**: Each heading section = separate chunk
- ✅ Section names preserved in metadata (`"section": "Heading Name"`)
- ✅ Heading hierarchy respected
- ✅ Code blocks preserved
- ✅ Each chunk is topically coherent

**Potential Issues**:
- ⚠️ Very long sections (>4000 chars) will be split
- ⚠️ Heading-only lines might create tiny chunks

**Verification**:
```bash
python src/test_chunking.py --domain chunking_test
# Look for MD chunks, verify section names match headings
```

---

#### 5. HTML (`.html`, `.htm`)
**Upload**:
- ✓ Saved web page
- ✓ HTML with semantic structure (h1, h2, article tags)

**What to Look For**:
- ✅ Scripts and styles removed
- ✅ Semantic sections preserved
- ✅ Heading hierarchy in metadata
- ✅ Text extracted cleanly (no HTML tags in chunks)

**Potential Issues**:
- ⚠️ Complex JavaScript-heavy pages might have minimal text
- ⚠️ Some formatting lost (expected)
- ⚠️ Navigation menus might create noise chunks

**Verification**:
```bash
python src/verify_chunks.py --domain chunking_test --samples --type html
```

---

### ✅ **Category 2: DATA FILES**

#### 6. CSV Files (`.csv`)
**Upload**:
- ✓ CSV with 5-10 rows
- ✓ Include headers

**What to Look For**:
- ✅ Each row formatted as: `Column1: value | Column2: value | ...`
- ✅ Headers included in context
- ⚠️ **EXPECTED**: Rows <100 chars will be FILTERED OUT
- ✅ Longer rows (>100 chars) kept as single chunks

**Potential Issues**:
- ⚠️ SHORT ROWS FILTERED: If your CSV rows are <100 chars, they won't appear in chunks
  - **Fix**: Lower `MIN_CHUNK_SIZE` in chunk_pages.py line 63 to `50`
- ⚠️ Very wide CSVs (many columns) might create large chunks

**Verification**:
```bash
# Check if CSV chunks exist
python src/verify_chunks.py --domain chunking_test --samples --type csv

# If empty, check pages.jsonl to see original row sizes
Get-Content data/chunking_test/pages.jsonl | Select-String "csv"
```

**Action if rows filtered**: Decide if you want to keep tiny chunks or accept the filtering.

---

#### 7. Excel Files (`.xlsx`, `.xls`)
**Upload**:
- ✓ Excel with 1-2 sheets
- ✓ Include some data rows
- ✓ If Vision API enabled: Excel with embedded charts/images

**What to Look For**:
- ✅ Each sheet processed separately
- ✅ Sheet name in metadata (`"section": "Sheet1"`)
- ✅ Rows formatted like CSV (Column: value pairs)
- ✅ If Vision API enabled: image descriptions included

**Potential Issues**:
- ⚠️ Same as CSV - short rows may be filtered
- ⚠️ Empty cells handled gracefully (should skip)
- ⚠️ Formulas evaluated to values (not formulas themselves)

**Verification**:
```bash
python src/verify_chunks.py --domain chunking_test --samples --type xlsx
```

---

### ✅ **Category 3: PRESENTATIONS**

#### 8. PowerPoint (`.pptx`)
**Upload**:
- ✓ PPTX with 3-5 slides
- ✓ Slides with text and images

**What to Look For**:
- ✅ Each slide = separate chunk (or kept together if small)
- ✅ Slide number in metadata (`"page": 1, 2, 3...`)
- ✅ Text from all shapes extracted
- ✅ If Vision API enabled: images described in "=== EMBEDDED IMAGES ==="
- ✅ Chunk size: slides up to 3000 chars kept intact

**Potential Issues**:
- ⚠️ Very text-heavy slides (>3000 chars) will be split
- ⚠️ Without Vision API, images ignored
- ⚠️ MAX_IMAGES_PER_FILE limit applies (default 5 images per file)

**Verification**:
```bash
python src/verify_chunks.py --domain chunking_test --samples --type pptx --count 5
# Verify slide numbers match your presentation
```

---

### ✅ **Category 4: IMAGES**

#### 9-11. Image Files (`.jpg`, `.jpeg`, `.png`)
**Upload**:
- ✓ Image with visible text (screenshot, diagram, infographic)
- ✓ Photo with objects/scenes

**What to Look For**:
- ✅ **WITH Vision API**: Detailed description of image content
- ✅ **WITH Tesseract**: Extracted text via OCR
- ✅ Each image = single chunk (atomic)
- ✅ Chunk not split (unless description >10,000 chars)

**Potential Issues**:
- ⚠️ **REQUIRES**: Either Vision API (USE_VISION_API=true) OR Tesseract OCR
- ⚠️ Without either: Images will fail to process
- ⚠️ Poor quality images = poor OCR results
- ⚠️ Vision API costs apply per image

**Setup Required**:
```bash
# For Vision API:
# Set in .env:
USE_VISION_API=true
OPENAI_API_KEY=your_key_here
VISION_MODEL=gpt-4o-mini

# OR for Tesseract:
# Install from: https://github.com/UB-Mannheim/tesseract/wiki
# Set in .env:
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

**Verification**:
```bash
python src/verify_chunks.py --domain chunking_test --samples --type jpg
python src/verify_chunks.py --domain chunking_test --samples --type png
# Check that descriptions are meaningful
```

---

### ✅ **Category 5: AUDIO FILES**

#### 15-19. Audio (`.mp3`, `.mpeg`, `.mpga`, `.m4a`, `.wav`)
**Upload**:
- ✓ Short audio file (<25MB, ideally <5MB for testing)
- ✓ Clear speech recording

**What to Look For**:
- ✅ Transcript generated via Whisper API
- ✅ Sentence-aware chunking applied to transcript
- ✅ No mid-sentence breaks
- ✅ Overlap of 400 chars (higher than other types for continuity)

**Potential Issues**:
- ⚠️ **REQUIRES**: OpenAI API key with Whisper access
- ⚠️ **SIZE LIMIT**: 25MB max file size
- ⚠️ **COST**: Whisper API charges per minute
- ⚠️ Poor audio quality = poor transcription
- ⚠️ Background noise affects accuracy

**Setup Required**:
```bash
# Set in .env:
OPENAI_API_KEY=your_key_here
```

**Verification**:
```bash
python src/verify_chunks.py --domain chunking_test --samples --type audio
# Read transcript to verify accuracy
```

---

## 🔍 COMPREHENSIVE TESTING WORKFLOW

### Step 1: Prepare Test Files
Create a test set with **at least one file from each category**:

**Minimum Test Set** (7 files):
1. PDF (multi-page)
2. TXT (long text)
3. MD (with headings)
4. CSV (data table)
5. XLSX (spreadsheet)
6. PPTX (presentation)
7. JPG/PNG (image)

**Full Test Set** (15+ files):
- Add: DOCX, HTML, multiple image formats, audio file

---

### Step 2: Upload via UI
1. Create new domain in UI: "production_test"
2. Upload all test files
3. Let system process (ingest → chunk → index)

---

### Step 3: Verify Processing
```bash
# Check ingestion
python src/test_chunking.py --domain production_test

# Verify chunk quality
python src/verify_chunks.py --domain production_test

# Sample each type
python src/verify_chunks.py --domain production_test --samples --type pdf --count 3
python src/verify_chunks.py --domain production_test --samples --type txt --count 3
# ... repeat for each type
```

---

### Step 4: Quality Checks

#### ✅ Check 1: No Data Loss
```bash
# Compare input vs output
# Should see reasonable ratios (0.5x - 2x typical)
python src/test_chunking.py --domain production_test
```

#### ✅ Check 2: No Mid-Sentence Breaks
```bash
# Run verification
python src/verify_chunks.py --domain production_test

# Look for: "[+] Quality: Good" for txt, md, html, docx, pdf types
# Should see: "0 potential mid-sentence breaks"
```

#### ✅ Check 3: Structure Preserved
- **Markdown**: Section names in chunk metadata
- **PDF**: Page numbers tracked
- **PPTX**: Slide numbers tracked
- **Excel**: Sheet names preserved

#### ✅ Check 4: Chunk Size Distribution
```bash
# All chunks should be in reasonable ranges:
# - Minimum: 100 chars (or filtered)
# - Maximum: 5000 chars (very rare)
# - Typical: 200-2000 chars
# - Images: Can be larger (up to 10,000)
```

#### ✅ Check 5: Test Retrieval
Use your app to query:
- "What is [specific fact from your PDF]?"
- "Tell me about [topic from your markdown]?"
- "What data is in [your CSV]?"

Verify chunks are retrieved correctly.

---

## ⚠️ KNOWN ISSUES & LIMITATIONS

### Issue 1: CSV Rows Too Short
**Symptom**: CSV chunks don't appear  
**Cause**: Rows <100 chars filtered by MIN_CHUNK_SIZE  
**Fix**: 
```python
# In chunk_pages.py line 63, change:
MIN_CHUNK_SIZE = 50  # Instead of 100
```
**Impact**: Will create more (smaller) chunks

---

### Issue 2: Images Not Processing
**Symptom**: Image files fail with errors  
**Cause**: Neither Vision API nor Tesseract configured  
**Fix**: Choose one:
```bash
# Option A: Vision API (better quality, costs money)
USE_VISION_API=true
OPENAI_API_KEY=sk-...

# Option B: Tesseract OCR (free, text-only)
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

---

### Issue 3: Audio Not Transcribing
**Symptom**: Audio files fail or skip  
**Cause**: No OpenAI API key or file >25MB  
**Fix**: 
1. Set `OPENAI_API_KEY=sk-...`
2. Keep files <25MB
3. Compress large files before upload

---

### Issue 4: Vision API Hit Limits
**Symptom**: Some images in DOCX/PPTX/XLSX not processed  
**Cause**: MAX_IMAGES_PER_FILE limit (default: 5)  
**Fix**:
```bash
# In .env, increase limit:
MAX_IMAGES_PER_FILE=20
```
**Warning**: Costs scale with image count

---

### Issue 5: Very Large Documents
**Symptom**: Processing takes very long or fails  
**Cause**: Single file >100MB or >1000 pages  
**Fix**: 
- Split large PDFs into smaller files
- Increase processing timeouts if needed
- Consider batching large datasets

---

## ✅ PRE-PRODUCTION CHECKLIST

Before going live, verify:

- [ ] **All document types tested** (minimum 7 types)
- [ ] **No critical errors** in verify_chunks.py output
- [ ] **Text quality good** (no mid-sentence breaks)
- [ ] **Structure preserved** (headings, pages, sections tracked)
- [ ] **Retrieval works** (tested queries return relevant chunks)
- [ ] **API keys configured** (if using Vision/Whisper)
- [ ] **File size limits documented** (25MB for audio, etc.)
- [ ] **Error handling tested** (try uploading corrupted file)
- [ ] **README updated** with supported formats
- [ ] **.env.example updated** with all required variables

---

## 🎯 MINIMUM VIABLE TEST (Quick Version)

**If short on time, test these 5 types minimum:**

1. **PDF** - Most common document type
2. **TXT** - Tests sentence-aware splitting  
3. **MD** - Tests structural chunking
4. **CSV** - Tests data handling (note size filter issue)
5. **JPG** - Tests image processing (if Vision API enabled)

**Commands**:
```bash
# Upload files via UI to "quick_test" domain

# Verify
python src/verify_chunks.py --domain quick_test
python src/test_chunking.py --domain quick_test

# If all show "Good" quality → Ready to ship!
```

---

## 📊 SUCCESS CRITERIA

**You're ready for production when:**

✅ At least 5 document types tested successfully  
✅ Zero critical issues in verification  
✅ No mid-sentence breaks in text documents  
✅ Chunk sizes in reasonable ranges (100-5000 chars)  
✅ Retrieval returns relevant results  
✅ No data loss (all content chunked or intentionally filtered)  

---

**Good luck with testing! 🚀**

*Last Updated: 2026-01-08*  
*System: KnowledgeBase-RAG v2.0*
