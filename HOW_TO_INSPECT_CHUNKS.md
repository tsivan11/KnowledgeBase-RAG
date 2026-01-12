# 🔍 How to Inspect Chunks After UI Upload

## Quick Answer
After uploading documents via the UI, you have **3 ways** to inspect the generated chunks:

---

## ✅ **Method 1: Use Verification Scripts (Recommended)**

### Check Overall Quality
```bash
python src/verify_chunks.py --domain your_domain_name
```
**Shows:**
- Statistics by document type
- Chunk size ranges
- Quality issues (mid-sentence breaks, etc.)
- Overall health report

### View Sample Chunks
```bash
python src/verify_chunks.py --domain your_domain_name --samples --type pdf --count 5
```
**Shows:**
- Actual chunk content
- Chunk IDs
- Metadata (page, section)
- Size

### Detailed Analysis
```bash
python src/test_chunking.py --domain your_domain_name --samples 3
```
**Shows:**
- Breakdown by document type
- Input/output ratios
- Sample chunks for each type
- Comprehensive statistics

---

## ✅ **Method 2: Open Files Directly (Quick Inspection)**

### Location
All chunks are stored in: `data/<domain_name>/chunks.jsonl`

### Open in Text Editor
```bash
# Windows
notepad data\your_domain_name\chunks.jsonl

# Or use VS Code
code data\your_domain_name\chunks.jsonl
```

### File Format
Each line is a JSON object:
```json
{
  "chunk_id": "document.pdf::p1::c0",
  "source": "document.pdf",
  "source_type": "pdf",
  "page": 1,
  "section": null,
  "text": "The actual chunk text goes here..."
}
```

**Fields:**
- `chunk_id`: Unique identifier (source::page::chunk_number)
- `source`: Original filename
- `source_type`: Document type (pdf, txt, csv, etc.)
- `page`: Page number (if applicable)
- `section`: Section name or row number
- `text`: The actual chunk content

### Quick Search in File
Use your editor's search (Ctrl+F) to find:
- Specific filename: Search for `"source": "myfile.pdf"`
- Specific type: Search for `"source_type": "csv"`
- Content: Search for keywords in the text field

---

## ✅ **Method 3: Command Line Inspection**

### Count Chunks
```bash
# Windows PowerShell
Get-Content data\your_domain\chunks.jsonl | Measure-Object -Line

# Count by type
Get-Content data\your_domain\chunks.jsonl | Select-String '"source_type": "pdf"' | Measure-Object -Line
```

### View Specific Chunks
```bash
# First 5 chunks
Get-Content data\your_domain\chunks.jsonl -TotalCount 5

# Search for specific file
Get-Content data\your_domain\chunks.jsonl | Select-String "myfile.pdf"

# Pretty print a chunk (requires Python)
Get-Content data\your_domain\chunks.jsonl -TotalCount 1 | python -m json.tool
```

---

## 📊 **Recommended Workflow After UI Upload**

### 1. Quick Health Check
```bash
python src/verify_chunks.py --domain your_domain
```
**Look for:**
- `[+] Quality: Good` for each document type
- `0 issues found` in the report

### 2. Sample Inspection
```bash
# Check your most important document type
python src/verify_chunks.py --domain your_domain --samples --type pdf --count 5
```
**Verify:**
- Chunks make sense
- No weird breaks
- Content is complete

### 3. Test Retrieval
Use your app to ask questions:
- Query about specific facts from your documents
- Verify the right chunks are retrieved
- Check answers are accurate

---

## 🎯 **What to Look For (Quality Checklist)**

### ✅ Good Chunks
- End at natural boundaries (sentences, paragraphs, sections)
- Contain complete thoughts
- Size is reasonable (200-2000 chars typical)
- Metadata is correct (page numbers, sections)
- No missing content

### ⚠️ Warning Signs
- Chunks ending mid-sentence: "...and then the"
- Chunks starting with partial words: "ystem works by..."
- Duplicate content (beyond normal overlap)
- Empty or very short chunks (<30 chars for non-CSV)
- Missing chunks (some content not processed)

---

## 🔍 **Specific Document Type Checks**

### PDF Files
```bash
python src/verify_chunks.py --domain your_domain --samples --type pdf --count 3
```
**Check:**
- Page numbers in metadata match actual pages
- Tables are formatted (with `|` separators)
- No pages missing

### CSV Files
```bash
python src/verify_chunks.py --domain your_domain --samples --type csv
```
**Check:**
- Each row formatted as: `Column: value | Column: value`
- All rows appear (should be ≥30 chars each)
- Headers make sense

### Markdown Files
```bash
python src/verify_chunks.py --domain your_domain --samples --type md
```
**Check:**
- Section names in metadata match your headings
- Each heading section is separate chunk
- Code blocks preserved

### Images
```bash
python src/verify_chunks.py --domain your_domain --samples --type jpg
```
**Check:**
- Descriptions are meaningful
- Text from images extracted (if OCR enabled)
- One chunk per image

---

## 💡 **Pro Tips**

### Compare Before/After
If you modify chunking settings:
```bash
# Save before
cp data/your_domain/chunks.jsonl data/your_domain/chunks_backup.jsonl

# Make changes, re-chunk

# Compare
python src/test_chunking.py --domain your_domain
```

### Export for Review
```bash
# Export all chunks from one file to readable format
python -c "import json; [print(f\"\\n=== CHUNK {i} ===\\n{json.loads(line)['text']}\\n\") for i, line in enumerate(open('data/your_domain/chunks.jsonl'))]" > review.txt
```

### Quick Stats
```bash
# Average chunk size
python -c "import json; chunks = [len(json.loads(line)['text']) for line in open('data/your_domain/chunks.jsonl')]; print(f'Avg: {sum(chunks)/len(chunks):.0f} chars, Min: {min(chunks)}, Max: {max(chunks)}')"
```

---

## 🚨 **Troubleshooting**

### Issue: Can't find chunks.jsonl
**Cause**: Domain not processed yet  
**Fix**: 
```bash
# Check if pages.jsonl exists
ls data/your_domain/

# If exists, run chunking
python src/chunk_pages.py --domain your_domain
```

### Issue: Chunks file is empty
**Cause**: All chunks filtered (too small)  
**Fix**: Check verify_chunks.py output, may need to lower MIN_CHUNK_SIZE

### Issue: Chunks look wrong
**Cause**: May need to re-ingest and re-chunk  
**Fix**:
```bash
# Re-process from scratch
python src/ingest_pdfs.py --domain your_domain
python src/chunk_pages.py --domain your_domain
python src/build_index.py --domain your_domain
```

---

## 📝 **Example Session**

```bash
# 1. Upload files via UI to domain "contracts"

# 2. Quick check
python src/verify_chunks.py --domain contracts
# Output: "[+] Quality: Good" ✓

# 3. Look at PDF chunks
python src/verify_chunks.py --domain contracts --samples --type pdf --count 3
# Output: Shows 3 sample PDF chunks with content

# 4. Open file to browse all
code data\contracts\chunks.jsonl

# 5. Test in app
# Ask: "What is the termination clause?"
# Verify: Correct chunk retrieved
```

---

**You now have full visibility into your chunking pipeline!** 🎉
