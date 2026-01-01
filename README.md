# KnowledgeBase-RAG

A multi-domain Retrieval-Augmented Generation (RAG) system that transforms your documents into an intelligent, queryable knowledge base. Supports 13+ file formats with advanced features like OCR, table extraction, and semantic search.

## ✨ Features

- **Multi-Domain Support**: Organize documents into separate knowledge domains (e.g., contracts, astronomy, research)
- **13+ File Formats**: PDF, DOCX, TXT, MD, HTML, CSV, XLSX, PPTX, JPG, PNG, TIFF, BMP
- **Advanced PDF Processing**: 
  - Structured table extraction with pdfplumber
  - OCR fallback for scanned documents
  - Automatic text extraction
- **Semantic Search**: Powered by OpenAI embeddings and FAISS vector database
- **Citation-Based Answers**: AI-generated responses with source citations
- **Recursive File Discovery**: Automatically processes all files in domain folders and subdirectories

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key
- (Optional) Tesseract OCR for image text extraction

### Installation

1. Clone the repository:
```bash
git clone https://github.com/tsivan11/KnowledgeBase-RAG.git
cd KnowledgeBase-RAG
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_api_key_here
```

4. (Optional) Install Tesseract OCR:
   - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
   - Mac: `brew install tesseract`
   - Linux: `sudo apt-get install tesseract-ocr`

### Usage

#### Option 1: Web Interface (Recommended)

1. Start the web server:
```bash
python app.py
```

2. Open your browser to `http://localhost:8000`

3. Use the web interface to:
   - Create and manage domains
   - Upload documents (drag & drop or click to browse)
   - Ask questions and view answers with citations
   - Monitor indexing status

#### Option 2: Command Line Interface

##### 1. Add Documents to a Domain

Create a domain folder and add your files:
```bash
mkdir -p kb/my_domain
# Copy your documents into kb/my_domain/
```

##### 2. Process Documents

Run the complete pipeline:
```bash
# Ingest documents
python src/ingest_pdfs.py --domain my_domain

# Create text chunks
python src/chunk_pages.py --domain my_domain

# Build vector index
python src/build_index.py --domain my_domain
```

##### 3. Query Your Knowledge Base

Interactive mode:
```bash
python src/ask.py --domain my_domain
```

Direct question mode:
```bash
python src/ask.py --domain my_domain --question "What is the main topic of this document?"
```

## 📁 Project Structure

```
KnowledgeBase-RAG/
├── app.py                       # FastAPI web server
├── static/                      # Web interface
│   ├── index.html              # Frontend UI
│   ├── styles.css              # Styling
│   └── app.js                  # Client logic
├── kb/                          # Knowledge base (your documents)
│   ├── domain1/
│   ├── domain2/
│   └── ...
├── data/                        # Processed data (auto-generated)
│   ├── domain1/
│   │   ├── pages.jsonl
│   │   ├── chunks.jsonl
│   │   ├── faiss.index
│   │   └── chunks_meta.json
│   └── ...
├── src/                         # Source code
│   ├── loaders.py              # File format handlers
│   ├── ingest_pdfs.py          # Document ingestion
│   ├── chunk_pages.py          # Text chunking
│   ├── build_index.py          # Vector index creation
│   └── ask.py                  # Query interface
├── requirements.txt
├── .gitignore
└── README.md
```

## 🔧 How It Works

1. **Document Ingestion**: Extracts text from various file formats, handling tables and images
2. **Chunking**: Splits documents into 2000-character chunks with 300-character overlap
3. **Embedding**: Converts chunks to vectors using OpenAI's text-embedding-3-small model
4. **Indexing**: Stores vectors in FAISS for fast similarity search
5. **Retrieval**: Finds the 8 most relevant chunks for each query
6. **Generation**: Uses GPT-4o-mini to generate answers with citations

## 📊 Supported File Formats

| Format | Extension | Features |
|--------|-----------|----------|
| PDF | `.pdf` | Table extraction, OCR fallback |
| Word | `.docx` | Paragraph extraction |
| Text | `.txt`, `.md` | Section-based parsing |
| HTML | `.html`, `.htm` | Heading-aware extraction |
| Spreadsheet | `.csv`, `.xlsx`, `.xls` | Row-wise ingestion |
| PowerPoint | `.pptx` | Slide-by-slide extraction |
| Image | `.jpg`, `.png`, `.tiff`, `.bmp` | OCR text extraction |

## 🎯 Use Cases

- **Legal Document Analysis**: Process contracts, agreements, and legal documents
- **Research Knowledge Base**: Organize academic papers and research materials
- **Technical Documentation**: Query API docs, manuals, and guides
- **Educational Content**: Create searchable study materials
- **Business Intelligence**: Analyze reports, presentations, and data

## ⚙️ Configuration

### Chunking Parameters
Edit `src/chunk_pages.py`:
```python
CHUNK_SIZE = 2000      # characters
CHUNK_OVERLAP = 300    # characters
```

### Retrieval Settings
Edit `src/ask.py`:
```python
TOP_K = 8              # number of chunks to retrieve
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
```

## 🛠️ Advanced Features

### Custom OCR Path
Set Tesseract path in `.env`:
```
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

### Subdomain Organization
Organize files in subdirectories within domains:
```
kb/
  research/
    machine-learning/
      paper1.pdf
      paper2.pdf
    nlp/
      paper3.pdf
```
All files are automatically discovered and processed.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests

## 📝 License

This project is open source and available under the MIT License.

## 🔗 Links

- Repository: https://github.com/tsivan11/KnowledgeBase-RAG
- Issues: https://github.com/tsivan11/KnowledgeBase-RAG/issues

## 🙏 Acknowledgments

Built with:
- [OpenAI](https://openai.com/) - Embeddings and language models
- [FAISS](https://github.com/facebookresearch/faiss) - Vector similarity search
- [PyMuPDF](https://pymupdf.readthedocs.io/) - PDF processing
- [pdfplumber](https://github.com/jsvine/pdfplumber) - Table extraction
- [pytesseract](https://github.com/madmaze/pytesseract) - OCR capabilities
>>>>>>> 10aefec (Front end and batch file)
