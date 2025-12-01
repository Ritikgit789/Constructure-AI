# Project Brain - Construction Document RAG System 🏗️

A powerful **multimodal RAG (Retrieval-Augmented Generation)** system for construction project document management. Built with **Gemini 1.5 Pro** for vision understanding and **FAISS** for fast vector search.

## 🌟 Features

### Core Capabilities
- **📄 Multimodal Document Ingestion**: Process PDFs with both text and images (drawings, schedules, diagrams)
- **🔍 Intelligent RAG Pipeline**: Hybrid search combining vector similarity and keyword matching
- **🤖 Gemini Vision**: Automatic description generation for construction drawings and diagrams
- **💬 Context-Aware Chat**: Ask questions with accurate citations from source documents
- **📊 Structured Extraction**: Extract door schedules, room schedules, and equipment lists
- **✅ Quality Evaluation**: Built-in evaluation framework for RAG accuracy

### Technical Highlights
- **Gemini 1.5 Pro**: Latest multimodal LLM for understanding text + images
- **FAISS Vector Store**: Lightning-fast similarity search
- **FastAPI Backend**: Modern async API with automatic documentation
- **Advanced Chunking**: Context-aware document splitting with page boundaries
- **Multi-Document Support**: Search across entire project document sets

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))
- 2GB+ RAM (for FAISS indexing)

### Installation

```bash
# Clone the repository
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

1. Copy the environment template:
```bash
copy .env.example .env
```

2. Edit `.env` and add your Gemini API key:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-1.5-pro-latest
GEMINI_EMBEDDING_MODEL=models/embedding-001
```

### Running the Backend

```bash
# From the backend directory
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

---

## 📚 API Usage

### 1. Upload Documents

```bash
curl -X POST "http://localhost:8000/api/documents/upload" \
  -F "file=@path/to/your/document.pdf" \
  -F "document_type=drawing"
```

**Document Types**: `drawing`, `specification`, `schedule`, `general`

### 2. Ask Questions (RAG Chat)

```bash
curl -X POST "http://localhost:8000/api/chat/query" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are the fire rating requirements for doors?",
    "include_images": true,
    "max_sources": 5
  }'
```

**Response includes**:
- AI-generated answer
- Source citations (file + page)
- Relevance scores
- Image references (if applicable)

### 3. Extract Structured Data

**Door Schedule**:
```bash
curl -X POST "http://localhost:8000/api/extract/door-schedule"
```

**Room Schedule**:
```bash
curl -X POST "http://localhost:8000/api/extract/room-schedule"
```

**Returns**: Structured JSON with:
- Extracted entities (doors/rooms)
- Specifications and measurements
- Source citations

### 4. List Documents

```bash
curl "http://localhost:8000/api/documents/list"
```

### 5. Run Evaluation

```bash
curl -X POST "http://localhost:8000/api/evaluate/run"
```

**Tests**:
- Factual Q&A accuracy
- Citation correctness
- Multimodal understanding
- Extraction quality

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        FastAPI Server                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Documents   │  │     Chat     │  │  Extraction  │      │
│  │   Routes     │  │    Routes    │  │    Routes    │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                  │                  │               │
│         └──────────────────┼──────────────────┘              │
│                            │                                  │
│         ┌──────────────────▼──────────────────┐              │
│         │      RAG Retrieval Service          │              │
│         │  (Hybrid Search + Re-ranking)       │              │
│         └──────────────┬─────────────────────┘              │
│                        │                                      │
│         ┌──────────────┼──────────────┐                     │
│         │              │               │                     │
│    ┌────▼────┐   ┌────▼────┐   ┌─────▼────┐               │
│    │  FAISS  │   │ Gemini  │   │ Document │               │
│    │ Vector  │   │ Service │   │ Chunker  │               │
│    │  Store  │   │(Vision) │   │          │               │
│    └─────────┘   └─────────┘   └──────────┘               │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Pipeline Flow

**1. Document Ingestion**:
```
PDF → Text Extraction → Image Extraction → 
Vision Description (Gemini) → Chunking → 
Embedding Generation → FAISS Storage
```

**2. RAG Query**:
```
User Query → Query Embedding → 
Hybrid Search (Vector + Keyword) → Re-ranking → 
Context Assembly (Text + Images) → 
Gemini Response → Citations
```

**3. Structured Extraction**:
```
Extraction Request → Targeted Retrieval → 
Context Assembly → Gemini Structured Output → 
JSON Validation → Response
```

---

## 📁 Project Structure

```
backend/
├── app/
│   ├── main.py                 # FastAPI application entry point
│   ├── config.py               # Configuration and settings
│   ├── models/
│   │   └── schemas.py          # Pydantic models (requests/responses)
│   ├── services/
│   │   ├── gemini_service.py   # Gemini API integration (vision + chat)
│   │   ├── ingestion.py        # Document ingestion pipeline
│   │   ├── retrieval.py        # RAG retrieval logic
│   │   └── extraction.py       # Structured extraction
│   ├── db/
│   │   └── vector_store.py     # FAISS vector database
│   ├── utils/
│   │   └── chunking.py         # Document chunking strategies
│   └── api/
│       └── routes/
│           ├── documents.py    # Document upload/management
│           ├── chat.py         # Chat Q&A endpoints
│           ├── extract.py      # Structured extraction endpoints
│           └── evaluate.py     # Evaluation endpoints
├── data/
│   ├── uploads/                # Uploaded PDFs
│   ├── extracted_images/       # Extracted construction drawings
│   └── faiss_index/            # FAISS vector index storage
├── evaluation/
│   ├── test_queries.json       # Test questions for evaluation
│   └── latest_results.json     # Evaluation results
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
└── .env.example                # Environment variables template
```

---

## 🧪 Testing with Sample PDFs

The repository includes sample construction documents in `tests/`:
- Construction Drawings (508-22-105)
- DBA Wages Document

**Ingest the sample PDFs**:

```bash
# Using curl
cd ../tests
curl -X POST "http://localhost:8000/api/documents/upload" \
  -F "file=@Attachment+7+-+Construction+Drawings+508-22-105.pdf" \
  -F "document_type=drawing"

curl -X POST "http://localhost:8000/api/documents/upload" \
  -F "file=@Attachment+5+-+DBA+Wages+GA20250305+01-03-2025 (2).pdf" \
  -F "document_type=specification"
```

**Try sample queries**:

```bash
# General question
curl -X POST "http://localhost:8000/api/chat/query" \
  -H "Content-Type: application/json" \
  -d '{"message": "What documents are available?", "include_images": true}'

# Document-specific question
curl -X POST "http://localhost:8000/api/chat/query" \
  -H "Content-Type: application/json" \
  -d '{"message": "What construction drawings are included?", "include_images": true}'

# Extraction
curl -X POST "http://localhost:8000/api/extract/door-schedule"
```

---

## 🎯 Evaluation Framework

The system includes a built-in evaluation framework to test RAG quality.

**Run evaluation**:
```bash
curl -X POST "http://localhost:8000/api/evaluate/run"
```

**Evaluation metrics**:
- ✅ **Keyword Match**: Are expected terms in the response?
- 📚 **Source Correctness**: Are the right documents cited?
- 🎯 **Overall Accuracy**: Combined confidence score

**Test queries** are defined in `evaluation/test_queries.json`:
- Factual Q&A questions
- Multimodal questions (requiring image understanding)
- Extraction tasks

**Results** are saved to `evaluation/latest_results.json` with:
- Individual query scores
- Overall accuracy percentage
- Detailed notes on matches/misses

---

## 🐳 Docker Deployment

```bash
# Build Docker image
docker build -t project-brain-backend .

# Run container
docker run -p 8000:8000 \
  -e GOOGLE_API_KEY=your_api_key_here \
  -v $(pwd)/data:/app/data \
  project-brain-backend
```

**Persistent storage**: Mount the `data/` directory to preserve:
- Uploaded documents
- Extracted images
- FAISS index

---

## 🌐 Deployment Options

### Option 1: Railway (Recommended)

1. Create account at [railway.app](https://railway.app)
2. Create new project
3. Add GitHub repo
4. Add environment variables:
   ```
   GOOGLE_API_KEY=your_key
   PORT=8000
   ```
5. Deploy!

### Option 2: Render

1. Create account at [render.com](https://render.com)
2. New Web Service
3. Connect GitHub repo
4. Build Command: `pip install -r requirements.txt`
5. Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
6. Add environment variables

### Option 3: Local with Tunnel (Testing)

```bash
# Install ngrok
# Run backend
python -m uvicorn app.main:app --port 8000

# In another terminal
ngrok http 8000
```

---

## 🔧 Configuration Options

### Environment Variables

```env
# Gemini API
GOOGLE_API_KEY=your_key
GEMINI_MODEL=gemini-1.5-pro-latest
GEMINI_EMBEDDING_MODEL=models/embedding-001

# Application
DEBUG=false
LOG_LEVEL=INFO
ALLOWED_ORIGINS=http://localhost:3000

# Upload
MAX_UPLOAD_SIZE=100000000   # 100MB
UPLOAD_DIR=./data/uploads
EXTRACTED_IMAGES_DIR=./data/extracted_images

# Vector Store (FAISS)
FAISS_INDEX_PATH=./data/faiss_index
VECTOR_DIMENSION=768        # Gemini embedding dimension

# RAG Configuration
TOP_K_RETRIEVAL=10          # Initial retrieval count
RERANK_TOP_K=5              # Final context count
CHUNK_SIZE=1000             # Text chunk size
CHUNK_OVERLAP=200           # Chunk overlap
MAX_IMAGE_SIZE=2048         # Max image dimension for vision

# Server
HOST=0.0.0.0
PORT=8000
```

---

## 📊 Performance Considerations

### Ingestion Speed
- **Text extraction**: ~1-2 pages/second
- **Image extraction**: ~0.5-1 images/second
- **Vision description**: ~2-3 seconds per image (Gemini API)
- **Embedding generation**: ~10-20 chunks/second

**Optimization tips**:
- Batch processing for multiple documents
- Async image description (concurrent API calls)
- Caching for frequently accessed documents

### Retrieval Speed
- **FAISS search**: <100ms for 10K vectors
- **Hybrid search**: ~150-300ms
- **Gemini response**: ~1-3 seconds

### Storage
- **FAISS index**: ~3KB per vector (768-dim)
- **Images**: Original size (varies)
- **Metadata**: ~1KB per chunk

---

## 🤝 API Reference

### Complete Endpoint List

**Documents**:
- `POST /api/documents/upload` - Upload PDF
- `GET /api/documents/list` - List all documents
- `GET /api/documents/stats` - Vector store statistics

**Chat**:
- `POST /api/chat/query` - Ask question (main RAG endpoint)
- `GET /api/chat/conversation/{id}` - Get conversation history
- `DELETE /api/chat/conversation/{id}` - Delete conversation

**Extraction**:
- `POST /api/extract/door-schedule` - Extract door schedule
- `POST /api/extract/room-schedule` - Extract room schedule
- `POST /api/extract/generic` - Generic extraction

**Evaluation**:
- `POST /api/evaluate/run` - Run evaluation suite
- `GET /api/evaluate/results` - Get latest results

**System**:
- `GET /` - API info
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation

---

## 🐛 Troubleshooting

### Common Issues

**1. Gemini API Rate Limits**
```
Error: Resource exhausted
Solution: Add delays between API calls, reduce batch sizes
```

**2. Memory Issues with Large PDFs**
```
Error: MemoryError
Solution: Reduce chunk size, process one document at a time
```

**3. FAISS Index Not Loading**
```
Error: Cannot read index
Solution: Delete data/faiss_index/* and re-ingest documents
```

**4. Image Extraction Fails**
```
Error: Failed to extract image
Solution: Check PDF is not protected, try different PDF renderer
```

---

## 📝 TODOs & Future Enhancements

- [ ] **Frontend**: Next.js chat interface (DEFERRED)
- [ ] **Re-ranking**: Implement cross-encoder re-ranking
- [ ] **Caching**: Redis cache for frequent queries
- [ ] **Batch Upload**: Process multiple PDFs simultaneously
- [ ] **PDF Protection**: Handle password-protected PDFs
- [ ] **More Extractors**: MEP equipment, material schedules
- [ ] **User Authentication**: JWT-based auth
- [ ] **Vector DB Cloud**: Option for Pinecone/Weaviate

---

## 🔒 Security Notes

- **API Keys**: Never commit `.env` file to git
- **File Upload**: Validates PDF format and file size
- **CORS**: Configure `ALLOWED_ORIGINS` for production
- **Rate Limiting**: Consider adding rate limits for production

---

## 📄 License

This is a demonstration project for construction document management.

---

## 🙏 Acknowledgments

- **Gemini 1.5 Pro**: Google's multimodal LLM
- **FAISS**: Facebook AI Similarity Search
- **FastAPI**: Modern Python web framework
- **PyMuPDF**: PDF processing library

---

## 📧 Contact & Support

For issues or questions:
- Check the `/docs` endpoint for API documentation
- Review logs in console output
- Ensure Gemini API key is valid and has credits

---

**Built with ❤️ for Construction Project Management**
