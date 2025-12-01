# Constructure AI - Project Brain 🏗️

A powerful **multimodal RAG (Retrieval-Augmented Generation)** system for construction project document management. This system intelligently processes construction PDFs (including drawings, specifications, and schedules) and provides:

- 💬 **Natural Language Q&A** with accurate citations
- 🔍 **Multimodal Search** across text and images (construction drawings)  
- 📊 **Structured Extraction** of door schedules, room schedules, and more
- 🤖 **Gemini 1.5 Pro** for vision understanding of construction drawings
- ⚡ **FAISS Vector Search** for lightning-fast retrieval

---

## 🎯 Project Status

### ✅ Backend: **COMPLETE & READY**
The FastAPI backend is fully implemented with:
- Multimodal document ingestion (text + images)
- Advanced RAG pipeline with hybrid search
- Structured extraction (doors, rooms)
- Evaluation framework
- Docker support

### ⏸️ Frontend: **DEFERRED**
Frontend development is planned for a future phase. The backend is fully functional and can be used via:
- REST API (documented at `/docs`)
- cURL commands
- API clients (Postman, HTTPie, etc.)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Google Gemini API key ([Get one free](https://makersuite.google.com/app/apikey))

### 1. Setup Backend
```bash
cd backend

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
copy .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### 2. Start Server
```bash
# Option 1: Using PowerShell script
.\start.ps1

# Option 2: Manual start
python -m uvicorn app.main:app --reload
```

### 3. Access API
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Root**: http://localhost:8000

### 4. Test with Sample PDFs
```bash
# In another terminal
cd backend
python tests\test_ingestion.py
```

This will:
1. Upload the 2 sample PDFs from `tests/` folder
2. Test RAG queries
3. Verify all functionality

---

## 📁 Repository Structure

```
Constructureai/
├── backend/              ⭐ Main backend application
│   ├── app/
│   │   ├── main.py       # FastAPI entry point
│   │   ├── services/     # RAG, ingestion, extraction
│   │   ├── db/           # FAISS vector store
│   │   ├── api/routes/   # API endpoints
│   │   └── ...
│   ├── data/             # Document storage
│   ├── evaluation/       # Test queries
│   ├── tests/            # Test scripts
│   ├── README.md         # Full backend documentation
│   ├── SETUP.md          # Quick setup guide
│   ├── start.ps1         # Startup script
│   └── requirements.txt
├── tests/                # Sample PDFs
│   ├── Construction Drawings PDF
│   └── DBA Wages PDF
└── README.md (this file)
```

---

## 🔧 Core Features

### 1. Document Ingestion
Upload construction PDFs via API. The system:
- ✅ Extracts text and images
- ✅ Describes images using Gemini Vision (makes drawings searchable!)
- ✅ Creates smart chunks with page boundaries
- ✅ Generates embeddings and stores in FAISS
- ✅ Tracks metadata (filename, page, type)

### 2. Intelligent Q&A (RAG)
Ask questions in natural language:
```bash
curl -X POST "http://localhost:8000/api/chat/query" \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the fire rating requirements for doors?"}'
```

**Returns**:
- AI-generated answer
- Source citations (file + page)
- Relevant images (if applicable)

### 3. Structured Extraction
Extract structured data from documents:
- **Door Schedules**: Mark, location, dimensions, fire rating, material
- **Room Schedules**: Number, name, area, finishes
- **MEP Equipment**: (extensible)

```bash
curl -X POST "http://localhost:8000/api/extract/door-schedule"
```

Returns clean JSON for integration with other systems.

### 4. Multi-Document Search
Search across all project documents simultaneously with context-aware ranking.

---

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **LLM** | Gemini 1.5 Pro | Multimodal understanding (text + images) |
| **Embeddings** | Gemini Embeddings | 768-dimensional vectors |
| **Vector DB** | FAISS | Fast similarity search |
| **Backend** | FastAPI | Modern async Python API |
| **PDF Processing** | PyMuPDF | Text + image extraction |
| **Deployment** | Docker | Containerization |

---

## 📚 Documentation

### Backend Documentation
See [`backend/README.md`](backend/README.md) for:
- Complete API documentation
- Architecture details
- Deployment guide
- Performance metrics
- Troubleshooting

### Quick Guides
- [`backend/SETUP.md`](backend/SETUP.md) - Quick setup steps
- [`backend/tests/test_ingestion.py`](backend/tests/test_ingestion.py) - Integration tests

---

## 🧪 Sample Data

The `tests/` folder includes 2 sample PDFs:
1. **Construction Drawings** (508-22-105) - 28.6 MB
   - Floor plans, construction details, drawings
2. **DBA Wages Document** - 71 KB
   - Wage specifications

These are automatically used by the test script.

---

## 🌐 API Endpoints

### Documents
- `POST /api/documents/upload` - Upload PDF
- `GET /api/documents/list` - List all documents
- `GET /api/documents/stats` - Vector store stats

### Chat (RAG)
- `POST /api/chat/query` - Ask questions
- `GET /api/chat/conversation/{id}` - Get history
- `DELETE /api/chat/conversation/{id}` - Delete conversation

### Extraction
- `POST /api/extract/door-schedule` - Extract doors
- `POST /api/extract/room-schedule` - Extract rooms

### Evaluation
- `POST /api/evaluate/run` - Run test suite
- `GET /api/evaluate/results` - Get results

### System
- `GET /` - API info
- `GET /health` - Health check
- `GET /docs` - Interactive documentation

---

## 🚢 Deployment

### Docker
```bash
cd backend
docker build -t project-brain .
docker run -p 8000:8000 -e GOOGLE_API_KEY=your_key project-brain
```

### Railway / Render
1. Connect your GitHub repo
2. Set environment variable: `GOOGLE_API_KEY`
3. Deploy!

See [`backend/README.md`](backend/README.md) for detailed deployment instructions.

---

## ✅ Testing

### Automated Test
```bash
cd backend
python tests\test_ingestion.py
```

**Tests**:
- ✅ API health check
- ✅ Document upload (2 PDFs)
- ✅ RAG queries (2 questions)
- ✅ Vector store statistics

### Manual Testing
1. Start server: `cd backend && .\start.ps1`
2. Open browser: http://localhost:8000/docs
3. Try "Upload Document" endpoint
4. Try "Chat Query" endpoint
5. View extracted schedules

---

## 📊 Performance

- **Ingestion**: 1-2 pages/second (text), 2-3 seconds/image (vision)
- **Search**: <100ms (FAISS on 10K vectors)
- **RAG Query**: 2-4 seconds end-to-end
- **Extraction**: 3-5 seconds per schedule

---

## 🎯 Use Cases

### Construction Project Management
- Search across all project documents
- Quickly find specifications, dimensions, materials
- Extract door/room schedules automatically

### Contract Review
- Ask questions about requirements
- Find relevant clauses with citations
- Compare specifications across documents

### Drawing Analysis
- Search construction drawings by content
- Find specific rooms, details, dimensions
- Cross-reference drawings with specs

---

## 🔒 Environment Variables

Required:
- `GOOGLE_API_KEY` - Your Gemini API key

Optional (see `.env.example`):
- `GEMINI_MODEL` - Model version
- `TOP_K_RETRIEVAL` - Search result count
- `CHUNK_SIZE` - Text chunk size
- `ALLOWED_ORIGINS` - CORS origins

---

## 🐛 Troubleshooting

### "Failed to start server"
- ✅ Check virtual environment is activated
- ✅ Verify all dependencies installed: `pip install -r requirements.txt`
- ✅ Check Python version: `python --version` (needs 3.11+)

### "GOOGLE_API_KEY not found"
- ✅ Copy `.env.example` to `.env`
- ✅ Add your API key to `.env`
- ✅ Restart the server

### "Test ingestion fails"
- ✅ Make sure server is running first
- ✅ Check PDFs exist in `tests/` folder
- ✅ Verify API key has credits

See [`backend/README.md`](backend/README.md#troubleshooting) for more help.

---

## 🔜 Future Enhancements

### Planned Features
- [ ] **Frontend**: Next.js chat interface
- [ ] **More Extractors**: MEP equipment, material schedules
- [ ] **Advanced Re-ranking**: Cross-encoder models
- [ ] **User Authentication**: JWT-based auth
- [ ] **Cloud Vector DB**: Option for Pinecone/Weaviate
- [ ] **Batch Upload**: Process multiple files

### Community Contributions Welcome!

---

## 📄 License

This is a demonstration project for construction document management.

---

## 🙏 Acknowledgments

- **Google Gemini** - Multimodal AI
- **FAISS** - Facebook AI Similarity Search
- **FastAPI** - Modern Python web framework
- **PyMuPDF** - PDF processing

---

## 📧 Support

- Check the [Backend README](backend/README.md) for detailed docs
- Review the API documentation at `/docs`
- Run the test script to verify setup

---

**Built with ❤️ for Construction Project Management**

**Backend Status**: ✅ Complete and ready for deployment  
**Frontend Status**: ⏸️ Deferred for future phase