## Quick Setup Script for Project Brain Backend

This script helps you set up the backend environment quickly.

### 1. Copy the .env file
```powershell
cd backend
copy .env.example .env
```

### 2. Edit the .env file
Open `.env` in a text editor and add your Gemini API key:
```
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 3. Install dependencies
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Run the server
```powershell
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Test with sample PDFs (in another terminal)
```powershell
cd backend
venv\Scripts\activate
python tests\test_ingestion.py
```

### 6. Access the API
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health
- Main API: http://localhost:8000

### Troubleshooting
- If you get "GOOGLE_API_KEY not found": Edit the .env file and add your key
- If dependencies fail: Make sure Python 3.11+ is installed
- If test_ingestion.py fails: Make sure the server is running first
