@echo off
echo Starting KnowledgeBase RAG...
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Start the server in background
echo Starting server on http://localhost:8000
start /B python app.py

REM Wait a moment for server to start
timeout /t 2 /nobreak > nul

REM Open browser
start http://localhost:8000

echo.
echo Server is running! Press Ctrl+C to stop.
echo.
