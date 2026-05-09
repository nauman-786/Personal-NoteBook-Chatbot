@echo off
title Notebook AI
echo.
echo  Starting Notebook AI...
echo  Open http://localhost:8000 in your browser
echo.

pip install fastapi uvicorn python-multipart --quiet

start "" http://localhost:8000
uvicorn api:app --host 0.0.0.0 --port 8000 --reload

pause