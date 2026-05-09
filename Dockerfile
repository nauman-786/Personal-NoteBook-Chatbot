FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Hugging Face Spaces specific port
ENV PORT=7860
EXPOSE 7860

# Force uvicorn to run your FastAPI app (api.py)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]