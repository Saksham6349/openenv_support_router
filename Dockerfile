FROM python:3.10-slim

WORKDIR /app

# Copy definitions and requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose standard HF Space port
EXPOSE 7860

# Command to run FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
