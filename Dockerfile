FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for compiling specific packages if needed
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code
COPY . .

# Create the data directory in case it doesn't exist
RUN mkdir -p data

EXPOSE 8000

# Start the FastAPI server using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]