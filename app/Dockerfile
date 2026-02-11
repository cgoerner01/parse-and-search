FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*
# Set working directory
WORKDIR /app
# Copy requirements file
#COPY app/requirements.txt .
COPY requirements.txt .
# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
# Copy application code
#COPY app/ .
COPY . .
# Expose the port Streamlit runs on
EXPOSE 8501
# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
# Command to run the application
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501"]