# Use official Python slim image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up non-root user with UID 1000 for Hugging Face Spaces compatibility
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy requirements first
COPY --chown=user requirements.txt .

# Install PyTorch CPU first to keep image lightweight
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY --chown=user . $HOME/app

# Switch to non-root user
USER user

# Expose port 7860 for Hugging Face Spaces
EXPOSE 7860

# Launch uvicorn directly on 0.0.0.0:7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
