# 1. Base Image
FROM python:3.11-slim

# 2. Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Set working directory
WORKDIR /app

# 4. Install system dependencies for OpenCV and other libraries
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy the project files
COPY . .

# 6. Install requirements
RUN pip install --no-cache-dir -r requirements.txt

# 7. Install the package in editable mode
# Already handled by requirements.txt having `-e .`, but keep it just in case
RUN pip install -e .

# 8. Expose the port for Streamlit
EXPOSE 8501

# 9. Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# 10. Run the application
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
