FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY . .

# Ensure submodules are populated even when built from a shallow/non-recursive context (e.g. HF Space)
RUN if [ -d .git ] && [ -f .gitmodules ]; then \
        git submodule update --init --recursive || true; \
    fi && \
    for mod in Dental_Core Dental_002 Dental_003 Dental_004 Dental_008 Dental_009 Dental_010 Dental_011 Dental_012 Dental_013 Dental_014; do \
        if [ ! -f "modules/${mod}/README.md" ]; then \
            rm -rf "modules/${mod}" && \
            git clone --depth 1 "https://github.com/HyunchanAn/${mod}.git" "modules/${mod}"; \
        fi; \
    done

# Install CPU-portable runtime dependencies
RUN pip install --no-cache-dir -r requirements-cpu.txt && \
    sed -i 's/\r$//' /app/docker_entrypoint.sh && \
    chmod +x /app/docker_entrypoint.sh

ENV PORT=7860 \
    SERVER_MODE=api

EXPOSE 7860 8000 8501

ENTRYPOINT ["/app/docker_entrypoint.sh"]
