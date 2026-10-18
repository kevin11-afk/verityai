# VerityAI product: FastAPI backend + built React app, one container, port 7860.
# NOTE: written but not built or run in this project (Docker was not available), so treat it as untested.

FROM node:22-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
# vite.config.ts writes to ../product/static  ->  /product/static
RUN npm run build

FROM python:3.11-slim
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface TOKENIZERS_PARALLELISM=false \
    VERITYAI_PERSISTENCE=ephemeral
WORKDIR /home/user/app

COPY --chown=user requirements.txt requirements-api.txt ./
RUN pip install --no-cache-dir torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-api.txt

COPY --chown=user . .
COPY --from=web --chown=user /product/static ./product/static

EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
