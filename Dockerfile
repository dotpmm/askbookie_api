FROM python:3.11-slim

RUN useradd -m -u 1000 user

USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app
RUN mkdir -p /app/cache
ENV HF_HOME=/app/cache

ENV PYTHONUNBUFFERED=1 

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user . .

ENV PYTHONPATH=/app/src

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]