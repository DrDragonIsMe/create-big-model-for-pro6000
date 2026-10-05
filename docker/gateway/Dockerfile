FROM python:3.12-slim
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    fastapi uvicorn httpx docker
COPY scripts/gateway.py /app/gateway.py
WORKDIR /app
CMD ["python3", "gateway.py"]
