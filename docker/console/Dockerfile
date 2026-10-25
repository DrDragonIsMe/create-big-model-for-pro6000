FROM python:3.12-slim
# console agent：FastAPI + docker CLI（manage.sh/init.sh 要在容器里驱动宿主 docker.sock）。
# docker-cli（不是 docker.io！trixie 把 CLI 拆包了，docker.io 只剩 daemon，装完没 /usr/bin/docker——实测踩过）。
# 源里没有 docker-compose-v2；compose 用宿主上的**静态**插件二进制
# （/usr/libexec/docker/cli-plugins/docker-compose，三台实测 not a dynamic executable），
# 由 compose.yaml 只读挂载进来——CLI 的插件搜索路径本就包含 /usr/libexec/docker/cli-plugins。
# apt/pip 走清华源，与网关/下载器同一网络现实。
RUN sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources \
 && apt-get update \
 && apt-get install -y --no-install-recommends docker-cli ca-certificates \
 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple fastapi uvicorn httpx
COPY console /opt/console
WORKDIR /opt/console
CMD ["python3", "agent.py"]
