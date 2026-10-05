# ComfyUI + WanVideoWrapper 运行镜像（Blackwell 兼容：CUDA 13 + 新版 PyTorch）
ARG PYTORCH_IMAGE=pytorch/pytorch:2.9.1-cuda13.0-cudnn9-runtime
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
FROM ${PYTORCH_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      git ffmpeg libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 build-essential \
    && rm -rf /var/lib/apt/lists/*

# GitHub 克隆带国内镜像兜底（ghfast.top）
RUN (git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git /opt/ComfyUI \
      || git clone --depth 1 https://ghfast.top/https://github.com/comfyanonymous/ComfyUI.git /opt/ComfyUI) \
    && pip install -r /opt/ComfyUI/requirements.txt

# 插件：WanVideoWrapper（Wan2.1/2.2 + InfiniteTalk + S2V 节点）与 GGUF 支持
RUN cd /opt/ComfyUI/custom_nodes \
    && (git clone --depth 1 https://github.com/kijai/ComfyUI-WanVideoWrapper.git \
        || git clone --depth 1 https://ghfast.top/https://github.com/kijai/ComfyUI-WanVideoWrapper.git) \
    && (git clone --depth 1 https://github.com/city96/ComfyUI-GGUF.git \
        || git clone --depth 1 https://ghfast.top/https://github.com/city96/ComfyUI-GGUF.git) \
    && (git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git \
        || git clone --depth 1 https://ghfast.top/https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git) \
    && pip install -r ComfyUI-WanVideoWrapper/requirements.txt \
    && pip install -r ComfyUI-GGUF/requirements.txt \
    && pip install -r ComfyUI-VideoHelperSuite/requirements.txt

# sageattention >=2.2 才完整支持 Blackwell(sm_120)；装不上则运行时回退 PyTorch SDPA
RUN (pip install "sageattention>=2.2.0" \
      || pip install --index-url https://mirrors.aliyun.com/pypi/simple/ "sageattention>=2.2.0" \
      || pip install sageattention \
      || echo "WARN: sageattention 不可用，使用 SDPA 回退") \
    && pip install opencv-python-headless

WORKDIR /opt/ComfyUI
EXPOSE 8188
CMD ["python3", "main.py", "--listen", "0.0.0.0", "--port", "8188"]
