# llama.cpp server 的兜底镜像：预构建拉取失败时本地编译（CUDA 13, Blackwell）
# 源码由宿主机浅克隆到本目录（容器内 clone GitHub 易 TLS 中断）：
#   git clone --depth 1 https://github.com/ggml-org/llama.cpp.git llama.cpp
#   需要 PR 时：git -C llama.cpp fetch --depth 1 origin pull/<PR>/head:pr && git -C llama.cpp checkout pr
FROM nvidia/cuda:13.0.2-devel-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
      git cmake build-essential libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY llama.cpp /src/llama.cpp
RUN cmake -S /src/llama.cpp -B /src/llama.cpp/build \
       -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF \
    && cmake --build /src/llama.cpp/build --config Release -j"$(nproc)" --target llama-server \
    && cp /src/llama.cpp/build/bin/llama-server /usr/local/bin/ \
    && rm -rf /src/llama.cpp

WORKDIR /
ENTRYPOINT ["llama-server"]
