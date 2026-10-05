# llama.cpp server 的兜底镜像：预构建拉取失败时本地编译（CUDA 13, Blackwell）
FROM nvidia/cuda:13.0.2-devel-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
      git cmake build-essential libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp.git /src/llama.cpp \
    && cmake -S /src/llama.cpp -B /src/llama.cpp/build \
       -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF \
    && cmake --build /src/llama.cpp/build --config Release -j"$(nproc)" --target llama-server \
    && cp /src/llama.cpp/build/bin/llama-server /usr/local/bin/ \
    && rm -rf /src/llama.cpp

WORKDIR /
ENTRYPOINT ["llama-server"]
