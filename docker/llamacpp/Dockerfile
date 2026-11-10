# llama.cpp server 的兜底镜像：预构建拉取失败时本地编译（CUDA 13, Blackwell）
# 源码由宿主机浅克隆到本目录（容器内 clone GitHub 易 TLS 中断）：
#   git clone --depth 1 https://github.com/ggml-org/llama.cpp.git llama.cpp
#   需要 PR 时：git -C llama.cpp fetch --depth 1 origin pull/<PR>/head:pr && git -C llama.cpp checkout pr
FROM nvidia/cuda:13.0.2-devel-ubuntu24.04 AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
      git cmake build-essential libcurl4-openssl-dev curl ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY llama.cpp /src/llama.cpp
RUN cmake -S /src/llama.cpp -B /src/llama.cpp/build \
       -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120a-real \
       -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
       -DBUILD_SHARED_LIBS=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF \
       -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /src/llama.cpp/build --config Release -j"$(nproc)" --target llama-server \
    && mkdir /app \
    && cp /src/llama.cpp/build/bin/llama-server /app/ \
    && find /src/llama.cpp/build -name '*.so*' -exec cp -P {} /app/ \;

FROM nvidia/cuda:13.0.2-runtime-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl ffmpeg libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /app /app

WORKDIR /app
ENTRYPOINT ["/app/llama-server"]
