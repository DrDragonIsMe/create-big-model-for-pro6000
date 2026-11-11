# llama.cpp server 的兜底镜像：预构建拉取失败时本地编译（CUDA 13, Blackwell）
# 源码由宿主机浅克隆到本目录（容器内 clone GitHub 易 TLS 中断）：
#   git clone --depth 1 https://github.com/ggml-org/llama.cpp.git llama.cpp
#   需要 PR 时：git -C llama.cpp fetch --depth 1 origin pull/<PR>/head:pr && git -C llama.cpp checkout pr
# CMAKE_CUDA_ARCHITECTURES 为 ARG：换 GPU 架构（如 Hopper sm_90a）时
#   docker build --build-arg CMAKE_CUDA_ARCHITECTURES=90a-real -t ai-llamacpp:master ./docker/llamacpp
# 构建元数据（编译机 CPU 型号/指令集 + CUDA arch）写入 /etc/ai-llamacpp-build.env，
# 网关启动后端前据其判断镜像与本机是否匹配（换 CPU/GPU 后提示重编，而不是运行时才崩）。
FROM nvidia/cuda:13.0.2-devel-ubuntu24.04

ARG CMAKE_CUDA_ARCHITECTURES=120a-real
LABEL ai.llamacpp.cuda-arch="${CMAKE_CUDA_ARCHITECTURES}"

RUN apt-get update && apt-get install -y --no-install-recommends \
      git cmake build-essential libcurl4-openssl-dev curl ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN printf 'AI_LLAMACPP_CUDA_ARCH=%s\nAI_LLAMACPP_CPU_MODEL=%s\nAI_LLAMACPP_CPU_FLAGS=%s\n' \
      "${CMAKE_CUDA_ARCHITECTURES}" \
      "$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //;s/[[:space:]]\+/ /g')" \
      "$(grep -m1 '^flags' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //;s/[[:space:]]\+/ /g')" \
    > /etc/ai-llamacpp-build.env \
    && cat /etc/ai-llamacpp-build.env

COPY llama.cpp /src/llama.cpp
RUN cmake -S /src/llama.cpp -B /src/llama.cpp/build \
       -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES} \
       -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF \
    && cmake --build /src/llama.cpp/build --config Release -j"$(nproc)" --target llama-server \
    && cp /src/llama.cpp/build/bin/llama-server /usr/local/bin/ \
    && rm -rf /src/llama.cpp

WORKDIR /
ENTRYPOINT ["llama-server"]
