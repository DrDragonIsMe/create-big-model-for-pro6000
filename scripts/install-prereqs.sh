#!/usr/bin/env bash
# 系统级依赖安装：Docker Engine + NVIDIA Container Toolkit
# 仅由 start.sh 或人工在需要时以 root 运行
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

echo "==> 安装基础工具"
apt-get update
apt-get install -y ca-certificates curl gnupg

echo "==> 添加 Docker 官方 apt 源"
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
CODENAME="$(. /etc/os-release && echo "$VERSION_CODENAME")"
# 新版 Ubuntu 发行版 Docker 源可能滞后，失败时回退 noble (24.04)
for TRY_CODENAME in "$CODENAME" noble; do
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $TRY_CODENAME stable" > /etc/apt/sources.list.d/docker.list
  if apt-get update -o Dir::Etc::sourcelist="sources.list.d/docker.list" -o Dir::Etc::sourceparts="-" -o APT::Get::List-Cleanup="0" 2>/dev/null | grep -q "Packages"; then
    echo "==> Docker 源使用 $TRY_CODENAME"
    break
  fi
  echo "==> Docker 源 $TRY_CODENAME 不可用，回退"
done

apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

echo "==> 添加 NVIDIA Container Toolkit 源"
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update
apt-get install -y nvidia-container-toolkit

echo "==> 配置镜像加速（国内网络环境）"
mkdir -p /etc/docker
cat > /etc/docker/daemon.json <<'EOF'
{
  "registry-mirrors": [
    "https://docker.m.daocloud.io",
    "https://docker.1ms.run"
  ]
}
EOF

# nvidia-ctk 会把 nvidia runtime 合并进既有 daemon.json；顺序不能反，
# 否则上面的整体覆盖会把 runtime 配置抹掉，--gpus 失效
echo "==> 配置 Docker 运行时使用 nvidia"
nvidia-ctk runtime configure --runtime=docker

systemctl restart docker

echo "==> 将当前用户加入 docker 组"
if [[ -n "${SUDO_USER:-}" ]]; then
  usermod -aG docker "$SUDO_USER" || true
else
  echo "提示：未识别调用用户（非 sudo 运行？），请手动执行: usermod -aG docker <用户名>"
fi

echo "==> 防火墙放行服务端口（若 ufw 处于启用状态）"
if command -v ufw >/dev/null 2>&1 && ufw status 2>/dev/null | grep -qE "Status: active|状态：.*活"; then
  ufw allow 8000,8188,3000/tcp comment "AI services" || true
  echo "ufw 已放行 8000/8188/3000"
else
  echo "ufw 未启用，无需放行（服务已绑定 0.0.0.0）"
fi

echo "==> 验证"
docker version
docker run --rm hello-world
docker run --rm --gpus all nvidia/cuda:13.0.2-base-ubuntu24.04 nvidia-smi -L

echo "==> 全部完成"
