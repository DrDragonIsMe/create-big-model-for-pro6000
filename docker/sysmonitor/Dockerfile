# sysmonitor Web 看板常驻镜像：只装运行依赖，代码从宿主 bind-mount（改代码只需 restart，不用 rebuild）
FROM python:3.12-slim
RUN pip install --no-cache-dir psutil
WORKDIR /opt/sysmonitor
EXPOSE 8088
CMD ["python3", "-u", "sysmon_web.py", "--host", "0.0.0.0", "--port", "8088"]
