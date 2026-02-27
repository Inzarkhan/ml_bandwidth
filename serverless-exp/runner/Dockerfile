FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 python3-pip time \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install psutil
RUN pip3 install numpy

WORKDIR /app
COPY serverless-exp/runner/harness_linux.py /app/harness_linux.py
COPY serverless-exp/workloads /app/workloads

ENTRYPOINT ["python3", "/app/harness_linux.py"]
