FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 python3-pip time \
    build-essential gfortran m4 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install psutil numpy jinja2 pillow igraph squiggle==0.3.1

WORKDIR /app
COPY serverless-exp/runner/harness_linux.py /app/harness_linux.py
COPY serverless-exp/workloads /app/workloads

ENTRYPOINT ["python3", "/app/harness_linux.py"]
