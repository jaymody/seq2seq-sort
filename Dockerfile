FROM nvidia/cuda:11.0-runtime-ubuntu20.04

# because NVIDIA: https://github.com/NVIDIA/nvidia-docker/issues/1631#issuecomment-1112828208
RUN rm -f /etc/apt/sources.list.d/cuda.list && \
    rm -f /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-key del 7fa2af80 && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# make sure we don't get prompted for input during apt install
ARG DEBIAN_FRONTEND=noninteractive

# don't cache pip packages, reduces the docker image size
ARG PIP_NO_CACHE_DIR=1

# see https://stackoverflow.com/questions/59812009/what-is-the-use-of-pythonunbuffered-in-docker-file
ENV PYTHONUNBUFFERED=1

RUN apt update -y && \
    apt upgrade -y && \
    apt install -y bash \
        build-essential \
        git \
        curl \
        ca-certificates \
        libjpeg-dev \
        libpng-dev \
        python3.9 \
        python3.9-venv && \
    rm -rf /var/lib/apt/lists

WORKDIR /app

# copying only poetry files before running poetry install makes sure docker
# reruns the build ONLY if poetry.lock/pyproject.toml change (otherwise,
# docker would need to rebuild this layer on ANY file change, which is annoying)
COPY poetry.lock pyproject.toml ./
RUN curl https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && \
    python3.9 /tmp/get-pip.py && \
    python3.9 -m pip install pip poetry && \
    python3.9 -m poetry install --no-dev && \
    poetry run pip install torch==1.11.0+cu113 --no-cache -f https://download.pytorch.org/whl/torch_stable.html

COPY . .

EXPOSE 8000

ENV UVICORN_HOST=0.0.0.0 \
    UVICORN_PORT=8000
CMD [ "/bin/bash", "-c", "python3.9 -m poetry run uvicorn app:app" ]
