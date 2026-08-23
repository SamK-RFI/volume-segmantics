FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget \
    git \
    python3 \
    python3-pip \
    libglib2.0-0 \
    locales \
    && locale-gen en_GB.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# locale settigns for perl
ENV LANG=en_GB.UTF-8
ENV LANGUAGE=en_GB:en
ENV LC_ALL=en_GB.UTF-8

# Add NVIDIA CUDA repository and install runtime libraries only
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && rm cuda-keyring_1.1-1_all.deb \
    && apt-get update && apt-get install -y \
    cuda-libraries-12-6 \
    libcudnn9-cuda-12 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/local/bin/python

RUN git clone https://github.com/rosalindfranklininstitute/volume-segmantics.git /opt/volume-segmantics
WORKDIR /opt/volume-segmantics

# Install torch cuda 12.6 wheel
RUN pip install --no-cache-dir poetry
RUN pip install --no-cache-dir \
    torch==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu126

# Install volume-segmantics in system-wide python, fix opencv version
RUN poetry config virtualenvs.create false \
    && poetry install \
    && pip uninstall -y opencv-python \
    && pip install --no-cache-dir "opencv-python-headless==4.11.0.86"

WORKDIR /root

CMD ["/bin/bash"]
