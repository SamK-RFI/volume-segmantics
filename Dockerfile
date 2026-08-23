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

RUN ln -s /usr/bin/python3 /usr/local/bin/python

RUN git clone https://github.com/rosalindfranklininstitute/volume-segmantics.git /opt/volume-segmantics
WORKDIR /opt/volume-segmantics

RUN pip install --no-cache-dir poetry

# Install volume-segmantics in system-wide python, fix opencv version
RUN poetry config virtualenvs.create false \
    && poetry install \
    && pip uninstall -y opencv-python \
    && pip install --no-cache-dir "opencv-python-headless==4.11.0.86"

WORKDIR /root

CMD ["/bin/bash"]
