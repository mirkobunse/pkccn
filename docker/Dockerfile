FROM ubuntu:18.04

# build-time variables to be set by the Makefile
ARG group
ARG gid
ARG user
ARG uid

RUN groupadd --gid $gid $group && \
    useradd --create-home --shell /bin/bash --gid $gid --uid $uid $user && \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        less \
        libfreetype6-dev \
        libjpeg-dev \
        libzmq3-dev \
        nano \
        pkg-config \
        python \
        python3 \
        python3-dev \
        python-pip \
        python3-pip \
        python3-setuptools \
        rsync \
        software-properties-common \
        unzip \
        wget \
        zlib1g-dev \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# tini is a process subreaper for Jupyter which prevents kernel crashes
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]

USER $user

CMD ["/bin/bash"]
