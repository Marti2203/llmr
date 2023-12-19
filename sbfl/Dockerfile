FROM ubuntu:20.04
LABEL maintainer="Martin Mirchev <mmirchev@comp.nus.edu.sg>"

RUN apt-get update && apt-get upgrade -y && apt-get autoremove -y

# install experiment dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends  \
    automake \
    autopoint \
    ca-certificates \
    cmake \
    curl \
    gcc \
    libpython3-dev \
    libtool \
    nasm \
    nano \
    openssl \
    pkg-config \
    psmisc \
    python3 \
    python3-pip \
    libc6-dev-i386 \
    gcc-multilib \
    g++-multilib \
    clang \
    g++ \
    gcc \
    wget \
    curl \
    unzip \
    make \
    libasan4 \
    gdb \
    build-essential \
    xxd

RUN mkdir /sbfl

COPY . /sbfl

WORKDIR /sbfl

RUN bash ./build.sh

RUN pip3 install polars numpy pandas matplotlib