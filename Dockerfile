FROM ubuntu:22.04
LABEL maintainer="Martin Mirchev <mirchevmartin2203@gmail.com>"

# install experiment dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y git python3 python3-pip
RUN mkdir tool
COPY requirements.txt tool/
RUN pip install -r /tool/requirements.txt
COPY repair.py tool/
