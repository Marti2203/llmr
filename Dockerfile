FROM ubuntu:22.04
LABEL maintainer="Martin Mirchev <mirchevmartin2203@gmail.com>"

# install experiment dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y git python3 python3-pip

WORKDIR /

RUN \
  apt-get update -y && \
  apt-get install software-properties-common -y && \
  apt-get update -y && \
  DEBIAN_FRONTEND=noninteractive \
  apt-get install -y openjdk-11-jdk \
                git \
                build-essential \
				subversion \
				perl \
				curl \
                gcc \
                gcovr \
                maven \
                ant \
				unzip \
				make 
# Java version
ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk-amd64

# Timezone
ENV TZ=Asia/Singapore
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Prepare Fault localization helpers

RUN git clone https://github.com/ASSERT-KTH/flacoco.git

WORKDIR /flacoco
RUN mvn install -DskipTests

# Prepare tool
WORKDIR /
RUN mkdir tool

COPY requirements.txt tool/

RUN pip install -r /tool/requirements.txt

COPY repair.py tool/


