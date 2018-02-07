FROM continuumio/miniconda3:4.3.14
MAINTAINER Amit Kushwaha <kushwahamit2016@gmail.com>

ADD . /usr/src/app

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

RUN conda update --yes pip

RUN set -eux \
    && apt-get update

RUN apt-get install -y make gcc g++ libsnappy-dev

COPY requirements.txt /tmp/requirements.txt
RUN  pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm -fv /tmp/requirements.txt \
    && conda clean --all --yes

COPY setup.sh /tmp/setup.sh

RUN bash /tmp/setup.sh 3 \
    && rm -fv /tmp/setup.sh
