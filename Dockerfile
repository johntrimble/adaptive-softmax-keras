FROM tensorflow/tensorflow:1.7.1-gpu-py3

ADD requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

RUN mkdir /code

WORKDIR /code
