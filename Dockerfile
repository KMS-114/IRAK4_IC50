FROM python:3.11

WORKDIR /workspace/src

COPY ./requirements.txt /workspace/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /workspace/requirements.txt

COPY ./src /workspace/src

COPY ./data /workspace/data