FROM python:3.9
MAINTAINER Aleksei_Rodionov
LABEL version="1.0"
WORKDIR /usr/src/app
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install libgl1 -y
COPY . .
ENTRYPOINT ["python", "./test.py"]