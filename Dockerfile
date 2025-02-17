FROM ubuntu:latest
RUN apt-get update -y
RUN apt-get install -y python3-pip python-dev-is-python3 build-essential
WORKDIR /usr/src/app
COPY . .
RUN pip install --break-system-packages -r requirements.txt
#RUN apt-get update && apt-get install libgl1 -y
ENTRYPOINT ["python", "./MotionControlApp.py"]