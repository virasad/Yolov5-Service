FROM ultralytics/yolov5:latest
RUN pip install Flask
RUN pip install preimutils

WORKDIR /code/
COPY . /code/

CMD ['python', 'main-train.py']