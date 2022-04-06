RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python
RUN apt-get update && apt-get install -y opencv-python-headless
RUN pip install opencv-python-headless
apt-get update && apt-get install libgl1
