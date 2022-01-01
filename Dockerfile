# https://github.com/facebookresearch/detectron2/blob/main/docker/Dockerfile

FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
	python3-opencv ca-certificates python3-dev git wget sudo ninja-build
RUN ln -sv /usr/bin/python3 /usr/bin/python

ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser
WORKDIR /home/appuser

ENV PATH="/home/appuser/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py --user && \
	rm get-pip.py


# Подберите тут https://pytorch.org/get-started/locally/
RUN pip3 install --user torch torchvision torchaudio

RUN pip install --user 'git+https://github.com/facebookresearch/fvcore'
RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
RUN pip install --user -e detectron2_repo
COPY requirements_loc.txt /home/appuser/detectron2_repo
RUN pip install --user -r /home/appuser/detectron2_repo/requirements_loc.txt
RUN pip install --user 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

ENV FVCORE_CACHE="/tmp"
WORKDIR /home/appuser/detectron2_repo
ENV PILLOW_VERSION=7.0.0

COPY . /home/appuser/detectron2_repo


ENV PORT 8080
EXPOSE 8080
CMD ["python3", "app_local.py"] 

