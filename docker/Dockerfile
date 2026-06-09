FROM supervisely/base-py-sdk:6.73.418
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update

# For PyTorch 1.10.0
RUN pip3 install --no-cache-dir setuptools==59.5.0
# ---------------------------------------------- #

RUN pip3 install torch==1.10.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip3 install torchvision==0.11.2+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip3 install ultralytics==8.0.112

RUN python3 -m pip install supervisely==6.73.486

LABEL python_sdk_version="6.73.486"