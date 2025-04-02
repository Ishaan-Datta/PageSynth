FROM huggingface/transformers-pytorch-gpu:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    build-essential \
    git \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk2.0-dev \
    pkg-config \
    python3 \
    python3-pip \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN pip3 install --upgrade --no-cache-dir --verbose \
    notebook \
    gradio \
    pillow \
    huggingface_hub \
    docling-core \
    accelerate \
    sentencepiece \
    safetensors \
    pix2text

RUN pip3 uninstall onnxruntime -y

RUN pip3 install onnxruntime-gpu

RUN pip3 install flash-attn --no-build-isolation

RUN mkdir -p /home/user/workspace
WORKDIR /home/user/workspace

EXPOSE 8888

ENTRYPOINT [ "/bin/bash" ]