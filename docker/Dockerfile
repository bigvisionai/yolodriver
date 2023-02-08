# Builds bigvisionai/yolovdriver:latest image on DockerHub https://hub.docker.com/r/bigvisionai/yolovdriver

# Start FROM NVIDIA PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM nvcr.io/nvidia/pytorch:22.11-py3
RUN rm -rf /opt/pytorch  # remove 1.2GB dir

# Downloads to user config dir
ADD https://ultralytics.com/assets/Arial.ttf https://ultralytics.com/assets/Arial.Unicode.ttf /root/.config/Ultralytics/

# Install linux packages
RUN apt update && apt install --no-install-recommends -y zip htop screen libgl1-mesa-glx

# Install pip packages
RUN python -m pip install --upgrade pip wheel
RUN pip uninstall -y Pillow torchtext  # torch torchvision
COPY requirements.txt .
RUN pip install --no-cache -r requirements.txt ultralytics albumentations comet gsutil notebook Pillow>=9.1.0 \
    'opencv-python<4.6.0.66' \
    --extra-index-url https://download.pytorch.org/whl/cu116

# Create working directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# Copy contents
# COPY . /usr/src/app  (issues as not a .git directory)
RUN git clone https://github.com/bigvisionai/yolodriver.git /usr/src/app

# Set environment variables
ENV OMP_NUM_THREADS=8


# Usage Examples -------------------------------------------------------------------------------------------------------

# Build and Push
# t=opencvcourses/yolodriver:0.1 && sudo docker build -f docker/Dockerfile -t $t . && sudo docker push $t

# Pull and Run
# t=opencvcourses/yolodriver:0.1 && sudo docker pull $t && sudo docker run -it --ipc=host --gpus all $t

# Pull and Run with local directory access
# t=opencvcourses/yolovdriver:0.1 && sudo docker pull $t && sudo docker run -it --ipc=host --gpus all -v "$(pwd)"/datasets:/usr/src/datasets $t

# Kill all
# sudo docker kill $(sudo docker ps -q)

# Kill all image-based
# sudo docker kill $(sudo docker ps -qa --filter ancestor=opencvcourses/yolovdriver:0.1)


# Clean up
# docker system prune -a --volumes

# Update Ubuntu drivers
# https://www.maketecheasier.com/install-nvidia-drivers-ubuntu/

# DDP test
# python -m torch.distributed.run --nproc_per_node 2 --master_port 1 train.py --epochs 3

# GCP VM from Image
# docker.io/opencvcourses/yolovdriver:0.1

# -e TZ=IST -p 8081:8081