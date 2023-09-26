#!/bin/bash

# Run the Docker container and mount a path for the models
# gpus all
sudo docker run -it --rm \
  --gpus all \
  -v ./data:/app/data \
  -p 8091:8091 \
  --name running-llama-fastapi \
  llama-fastapi
