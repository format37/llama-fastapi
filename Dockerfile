# Use an official PyTorch runtime as a parent image
# FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set the working directory in the container to /app
WORKDIR /app

# Copy the requirements.txt into the container at /app
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# pip install inference-gpu
# RUN pip install --no-cache-dir inference-gpu

# Copy the current directory contents into the container at /app
COPY . /app/

# Make port 80 available to the world outside this container
EXPOSE 8091

# Run FastAPI application
# 
# CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8091"]
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8091"]
