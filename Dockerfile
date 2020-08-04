# Based on alpine python image
FROM python:3.6-slim-buster

# Install Curl
RUN apt-get update -y && \
    apt-get install -y gnupg curl

# Get the pci libraries from google
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
    && apt-get update \
    && apt-get -y install libedgetpu1-std

# Install python deps
RUN /usr/local/bin/python -m pip install --upgrade pip
RUN pip install cython pillow numpy gunicorn flask
RUN pip install "https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp36-cp36m-linux_aarch64.whl"

# Copy in python app files
COPY ./data /flask-ai/data
COPY ./edge_server /flask-ai/edge_server

# Add the folder to the python path
ENV PYTHONPATH "${PYTHONPATH}:/flask-ai"

# Add the app env variables
ENV model_path="/flask-ai/data/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite" \
    labels_path="/flask-ai/data/coco_labels.txt" \
    threshold="0.1" \
    top_k="3"

# Add flow control varibles
ENV return_image='True' \
    restrict_cores='True'

# Add bebug variables (optional)
# ENV debug_image_path="/flask-ai" 
# ENV debug_form_path="/debug"

# Start at the python app using gunicorn
ENV SERVER_PORT=1890
EXPOSE $SERVER_PORT
WORKDIR /flask-ai
CMD gunicorn --bind 0.0.0.0:$SERVER_PORT --log-level=info edge_server:app