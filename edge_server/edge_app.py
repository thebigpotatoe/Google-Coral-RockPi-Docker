# Inbuilt modules
import time, collections, os, subprocess, re, operator, json, base64, logging, sys, base64

# Add logging to guicorn logs
gunicorn_logger = logging.getLogger('gunicorn.error')
app_logger = logging.getLogger('edge_server')
app_logger.handlers = gunicorn_logger.handlers
app_logger.setLevel(gunicorn_logger.level)

# Server modules
from flask import Flask, flash, request, redirect, url_for, Response
from werkzeug.utils import secure_filename

# Image modules
import numpy as np
from PIL import Image
from io import BytesIO

# TF Lite runtime module
from .edge_ai import edge_coco_ssd

# Globals from environment
model_path = os.environ.get('model_path', 'data/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite')
labels_path = os.environ.get('labels_path', 'data/coco_labels.txt')
threshold = float(os.environ.get('threshold', '0.6'))
top_k = int(os.environ.get('top_k', '5'))
return_image = bool(os.environ.get('return_image', 'False') == 'True')
restrict_cores = bool(os.environ.get('restrict_cores', 'True') == 'True')
debug_image_path = os.environ.get('debug_image', None)
debug_form_path = os.environ.get('debug_form_path', None)

# For rock pi run on cores 5 and 6 which are A72's speeds things up a little
if restrict_cores: 
    app_logger.info("Restricting core operations to A72 cores")
    os.system("taskset -p -c 4-5 %d" % os.getpid())

# Load the interpreter
tpu_instance = edge_coco_ssd()
tpu_instance.load_coco_ssd_model(model_path, labels_path)

# Initialize the Flask application
app = Flask(__name__)

# Response for root path
@app.route('/')
def deny_root():
    return Response(status=404)

# Response for favicon path
@app.route('/favicon')
def deny_favicon():
    return Response(status=404)

# Route to analyse image and return JSON
@app.route('/analyse', methods=['POST'])
def analyse_image():
    try:
        # Convert string of image data to uint8
        img = Image.open(BytesIO(request.data))

        # Analyse image to find objects
        analysis = tpu_instance.coco_ssd_analyse(img, **{"threshold": threshold, "top_k" : top_k})
        if debug_image_path and os.path.exists(debug_image_path): analysis['labelled_image'].save(debug_image_path + '/debug_frame.jpeg', format='jpeg')

        # Convert the labelled image to buffer
        if return_image:
            img_buffer = BytesIO()
            analysis['labelled_image'].save(img_buffer, format='png')
            analysis['labelled_image'] = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        else: 
            del analysis['labelled_image']

        # Convert the python object to json
        json_analysis = json.dumps(analysis)

        # Return response
        return Response(response=json_analysis, status=200, mimetype="application/json")
    except Exception as e:
        app_logger.exception(e)
        return Response(status=500)

# Route to test and debug the algorithim is working
if debug_form_path:
    # Repalce illegal characters
    debug_form_path.replace(' ', '-')

    # Logging
    app_logger.info("Using debug path of {}".format(debug_form_path))

    # Create route
    @app.route(debug_form_path, methods=['GET', 'POST'])
    def debug_image():
        if request.method == 'POST':
            # check if the post request has the file part
            if 'file' not in request.files:
                flash('No file part')
                return redirect(request.url)
            file = request.files['file']

            # if user does not select file, browser also submit an empty part without filename
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)

            # If file is finally valid use it
            if file:
                try:
                    # Logging
                    app_logger.info("File uploaded successfully")

                    # Open the file sent via post
                    img = Image.open(file)

                    # Analyse image to find objects
                    analysis = tpu_instance.coco_ssd_analyse(img, **{"threshold": threshold, "top_k" : top_k})
                    if debug_image_path and os.path.exists(debug_image_path): analysis['labelled_image'].save(debug_image_path + '/debug_frame.jpeg', format='jpeg')

                    # Convert the labelled image to buffer
                    img_buffer = BytesIO()
                    analysis['labelled_image'].save(img_buffer, format='png')
                    response_img = img_buffer.getvalue()

                    # Return response
                    return Response(response=response_img, status=200, mimetype="image/jpeg")
                except Exception as e:
                    app_logger.exception(e)
                    return Response(status=500)
        else:
            return '''
            <!doctype html>
            <title>Google Coral Test Form</title>
            <h1>Upload new Image for analysis</h1>
            <form method=post enctype=multipart/form-data>
                <input type=file name=file accept="image/*">
                <input type=submit value=Upload>
            </form>
            '''

# Logging 
app_logger.info('Successfully initialised Flask application')