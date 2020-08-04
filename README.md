# Google-Coral-RockPi-Docker

The motivation behind this image was to produce a local server that can analyse an image for known objects very quickly in applications such as home automation and security. It's not perfect, so any feedback, issues, or pull requests are most welcome :smile:

The main application is built with Flask which runs using gunicorn as the WSGI server. The default port for the application is `1890` but can be changed in the Dockerfile with ease as it will automatically update all valid references to the port.

Two possible endpoints are available on the server depending on the environmental variables the user has set (see below). The primary endpoint is `/analyse` which expects an image to be uploaded in the body of a POST request. The second is the debug endpoint at the users chosen destination such as `/debug`. At this endpoint a very basic webpage is available to upload images using a browser for testing and validation. The server will take any uploaded image and pass it to the Coral board for analysis. The analysis returned by the board is parsed into a JSON object which is then returned. The structure of this response is described below.

## Quick start

These are the easiest commands to run in series to get up and going assuming your hardware is working. Check out the other sections for more information on how to customise this image.

### Assemble your Board

To get started you will need a Rock Pi 4B or similar, a google coral PCIe module, and a decent power supply. For reference see the image below on how to assemble:

![RockPi4bB With Coral](docs/RockPi.jpg)

### Install Docker

Install docker however you would like. This is the most convenient way:

``` bash
curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh
```

### Run the pre-compiled image

Run the pre compiled image in debug mode allowing you to send test images at `http://<Rock-Pi-IP-Address>:1890/debug`

``` Docker
docker run -it --rm --privileged \
    --env return_image=True \
    --env debug_form_path=/debug \
    -p 1890:1890 \
    thebigpotatoe/google-coral-rockpi-docker
```

### Run the image in production mode

When ready, run the image in production mode. This accepts images via a POST request at `http://<Rock-Pi-IP-Address>:1890/analyse`

``` Docker
docker run -d --privileged \
    -p 1890:1890 \
    thebigpotatoe/google-coral-rockpi-docker
```

## Usage

### Environmental Variables

The container utilises several environmental variables to customise how the container runs. Each is optional and will default to a specific value aimed at running in production. The complete list of the variables available for changing is:

ENV Variable | Description | Input Type | Default Value
------------ | ----------- | ---------- | -------------
model_path | The path to the model within the container | String | data/mobilenet _ssd_v2_coco_ quant_postprocess_ edgetpu.tflite
labels_path | The path to the labels file within the container | String | data/coco_labels.txt
threshold | The threshold value to define valid objects | Float | 0.6
top_k | The max number of objects to list per image | Int | 5
return_image | If true return the labelled image in the response | 'True' or 'False' | 'False'
restrict_cores | For the Rock Pi, restrict the script to running on the A72 cores. This helps to speed up the response. | 'True' or 'False' | 'False'
debug_image_path | An optional file path to save the last analysed image at. This could be useful if running with a volume. | String | None
debug_form_path | An optional endpoint of the server to upload test images using a browser | String |  None

### Privileged Mode

The only downside to this image is it must be run in privileged mode to allow the container access to the hardware. When running make sure to use the  `--privileged` flag with `docker run`

### Message Structure

Upon successful analysis of an image, the following message structure will be returned:

``` javascript
{
    "objects": [                                // A list of found objects
        {
            "id": 0,                                // The COCO class ID number
            "id_str": "person",                     // The COCO class descriptor for labels.txt
            "score": 0.87890625,                    // The confidence score of the current object
            "bbox": {                               // The bounding box dimensions
                "xmin": 0.0006975233554840088,
                "ymin": 0.25995802879333496,
                "xmax": 0.9930738210678101,
                "ymax": 1.0
            }
        },
        ...
    ],
    "inference_time": 53.94124984741211,            // The inference time spent on the Coral TPU
    "labelled_image": "iVBORw0KGgoAAAANSU..."       // (Optional) A Base64 encoded string of the analysed and labelled image
}
```

## Build from source

The image can also be compiled from the Dockerfile in this repo. The base image is python:3.6-slim-buster for simplicity and size. Each of the modules required to run the Coral board are then installed along with the appropriate Python modules. Here you can also change the base port number if `1890` does not suit.

Start with cloning the repo into a known location:

``` bash
git clone https://github.com/thebigpotatoe/Google-Coral-RockPi-Docker.git && cd Google-Coral-RockPi-Docker
```

Then run the docker build command from the base folder of the repo

``` bash
docker build . < Dockerfile --tag google-coral-rockpi-docker-local
```

## Performance

Below are the results of a basic test running over a local network. The results tabularised are for a quick reference on what to expect:

Times (ms) | Return Image Off | Return Image On
---------- | ---------------- | ---------------
Restrict Cores On | 110ms | 320ms
Restrict Cores Off | 175ms | 470ms
