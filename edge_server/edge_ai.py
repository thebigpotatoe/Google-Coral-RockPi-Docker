# OS Modules 
import os, time, re, logging

# Start a logger 
# print("AI {}".format(__name__))
logger = logging.getLogger(__name__)

# Image modules
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# TF Lite runtime module
import tflite_runtime.interpreter as tflite

# coco class
class edge_coco_ssd:
    def __init__(self):
        self._ready = False
        self._model_path = ''
        self._interpreter = None

    # Convenient Image Functions
    def load_image_from_file(self, file_path):
        try:
            return cv2.imread(file_path)
        except Exception:
            raise

    def parse_image_from_string(self, data_str):
        try:
            return np.fromstring(data_str, np.uint8)
        except Exception:
            raise

    # Model Loading Methods
    def load_coco_ssd_model(self, i_model_path, i_labels_path):
        # Try load labels
        try:
            logger.info('Loading labels from {}'.format(i_labels_path))

            p = re.compile(r'\s*(\d+)(.+)')
            with open(i_labels_path, 'r', encoding='utf-8') as f:
                lines = (p.match(line).groups() for line in f.readlines())
                self._labels = {int(num): text.strip() for num, text in lines}
            
            logger.info('Successfully loaded {} labels'.format(len(self._labels)))
        except Exception:
            raise

        # Try load model 
        try:
            logger.info('Loading model from {}'.format(i_model_path))
            
            i_model_path, *device = i_model_path.split('@')
            self._interpreter = tflite.Interpreter(
                model_path=i_model_path,
                experimental_delegates=[
                    tflite.load_delegate('libedgetpu.so.1',
                                        {'device': device[0]} if device else {})
                ])
            self._interpreter.allocate_tensors()

            logger.info('Successfully loaded model')
        except Exception:
            raise
        
        # Set the interpreter as ready
        self._ready = True

    # Inference Methods
    def coco_ssd_analyse(self, input_img, **kwargs):
        # Get the required kwargs
        threshold = kwargs.get('threshold', 0.1)
        top_k = kwargs.get('top_k', 3)
        resample = kwargs.get('resample', Image.NEAREST)
        label_image = kwargs.get('label_image', True)

        # Create data object 
        objects_data = dict()

        # Convert image to PIL image
        # analysis_img = Image.fromarray(input_img)
        analysis_img = input_img

        # Set the input tensor to the image
        self._set_coco_ssd_input_tensor(analysis_img)

        # Forward pass the image
        start_ms = time.time()
        self._interpreter.invoke()
        end_ms = time.time()
        logger.info('COCO SSD forward pass took {:.2f}ms'.format((end_ms - start_ms) * 1000))

        # Create the objects structure
        scores = self._get_coco_ssd_output_tensor(2)
        objects_data['objects'] = [self._create_coco_ssd_object_structure(i) for i in range(top_k) if scores[i] >= threshold]
        
        # Append objects onto image and add to structure
        if label_image: objects_data['labelled_image'] = self._annotate_coco_ssd_image(input_img, objects_data['objects'])

        # Add global data
        objects_data['inference_time'] = (end_ms - start_ms) * 1000

        # Return the objects structure
        return objects_data

    # Convenience Methods 
    def _get_coco_ssd_input_tensor(self):
        tensor_index = self._interpreter.get_input_details()[0]['index']
        return self._interpreter.tensor(tensor_index)()[0]

    def _set_coco_ssd_input_tensor(self, image, resample=Image.NEAREST):
        image = image.resize((self._get_coco_ssd_input_size()[0:2]), resample)
        image_tensor = self._get_coco_ssd_input_tensor()
        image_tensor[:, :] = image.copy()

    def _get_coco_ssd_input_size(self):
        _, height, width, channels = self._interpreter.get_input_details()[0]['shape']
        return width, height, channels

    def _get_coco_ssd_output_tensor(self, index):
        output_details = self._interpreter.get_output_details()[index]
        output_data = np.squeeze(self._interpreter.tensor(output_details['index'])())
        if 'quantization' not in output_details:
            return output_data
        scale, zero_point = output_details['quantization']
        if scale == 0:
            return output_data - zero_point
        return scale * (output_data - zero_point)

    def _create_coco_ssd_object_structure(self, i):
        boxes = self._get_coco_ssd_output_tensor(0)
        class_ids = self._get_coco_ssd_output_tensor(1)
        scores = self._get_coco_ssd_output_tensor(2)
        count = int(self._get_coco_ssd_output_tensor(3))

        ymin, xmin, ymax, xmax = boxes[i]
        return {
            "id" : int(class_ids[i]),
            "id_str" : self._labels.get(class_ids[i], class_ids[i]),
            "score" : float(scores[i]),
            "bbox" : {
                "xmin" : np.maximum(0.0, xmin),
                "ymin" : np.maximum(0.0, ymin),
                "xmax" : np.minimum(1.0, xmax),
                "ymax" : np.minimum(1.0, ymax)
            }
        }

    def _annotate_coco_ssd_image(self, image, objs) :
        # Get the height and width of the input image
        width, height = image.size
        channels = len(image.getbands())
        
        # Got through each object to annotate on image
        for obj in objs:
            # Get the box data of the object
            x0 = obj['bbox']['xmin']
            y0 = obj['bbox']['ymin']
            x1 = obj['bbox']['xmax']
            y1 = obj['bbox']['ymax']
            x0, y0, x1, y1 = int(x0*width), int(y0*height), int(x1*width), int(y1*height)
            
            # Create a label
            percent = int(100 * obj['score'])
            label = '{}% {}'.format(percent, obj['id_str'].title())

            # Create a draw object
            draw = ImageDraw.Draw(image)

            # Calculate text size
            text_width = draw.textsize(label)
            text_height = min(max((y1-y0) / text_width[1] * 30, 8), 36)
            text_font = ImageFont.truetype(font='data/Nasa.ttf', size=text_height)
            
            # Annotate image
            draw.rectangle([(x0, y0), (x1, y1)], outline=(0, 0, 0), width=4)
            draw.rectangle([(x0, y0), (x1, y1)], outline=(0, 96, 255), width=2)
            draw.text((x0+10, y0+10), label, fill=(0,0,0), font=text_font, stroke_width=1)
            draw.text((x0+10, y0+10), label, fill=(0, 96, 255), font=text_font, stroke_width=0)

        # Return annoted image
        return image

if __name__ == "__main__" :
    # Get env data
    model_path = os.environ.get('model_path', 'data/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite')
    labels_path = os.environ.get('labels_path', 'data/coco_labels.txt')
    threshold = float(os.environ.get('threshold', '0.6'))
    top_k = int(os.environ.get('top_k', '5'))

    # Create instance
    tpu_instance = edge_coco_ssd()
    tpu_instance.load_coco_ssd_model(model_path, labels_path)

    pass

# python3 /flask-ai/edge-server/edge_ai.py