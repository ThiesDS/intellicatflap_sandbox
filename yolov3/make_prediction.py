import cv2
import os
import time

import numpy as np

from imageai import Detection
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

def load_image_pixels(filename, shape):
    # load the image to get its shape
    image = load_img(filename)
    width, height = image.size
    # load the image with the required size
    image = load_img(filename, target_size=shape)
    # convert to numpy array
    image = img_to_array(image)
    # scale pixel values to [0, 1]
    image = image.astype('float32')
    image /= 255.0
    # add a dimension so that we have one sample
    image = np.expand_dims(image, 0)
    return image, width, height

# Get image locations
base_dir = "/Users/administrator/private/intellicatflap_analytics/analysis/sample_data/calibration_sample/"
files = os.listdir(base_dir)
images = [img for img in files if img.endswith('.jpg')]
images.sort()

# Initialize yolo model
yolo = Detection.ObjectDetection()
yolo.setModelTypeAsTinyYOLOv3()
yolo.setModelPath('yolo-tiny.h5')
yolo.loadModel()

# Loop and predict
for image in images:

    start_time = time.time()

    # image path 
    input_image_path = base_dir + image
    output_image_path = base_dir + '.'.join(image.split('.')[0:-1]) + '_pred.jpg'
    
    # load image
    #image_array, image_w, image_h = load_image_pixels(input_image_path,(480,640))

    image_array = cv2.imread(input_image_path)
    print(image_array.shape)
    #image_array = cv2.resize(image_array,(120,160))

    print(f"Loadtime {time.time() - start_time}")
    
    ## predict yolo
    returned_images, detections = yolo.detectObjectsFromImage(
        input_type="array", 
        input_image=image_array,
        #input_image=input_image_path,
        #output_image_path=output_image_path,
        output_type='array',
        minimum_percentage_probability=50
    )

    print(f"Predtime {time.time() - start_time}")