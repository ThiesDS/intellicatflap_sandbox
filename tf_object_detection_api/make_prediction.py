import os
import pathlib
import time

import matplotlib
import matplotlib.pyplot as plt

import io
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

def load_image_into_numpy_array(path):
  """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: the file path to the image

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
  img_data = tf.io.gfile.GFile(path, 'rb').read()

  image = Image.open(BytesIO(img_data))

  image = image.resize((640,640))
  
  (im_width, im_height) = image.size

  image_array = np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
  
  return image_array

def get_model_detection_function(model):
  """Get a tf.function for detection."""

  @tf.function
  def detect_fn(image):
    """Detect objects in image."""
    
    start_time = time.time()
    image, shapes = model.preprocess(image)
    preprocess_time = time.time()

    prediction_dict = model.predict(image, shapes)
    predict_time = time.time()
    
    detections = model.postprocess(prediction_dict, shapes)
    postprocess_time = time.time()

    print(f"Times: {preprocess_time-start_time}, {predict_time-preprocess_time}, {postprocess_time-predict_time}")

    return detections, prediction_dict, tf.reshape(shapes, [-1])

  return detect_fn


## Load pretrained model 
# Set paths
pipeline_config = "/Users/administrator/private/models/research/object_detection/configs/tf2/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.config"
model_dir = "/Users/administrator/private/intellicatflap_analytics/analysis/tf_object_detection_api/models/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/checkpoint/"

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(pipeline_config)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(model_dir, 'ckpt-0')).expect_partial()

# Get detection function
detect_fn = get_model_detection_function(detection_model)


# Label map
#label_map_path = "/Users/administrator/private/intellicatflap_analytics/analysis/tf_object_detection_api/labels/mscoco_label_map.pbtxt"
#label_map = label_map_util.load_labelmap(label_map_path)
#categories = label_map_util.convert_label_map_to_categories(
#    label_map,
#    max_num_classes=label_map_util.get_max_label_map_index(label_map),
#    use_display_name=True)
#category_index = label_map_util.create_category_index(categories)
#label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)


## Main part
# Detect objects
image_dir = '/Users/administrator/private/intellicatflap_analytics/analysis/sample_data/calibration_sample/'

thresh = 0.3

files = os.listdir(image_dir)
image_paths = [image_dir + img for img in files if img.endswith('.jpg')]
image_paths.sort()

for image_path in image_paths:
    image_np = load_image_into_numpy_array(image_path)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    
    start_time = time.time()
    detections, predictions_dict, shapes = detect_fn(input_tensor)
    print(f"Time: {time.time()-start_time}")

    cat_detections = [[cat_class.numpy(),cat_score.numpy(),cat_box.numpy()] for cat_class,cat_score,cat_box in zip(detections['detection_classes'][0],detections['detection_scores'][0],detections['detection_boxes'][0]) if cat_class == 16 and cat_score >= thresh]
    print(cat_detections)
    
    for cat_detection in cat_detections:
        print(f"Predict cat with {cat_detection[1]} probability. At location {cat_detection[2]}")
        
        with open("tmp.det", "w") as file:
            file.write(f"{cat_detection[1]}, {cat_detection[2]}")



#label_id_offset = 1
#image_np_with_detections = image_np.copy()
#
## Use keypoints if available in detections
#keypoints, keypoint_scores = None, None
#if 'detection_keypoints' in detections:
#  keypoints = detections['detection_keypoints'][0].numpy()
#  keypoint_scores = detections['detection_keypoint_scores'][0].numpy()
#
#viz_utils.visualize_boxes_and_labels_on_image_array(
#      image_np_with_detections,
#      detections['detection_boxes'][0].numpy(),
#      (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
#      detections['detection_scores'][0].numpy(),
#      category_index,
#      use_normalized_coordinates=True,
#      max_boxes_to_draw=200,
#      min_score_thresh=.30,
#      agnostic_mode=False,
#      keypoints=keypoints,
#      keypoint_scores=keypoint_scores,
#      keypoint_edges=None)
#
#image_with_detections = Image.fromarray(image_np_with_detections)
#image_with_detections.save(image_dir + 'pred_' + image_name, "JPEG")