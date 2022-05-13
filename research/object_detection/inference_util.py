from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import pandas as pd
import numpy as np
from PIL import Image
import cv2

import tensorflow as tf

from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops

"""
this function run inference on a single image and return the result as output_dict 
"""
def run_inference_for_single_image(model, image):
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  model_fn = model.signatures['serving_default']
  output_dict = model_fn(input_tensor)

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   
  # Handle models with masks:
  if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])      
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
  return output_dict


"""
this function run first run inference on image and then
do non max suppression and draw bounding boxes to the image
and return the image with bounding boxes.
"""
def post_process(model, image_np):
  # Actual detection.
  output_dict = run_inference_for_single_image(model, image_np)

  # convert box format to y1,x1,y2,x2 to use in tf non max suppression function
  det_boxes_yxyx = output_dict['detection_boxes'].copy()
  det_boxes_yxyx[:, [0,1,2,3]] = det_boxes_yxyx[:, [1,0,3,2]]

  # non maximum suppression
  det_boxes_tensor_yxyx = tf.convert_to_tensor(det_boxes_yxyx)
  selected_indices = tf.image.non_max_suppression(det_boxes_tensor_yxyx, output_dict['detection_scores'], 10, 0.15, 0.2) # get indexs of picked boxes after NMS

  # get the values corespond to the index in the previous step
  output_dict['detection_boxes'] = output_dict['detection_boxes'][selected_indices.numpy(), :]
  output_dict['detection_scores'] = output_dict['detection_scores'][selected_indices.numpy()]
  output_dict['detection_classes'] = output_dict['detection_classes'][selected_indices.numpy()]

  # Visualization of the results of a detection.

  category_index = label_map_util.create_category_index_from_labelmap('/content/labelmap.pbtxt', use_display_name=True)
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      min_score_thresh=0.2,
      instance_masks=output_dict.get('detection_masks_reframed', None),
      use_normalized_coordinates=True,
      line_thickness=2)

  return image_np