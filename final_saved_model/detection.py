'''
Authors: Prateek Kumar Singh, s3890089
         Kanimozhi Udayakumar, s3913700

Org: RMIT University
'''

#Loading the saved_model
import tensorflow as tf
import time
import numpy as np
import warnings
import cv2 as cv
import time
warnings.filterwarnings('ignore')
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="image path")
ap.add_argument("-m","--saved_model",required=True,help="saved model path")
args = vars(ap.parse_args())

def load_image_into_numpy_array(path):
    return np.array(Image.open(path))


PATH_TO_SAVED_MODEL=args["saved_model"]
# Load saved model and build the detection function
detect_fn=tf.saved_model.load(PATH_TO_SAVED_MODEL)

start = time.time()

image_path = args["image"]
image = Image.open(image_path)
image_np = load_image_into_numpy_array(image_path)
image_np = cv.cvtColor(image_np, cv.COLOR_BGR2RGB)

# The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
input_tensor = tf.convert_to_tensor(image_np)
# The model expects a batch of images, so add an axis with `tf.newaxis`.
input_tensor = input_tensor[tf.newaxis, ...]
detections = detect_fn(input_tensor)
# All outputs are batches tensors.
# Convert to numpy arrays, and take index [0] to remove the batch dimension.
# We're only interested in the first num_detections.
num_detections = int(detections.pop('num_detections'))
detections = {key: value[0, :num_detections].numpy()
              for key, value in detections.items()}
detections['num_detections'] = num_detections

# detection_classes should be ints.
detections['detection_classes'] = detections['detection_classes'].astype(np.int64)


if(detections['detection_scores'][0]>0.2):
    im_width, im_height = image.size
    (left, right, top, bottom) = (detections['detection_boxes'][0][1] * im_width, detections['detection_boxes'][0][3] * im_width,
                                  detections['detection_boxes'][0][0] * im_height, detections['detection_boxes'][0][2] * im_height)
    
    im1 = image.crop((left, top, right, bottom))
    rgb_im = im1.convert('RGB')
    rgb_im.save('plate.jpg')

    # overlay
    logo_path ='logo.png'
    logo = Image.open(logo_path)
    width, height = im1.size
    logo_resized = logo.resize((width,height))
    image.paste(logo_resized, (int(left),int(top)), mask = logo_resized)

    end = time.time()
    print(end - start)

    # Displaying the image
    image.show()
    rgb_im2 = image.convert('RGB')
    rgb_im2.save('out.jpg')

else: 
    print('No License Plate Detected')
    if os.path.exists("plate.jpg"):
        os.remove("plate.jpg")
