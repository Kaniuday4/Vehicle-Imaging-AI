'''
Authors: Prateek Kumar Singh, s3890089
         Kanimozhi Udayakumar, s3913700

Org: RMIT University
USAGE: python3 detection.py -i inputs/<input_image> -m saved_model/saved_model ;python3 OCR.py
'''
# Loading Libraries
import cv2
import os
import pandas as pd
import df2img
import numpy as np
import math
from google.cloud import vision_v1p3beta1 as vision

# --------------------------------------------------------------------
# IMPORTANT: Change Google Cloud Credentials before running this file.
# --------------------------------------------------------------------
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/prateek/Desktop/MDS/Semester4/Project/calcium-doodad-384001-2375ac3209b7.json'

try:
    img_path = "plate.jpg"
    Plate = cv2.imread(img_path)
    cv2.imshow("Detected Plate", Plate)
    cv2.waitKey(0)

except:
    exit()

'''
Below are the Functions to fix the orientation of the license plate for better text detection.
'''
def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def compute_skew(src_img):

    if len(src_img.shape) == 3:
        h, w, _ = src_img.shape
    elif len(src_img.shape) == 2:
        h, w = src_img.shape
    else:
        print('unsupported image type')

    img = cv2.medianBlur(src_img, 3)

    edges = cv2.Canny(img,  threshold1 = 30,  threshold2 = 100, apertureSize = 3, L2gradient = True)
    lines = cv2.HoughLinesP(edges, 1, math.pi/180, 30, minLineLength=w / 4.0, maxLineGap=h/4.0)
    angle = 0.0
    nlines = lines.size

    cnt = 0
    for x1, y1, x2, y2 in lines[0]:
        ang = np.arctan2(y2 - y1, x2 - x1)
        if math.fabs(ang) <= 30: # excluding extreme rotations
            angle += ang
            cnt += 1

    if cnt == 0:
        return 0.0
    return (angle / cnt)*180/math.pi

def deskew(src_img):
    return rotate_image(src_img, compute_skew(src_img))
    

corrected_img = deskew(Plate)
cv2.imshow("Plate with Fixed Orientation", corrected_img)
cv2.waitKey(0)


### Google Vision

client = vision.ImageAnnotatorClient()
success, encoded_image = cv2.imencode('.jpg', corrected_img)
image = encoded_image.tobytes()
image = vision.types.Image(content=image)
response = client.text_detection(image=image)

texts=response.text_annotations

try:
    txt = texts[0].description
except:
    print("Text Detection Unsuccessful!")
    exit()
         
tokens = txt.split('\n')
tokens = [x.replace(" ", "") for x in tokens] # Removing all whitespaces b/w the detected text.
discard = []
for lic_num in tokens:
    if len(lic_num) > 8:
        discard.append(lic_num)
possible_license=[x for x in tokens if x not in discard]
print(possible_license)

# Converting Results to a pandas dataframe.
result = pd.DataFrame(possible_license,index = range(1,len(possible_license)+1), columns = ['Possible Plate Number'])  
result_fig = df2img.plot_dataframe(result)
df2img.save_dataframe(fig=result_fig, filename="result.png") # Visualzing dataframe as an image.

result_df = cv2.imread("result.png")
cv2.imshow("Plate Numbers", result_df )
cv2.waitKey(0)

# END
