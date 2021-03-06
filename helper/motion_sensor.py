import numpy as np
import cv2
import os

base_dir = "/Users/administrator/private/intellicatflap_analytics/analysis/sample_data/calibration_sample/"
files = os.listdir(base_dir)

images = [img for img in files if img.endswith('.jpg')]

images.sort(reverse=False)

print(images)

for i in range(1,len(images)):

    img1 = cv2.imread(base_dir + images[i-1],0)
    img2 = cv2.imread(base_dir + images[i],0)
    
    img_diff_12 = (img2-img1)**2
    img_diff_12_sum = np.sum(img_diff_12)/(img_diff_12.shape[0]*img_diff_12.shape[1])

    print(img_diff_12_sum)

# THRESHOLD: 10