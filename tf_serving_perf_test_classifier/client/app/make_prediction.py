
import json
import requests
import os
import time
import aiohttp
import asyncio
import random
#import cv2
from PIL import Image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

import numpy as np

def load_image(filename):
    # load the image
    img = load_img(filename, target_size=(224, 224))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 3 channels
    img = img.reshape(1, 224, 224, 3)
    # center pixel data
    img = img.astype('float32')
    img = img - [123.68, 116.779, 103.939]
    return img


# USE conda env "intellicatflap_analytics"

# Main spec
REQUEST_TYPE = 'sequential'


# Detect objects
cwd = '/app/data'
image_dir = cwd + '/sample_data/calibration_sample/'

files = os.listdir(image_dir)
image_paths = [image_dir + img for img in files if img.endswith('.jpg')]
image_paths.sort()

headers = {"content-type": "application/json"}
url = 'http://cat-classifier-tfserving:8501/v1/models/cat_classifier:predict'

# 1. Sequentially sending requests
if REQUEST_TYPE == 'sequential':
    
    start_time = time.time()

    for image_path in image_paths:
        
        image_np = load_image(image_path)

        data = json.dumps({"signature_name": "serving_default", "instances": image_np.tolist()})
        
        json_response = requests.post(url, data=data, headers=headers)

        predictions = json.loads(json_response.text)['predictions']

        print('Predicted probability: ' + str(predictions[0][0]))


    print(f"It took {time.time() - start_time}s to process {len(image_paths)} images.")


# 2. Sending asyncronous requests
elif REQUEST_TYPE == 'concurrent':

    async def fetch(session, url, data, headers):
        async with session.post(url,data=data,headers=headers) as response:
            resp = await response.json()
            return resp


    async def fetch_all(image_paths):
        async with aiohttp.ClientSession() as session:
            
            tasks = []
            for image_path in image_paths:
                
                image_np = load_image(image_path)

                data = json.dumps({"signature_name": "serving_default", "instances": image_np.tolist()})
                
                tasks.append(
                    fetch(
                        session,
                        url,
                        data,
                        headers
                    )
                )

            responses = await asyncio.gather(*tasks, return_exceptions=True)
            return responses


    # Make requests

    print("Warmup")
    warumup_imgs = image_paths[:3]
    start_time = time.time()
    responses = asyncio.run(fetch_all(warumup_imgs))
    print(f"It took {time.time() - start_time}s to process {len(image_paths[:3])} images.")

    start_time = time.time()
    responses = asyncio.run(fetch_all(image_paths))
    print(f"It took {time.time() - start_time}s to process {len(image_paths)} images.")

    idx = 0
    for image_path in image_paths:
        try:
            # Extract predictions from responses
            #print(responses[idx])
            predictions = responses[idx]['predictions']

            print('Predicted probability: ' + str(predictions[0][0]))

        except:
            print(responses[idx])
            #print(f"Image {image_path} is not behaving well.")
