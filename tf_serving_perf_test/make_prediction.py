
import json
import requests
import os
import time
import aiohttp
import asyncio
import random
#import cv2
from PIL import Image

import numpy as np

# USE conda env "intellicatflap_analytics"


# Main spec
REQUEST_TYPE = 'sequential'


# Detect objects
cwd = os.getcwd()
image_dir = cwd + '/sample_data/calibration_sample/'

files = os.listdir(image_dir)
image_paths = [image_dir + img for img in files if img.endswith('.jpg')]
image_paths.sort()

headers = {"content-type": "application/json"}

# 1. Sequentially sending requests
if REQUEST_TYPE == 'sequential':
    
    start_time = time.time()

    for image_path in image_paths:

        #image_np = cv2.imread(image_path).astype('uint8')
        image_np = Image.open(image_path)
        image_np = np.expand_dims(image_np, axis=0)
        image_np = image_np.astype('uint8')

        data = json.dumps({"signature_name": "serving_default", "instances": image_np.tolist()})
        
        json_response = requests.post('http://localhost:8501/v1/models/mobilenet:predict', data=data, headers=headers)

        predictions = json.loads(json_response.text)['predictions']
        
        print('Predicted score: ' + str(predictions[0]['detection_scores'][0]))
        print('Predicted class: ' + str(predictions[0]['detection_classes'][0]))

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
                image_np = cv2.imread(image_path).astype('uint8')
                image_np = np.expand_dims(image_np, axis=0)
                image_list = image_np.tolist()
                
                url = 'http://localhost:8501/v1/models/mobilenet:predict'
                data = json.dumps({"signature_name": "serving_default", "instances": image_list})
                
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
    #image_paths.extend(image_paths)
    #image_paths.extend(image_paths)

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
            predictions = responses[idx]['predictions'][0]

            try:
                # get location of (first) cat class in detected classes and 
                cat_class_idx = predictions['detection_classes'].index(17.0)
                cat_probability = predictions['detection_scores'][cat_class_idx]
            except:
                # if not there, probability is 0 
                cat_probability = 0.0
            
            idx += 1
            print(str(idx) + '. image cat probability: ' + str(cat_probability))

        except:
            print(responses[idx])
            #print(f"Image {image_path} is not behaving well.")
