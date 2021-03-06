
import json
import requests
import os
import time
import aiohttp
import asyncio
import random
import cv2

import numpy as np

## Main part
# USE conda env 

# Detect objects
image_dir = '/Users/administrator/private/intellicatflap_analytics/analysis/sample_data/calibration_sample/'

files = os.listdir(image_dir)
image_paths = [image_dir + img for img in files if img.endswith('.jpg')]
image_paths.sort()

headers = {"content-type": "application/json"}

## 1. Sequentially sending requests
#for image_path in image_paths:
#    
#    start_time = time.time()
#
#    image_np = cv2.imread(image_path).astype('uint8')
#    image_np = np.expand_dims(image_np, axis=0)
#
#    data = json.dumps({"signature_name": "serving_default", "instances": image_np.tolist()})
#    
#    print(f"Load image: {time.time()-start_time}")
#    json_response = requests.post('http://localhost:8501/v1/models/resnet:predict', data=data, headers=headers)
#    print(f"Load image + api call: {time.time()-start_time}")
#
#    predictions = json.loads(json_response.text)['predictions']
#    
#    print(predictions[0]['detection_scores'][0])
#    print(predictions[0]['detection_classes'][0])


# 2. Sending batch requests -> NOT WORKING, BC the model does not allow batch processing of inputs
#start_time = time.time()
#image_np = np.ndarray(shape=(0, 640, 480, 3))
#for image_path in image_paths[0:3]: 
#    image_np_load = cv2.imread(image_path)
#    image_np = np.append(image_np,[image_np_load],0).astype('uint8')
#
#data = json.dumps({"signature_name": "serving_default", "instances": image_np.tolist()})
#    
#print(f"Load image: {time.time()-start_time}")
#json_response = requests.post('http://localhost:8501/v1/models/resnet:predict', data=data, headers=headers)
#print(f"Load image + api call: {time.time()-start_time}")
#
#predictions = json.loads(json_response.text)
#print(predictions['error'])
#
#print(predictions[0]['detection_scores'][0])
#print(predictions[0]['detection_classes'][0])


# 3. Sending asyncronous requests
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
            
            url = 'http://localhost:8501/v1/models/resnet:predict'
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
image_paths.extend(image_paths)
image_paths.extend(image_paths)

print("Warmup")
start_time = time.time()
responses = asyncio.run(fetch_all(image_paths[:3]))
print(f"It took {time.time() - start_time}s to process {len(image_paths[:3])} images.")

print(len(image_paths))
print(image_paths[:3])

start_time = time.time()
responses = asyncio.run(fetch_all(image_paths))
print(f"It took {time.time() - start_time}s to process {len(image_paths)} images.")


random.shuffle(image_paths)
print(image_paths[:3])

start_time = time.time()
responses = asyncio.run(fetch_all(image_paths))
print(f"It took {time.time() - start_time}s to process {len(image_paths)} images.")


random.shuffle(image_paths)
print(image_paths[:3])

start_time = time.time()
responses = asyncio.run(fetch_all(image_paths))
print(f"It took {time.time() - start_time}s to process {len(image_paths)} images.")