import cv2
import json
import aiohttp
import asyncio

import numpy as np

from google.cloud import storage
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


def download_blob(bucket_name, source_blob_path, destination_path, credentials):
    """Downloads a blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket(bucket_name)

    # Get all blobs in that path (Construct a client side representation of a blob.)
    blobs = bucket.list_blobs(prefix=source_blob_path)
    
    # Iterate over blobs and download
    for blob in blobs:
        file_name = '_'.join(blob.name.split('/')[1:])
        blob.download_to_filename(destination_path + file_name)
        print('Downloaded ' + str(file_name))

    
def upload_blob(bucket_name, source_file_name, destination_blob_name, credentials):
    """Uploads a file to the bucket."""
    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)


# Asyncronous requests
async def fetch(session, url, data, headers):
    async with session.post(url,data=data,headers=headers) as response:
        resp = await response.json()
        return resp


async def detect_cat(image_paths):
    async with aiohttp.ClientSession() as session:
        
        tasks = []
        for image_path in image_paths:
            image_np = cv2.imread(image_path).astype('uint8')
            image_np = np.expand_dims(image_np, axis=0)
            image_list = image_np.tolist()
            
            headers = {"content-type": "application/json"}
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