import time
import json
import requests

from helper import detect_cat,load_image

def batch_classify_cat(local_path,images):

    # Define request parameters
    headers = {"content-type": "application/json"}
    url = 'http://localhost:8501/v1/models/cat_classifier:predict'

    # Initialize
    cat_probabilities = dict()
    
    # Loop over every image
    print('Start prediction loop.')
    for image in images:
        
        # Load image
        image_path = local_path+image
        image_np = load_image(image_path)

        # Create json structure for request
        data = json.dumps({"signature_name": "serving_default", "instances": image_np.tolist()})
        
        # Send request
        json_response = requests.post(url, data=data, headers=headers)

        # Cat: 0; No Cat: 1
        cat_probability = json.loads(json_response.text)['predictions'][0][0]

        # Save to dict
        cat_probabilities[image] = cat_probability

    # Return
    return cat_probabilities