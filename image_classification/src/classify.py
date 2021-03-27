import asyncio
import time
import json

from helper import detect_cat

def parallel_cat_detection(image_paths):

    # TODO: Parallelize paths 


    # Make async calls to tf-serving
    start_time = time.time()
    print(image_paths)
    responses = asyncio.run(detect_cat(image_paths))
    print(f"It took {time.time() - start_time}s to process {len(image_paths)} images.")
    
    cat_probabilities = dict()
    idx = 0
    
    for image_path in image_paths:
        try:        
            # Extract predictions from responses
            predictions = responses[idx]['predictions'][0]

            try:
                # get location of (first) cat class in detected classes and 
                cat_class_idx = predictions['detection_classes'].index(17.0)
                cat_probability = predictions['detection_scores'][cat_class_idx]
            except:
                # if not there, probability is 0 
                cat_probability = 0.0
            
            # Save to dict
            cat_probabilities[image_path] = cat_probability

            idx += 1

            print('try')
        except:
            print(f"Image {image_path} is not behaving well.")
            print('except')
    

    # Collect and convert responses


    # Return
    return cat_probabilities