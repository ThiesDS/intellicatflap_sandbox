import os
import argparse
import sys

import numpy as np

from google.oauth2 import service_account
from helper import download_blob, upload_blob
from classify import parallel_cat_detection


def main(gcs_path):
    # Define paths
    gcs_download_folder = 'raw/'
    gcs_upload_folder = 'classified/'
    local_path = 'data/'

    # Create credentials 
    credentials = service_account.Credentials.from_service_account_file('../config/gcs_serviceaccount.json')

    # Download image
    download_blob('intellicatflap', gcs_download_folder + gcs_path, local_path, credentials)

    # Get downloaded files
    files = os.listdir(local_path)
    images = [file for file in files if file.endswith('.jpg')]
    images.sort()
    image_paths = [local_path+image for image in images]
    print(images)
    
    # Classify image and upload it depending on the classification
    
    # Classification
    cat_probabilities = parallel_cat_detection(image_paths)
    print(cat_probabilities)

    # ...
    for image in images:
        
        cat_detected = False

        
        # Upload image
        source_file_name = local_path + image
        gcs_name = gcs_path.replace('/','_') + image
        destination_blob_name = gcs_upload_folder + 'no_cat/' + gcs_path + gcs_name

        if cat_detected:
            upload_blob('intellicatflap', source_file_name, destination_blob_name, credentials)
        else:
            upload_blob('intellicatflap', source_file_name, destination_blob_name, credentials)
        
        # Clean up
        os.remove(source_file_name)
        print(source_file_name)


if __name__ == "__main__":
    
    # Get gcs file path from argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--gcs_path", "--gcs_path", type=str, required=True, help='Path must not start or end with /.')
    args = parser.parse_args(sys.argv[1:])
    print('test print')
    # Call main
    main(gcs_path=args.gcs_path)