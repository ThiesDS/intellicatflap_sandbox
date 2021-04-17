import os
import argparse
import sys
import shutil

import numpy as np

from google.oauth2 import service_account
from helper import download_blob, upload_blob
from classify import batch_classify_cat


def main(gcs_path,destination):
    # Define paths
    gcs_download_folder = 'raw/'
    gcs_upload_folder = 'classified/'
    local_path = 'data/'

    # Create credentials 
    credentials = service_account.Credentials.from_service_account_file('../config/gcs_serviceaccount.json')
    
    # Download image
    print('Start downloading images.')
    download_blob('intellicatflap', gcs_download_folder + gcs_path, local_path, credentials)

    # Get downloaded files
    files = os.listdir(local_path)
    images = [file for file in files if file.endswith('.jpg')]
    images.sort()

    # Classification
    cat_probabilities = batch_classify_cat(local_path,images)
    print('Prediction results:')
    print(cat_probabilities)

    # Loop over predictions and upload to specific folder
    for image in images:
        
        # Decission if cat was detected (Cat: 0; No Cat: 1)
        if cat_probabilities[image] < 0.5:
            cat_detected = True
        else:
            cat_detected = False
        
        # Upload image depending on if cat was detected
        source_file_name = local_path + image
        
        if cat_detected:
            
            # Create destination folder
            destination_blob_folder = gcs_upload_folder + 'cat/'

            # Save
            if destination=='gcs':
                
                # Save to gcs
                upload_blob('intellicatflap', source_file_name, destination_blob_folder + image, credentials)

            elif destination=='local':

                # Save locally
                os.makedirs(local_path + destination_blob_folder,exist_ok=True)
                dest = shutil.copy(source_file_name, local_path + destination_blob_folder + image)
        else:

            # Create destination folder
            destination_blob_folder = gcs_upload_folder + 'no_cat/'

            # Save
            if destination=='gcs':
                
                # Save to gcs
                upload_blob('intellicatflap', source_file_name, destination_blob_folder + image, credentials)

            elif destination=='local':

                # Save locally
                os.makedirs(local_path + destination_blob_folder,exist_ok=True)
                dest = shutil.copy(source_file_name, local_path + destination_blob_folder + image)
        
        # Clean up
        os.remove(source_file_name)
        print(source_file_name)


if __name__ == "__main__":
    
    # Get gcs file path from argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--gcs_path", "--gcs_path", type=str, required=True, help='Path must not start with / but end with /.')
    parser.add_argument("--destination", "--destination", type=str, required=True, help='Upload to gcs or local')
    args = parser.parse_args(sys.argv[1:])
    
    # Call main
    print('Start Main.')
    main(gcs_path=args.gcs_path,destination=args.destination)