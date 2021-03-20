import os

import numpy as np

from google.oauth2 import service_account
from helper import download_blob, upload_blob

# Define paths
gcs_download_folder = 'raw/'
gcs_upload_folder = 'classified/'
local_path = 'data/'

# Create credentials 
credentials = service_account.Credentials.from_service_account_file('../config/gcs_serviceaccount.json')

# Download image
gcs_path = '2021/03/20/10/12/'
download_blob('intellicatflap', gcs_download_folder + gcs_path, local_path, credentials)

# Get downloaded files
images = os.listdir(local_path)

# Classify image and upload it depending on the classification
for image in images:
    
    # Classification
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
