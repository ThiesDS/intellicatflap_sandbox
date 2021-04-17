#!/bin/bash

# Set up correct environment
eval "$(conda shell.bash hook)"
conda deactivate
conda activate intellicatflap_analytics_image_classification

# Define parent directory, where intellicatflap and intellicatflap_analyitcs is located
export PARENTDIR=$HOME/private/

# Kill and/or remove running containers
docker kill tf_serving_for_classification
docker rm tf_serving_for_classification

# Spin up tf serving
docker run -d -p 8501:8501 \
  --name=tf_serving_for_classification \
  -v $PARENTDIR/intellicatflap_analytics/models/cat_classifier_v0_tobivscatflap/saved_model:/models/cat_classifier/1 \
  -e MODEL_NAME=cat_classifier \
  -t tensorflow/serving

# Run classification (path must end with /; destination: 'local' or 'gcs')
python src/main.py --gcs_path='2021/04/09/05/37/' --destination='local'