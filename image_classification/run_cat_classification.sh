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
docker run -d -p 8501:8501 -p 8500:8500 \
  --name=tf_serving_for_classification \
  -v $PARENTDIR/intellicatflap/src/tf_serving/models/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/saved_model:/models/mobilenet/1 \
  -v $PARENTDIR/intellicatflap/src/tf_serving/config:/config \
  -e MODEL_NAME=mobilenet \
  -t tensorflow/serving \
  #--enable_batching \
  #--batching_parameters_file=/config/batching_config.txt

# Run classification 
python src/main.py --gcs_path='2021/03/23/03/46/'