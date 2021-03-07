#!/bin/bash

docker kill tfserving_mobilenet
docker rm tfserving_mobilenet
docker run -p 8501:8501 -p 8500:8500 \
    --name tfserving_mobilenet \
    --mount type=bind,source=$(pwd)/models/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model,target=/models/resnet/1/ \
    --mount type=bind,source=$(pwd)/tf_serving/config,target=/config/ \
    -e MODEL_NAME=resnet \
    -e TF_CPP_MIN_VLOG_LEVEL=2 \
    -t tensorflow/serving \
    --enable_batching \
    --batching_parameters_file=/config/batching_config.txt
