#!/bin/bash

docker kill tfserving_mobilenet
docker rm tfserving_mobilenet
docker run -p 8501:8501 -p 8500:8500 \
    --name tfserving_mobilenet \
    -v $(pwd)/models/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/saved_model:/models/resnet/1/ \
    -v $(pwd)/tf_serving/config:/config/ \
    -d \
    -e MODEL_NAME=resnet \
    -e TF_CPP_MIN_VLOG_LEVEL=4 \
    -t emacski/tensorflow-serving:2.4.1-linux_arm_armv7-a_neon_vfpv4 \
    --enable_batching \
    --batching_parameters_file=/config/batching_config.txt
