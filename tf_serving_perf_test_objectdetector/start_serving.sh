#!/bin/bash

docker kill tfserving_mobilenet
docker rm tfserving_mobilenet
docker run -p 8501:8501 \
    --name tfserving_mobilenet \
    --mount type=bind,source=$(pwd)/models/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/saved_model,target=/models/mobilenet/1/ \
    --mount type=bind,source=$(pwd)/tf_serving_perf_test_objectdetector/config,target=/config/ \
    -e MODEL_NAME=mobilenet \
    -e OMP_NUM_THREADS=4 \
    -e TENSORFLOW_INTER_OP_PARALLELISM=2 \
    -e TENSORFLOW_INTRA_OP_PARALLELISM=4 \
    -t tensorflow/serving
    #--enable_batching \
    #--batching_parameters_file=/config/batching_config.txt