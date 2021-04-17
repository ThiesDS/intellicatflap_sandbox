#!/bin/bash

# Build optimized tf serving for the hardware
docker buildx build --pull -t $USER/tensorflow-serving-devel -f /Users/administrator/private/serving/tensorflow_serving/tools/docker/Dockerfile.devel . --builder larger_log2