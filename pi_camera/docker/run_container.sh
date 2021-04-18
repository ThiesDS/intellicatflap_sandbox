#!/bin/bash

docker run -v /home/pi/sandbox:/home \ 
           --device /dev/video0 \
           -it \
           sgtwilko/rpi-raspbian-opencv \ 
           /bin/bash