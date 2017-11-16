#!/usr/bin/env bash

mkdir /data/tensorboard-logs/

tensorboard --logdir /data/tensorboard-logs/ &

jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 "$@"
