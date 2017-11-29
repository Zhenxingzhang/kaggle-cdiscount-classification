#!/usr/bin/env bash

mkdir /data/tensorboard-logs/

tensorboard --logdir /data/summary/ &

jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --notebook-dir='/notebooks' "$@"

#python -m src.training.train_model -c ./config/linear_LR_1.yml