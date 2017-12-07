#!/usr/bin/env bash

mkdir -p /data/frozen/inception
curl -o /data/frozen/inception/inception.tgz http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
tar xfz /data/frozen/inception/inception.tgz -C /data/frozen/inception

curl -o /data/inception/2016/inception_v3.tar.gz http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
tar xfz /data/inception/2016/inception_v3.tar.gz -C /data/inception/2016