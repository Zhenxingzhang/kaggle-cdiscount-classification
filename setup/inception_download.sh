#!/usr/bin/env bash

mkdir -p /data/frozen/inception
curl -o /data/frozen/inception/inception.tgz http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
tar xfz /data/frozen/inception/inception.tgz -C ./data/frozen/inception
