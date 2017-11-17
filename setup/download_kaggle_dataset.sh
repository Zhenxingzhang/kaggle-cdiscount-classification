#!/usr/bin/env bash
kg download -u ${1} -p ${2} -c cdiscount-image-classification-challenge -f train.bson
kg download -u ${1} -p ${2} -c cdiscount-image-classification-challenge -f test.bson

mv *.bson /data/data/