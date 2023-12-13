#!/bin/bash
# source ~/miniconda3/etc/profile.d/conda.sh
# source activate hyper

# python -m train \
#     euclidean-8-16-32-resnet-20 \
#     cifar10 \
#     -e 100 \
#     -s \
#     --opt=adam \
#     --lr=0.001 \
#     --weight-decay=1e-4



python -m train \
    hyperbolic-8-16-768-resnet-20 \
    ffppc23 \
    -e 100 \
    -s \
    --opt=adam \
    --lr=0.0001 \
    --weight-decay=1e-4