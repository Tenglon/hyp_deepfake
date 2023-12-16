#!/bin/bash
# source ~/miniconda3/etc/profile.d/conda.sh
# source activate hyper

# python -m train \
#     euclidean-8-16-768-resnet-20 \
#     ffppc23 \
#     -e 100 \
#     -s \
#     --opt=adam \
#     --lr=0.0001 \
#     --weight-decay=1e-4



# python -m train \
#     hyperbolic-8-16-768-resnet-20 \
#     ffppc23 \
#     -e 100 \
#     -s \
#     --opt=adam \
#     --lr=0.0001 \
#     --weight-decay=1e-4



# python -m train \
#     hyperbolic-8-16-768-fc0-20 \
#     ffppc23 \
#     -e 100 \
#     -s \
#     --opt=adam \
#     --lr=0.001 \
#     --weight-decay=1e-4



# python -m train \
#     euclidean-8-16-768-fc0-20 \
#     ffppc23 \
#     -e 100 \
#     -s \
#     --opt=adam \
#     --lr=0.001 \
#     --weight-decay=1e-4




python -m train \
    hyperbolic-8-16-768-fc0relufc1relufc2lr-20 \
    ffppc23 \
    -e 100 \
    -s \
    --opt=adam \
    --lr=0.0001 \
    --weight-decay=1e-4




# python -m train \
#     euclidean-8-16-768-fc0relufc1relufc2-20 \
#     ffppc23 \
#     -e 100 \
#     -s \
#     --opt=adam \
#     --lr=0.001 \
#     --weight-decay=1e-4