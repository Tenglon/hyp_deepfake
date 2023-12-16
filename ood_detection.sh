#!/bin/bash
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate poincare_resnet

# declare -a datasets=(
#     "ffppc23"
  
# )

# declare -a models=(
#     "hyperbolic-8-16-768-resnet-20"
    
# )

# for dataset in "${datasets[@]}"; do
#     echo $dataset
#     for model in "${models[@]}"; do
#         echo $model
#         python -m ood_detection \
#             --model $model \
#             --dataset $dataset \
#             --num_to_avg 10
#     done
# done


# declare -a datasets=(
#     "ffppc23"
  
# )

# declare -a models=(
#     "euclidean-8-16-768-resnet-20"
    
# )

# for dataset in "${datasets[@]}"; do
#     echo $dataset
#     for model in "${models[@]}"; do
#         echo $model
#         python -m ood_detection \
#             --model $model \
#             --dataset $dataset \
#             --num_to_avg 10
#     done
# done




# declare -a datasets=(
#     "ffppc23"
  
# )

# declare -a models=(
#     "hyperbolic-8-16-768-fc0-20"
    
# )

# for dataset in "${datasets[@]}"; do
#     echo $dataset
#     for model in "${models[@]}"; do
#         echo $model
#         python -m ood_detection \
#             --model $model \
#             --dataset $dataset \
#             --num_to_avg 10
#     done
# done




# declare -a datasets=(
#     "ffppc23"
  
# )

# declare -a models=(
#     "euclidean-8-16-768-fc0-20"
    
# )

# for dataset in "${datasets[@]}"; do
#     echo $dataset
#     for model in "${models[@]}"; do
#         echo $model
#         python -m ood_detection \
#             --model $model \
#             --dataset $dataset \
#             --num_to_avg 10
#     done
# done

# declare -a datasets=(
#     "ffppc23"
  
# )

# declare -a models=(
#     "hyperbolic-8-16-768-fc0relufc1relufc2-20"
    
# )

# for dataset in "${datasets[@]}"; do
#     echo $dataset
#     for model in "${models[@]}"; do
#         echo $model
#         python -m ood_detection \
#             --model $model \
#             --dataset $dataset \
#             --num_to_avg 10
#     done
# done

declare -a datasets=(
    "ffppc23"
  
)

declare -a models=(
    "euclidean-8-16-768-fc0relufc1relufc2-20"
    
)

for dataset in "${datasets[@]}"; do
    echo $dataset
    for model in "${models[@]}"; do
        echo $model
        python -m ood_detection \
            --model $model \
            --dataset $dataset \
            --num_to_avg 10
    done
done