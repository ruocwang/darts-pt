#!/bin/bash
script_name=`basename "$0"`
id=${script_name%.*}
dataset=${dataset:-cifar10}
seed=${seed:-2}
gpu=${gpu:-"auto"}
space=${space:-s5}

while [ $# -gt 0 ]; do
    if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
        # echo $1 $2 // Optional to see the parameter:value result
    fi
    shift
done


echo 'id:' $id 'seed:' $seed 'dataset:' $dataset 'space:' $space
echo 'gpu:' $gpu

cd ../sota/cnn
python train_search.py \
    --method sdarts --perturb_alpha random \
    --search_space $space --dataset $dataset \
    --seed $seed --save $id --gpu $gpu \
    --expid_tag debug \