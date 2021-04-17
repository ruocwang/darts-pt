#!/bin/bash
script_name=`basename "$0"`
id=${script_name%.*}
dataset=${dataset:-cifar10}
seed=${seed:-0}
gpu=${gpu:-"auto"}

while [ $# -gt 0 ]; do
    if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
    fi
    shift
done

echo 'id:' $id
echo 'seed:' $seed
echo 'dataset:' $dataset
echo 'gpu:' $gpu

cd ../nasbench201/
python train_search.py \
    --method blank \
    --dataset $dataset \
    --save $id --gpu $gpu --seed $seed \
    # --fast --expid_tag debug \