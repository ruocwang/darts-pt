#!/bin/bash
script_name=`basename "$0"`
id=${script_name%.*}
dataset=${dataset:-cifar10}
seed=${seed:-2}
gpu=${gpu:-"auto"}

## dev mode
space=${space:-s5}
resume_epoch=${resume_epoch:-50}
resume_expid=${resume_expid:-'search-sdarts-rs-sota-s5-2'}

while [ $# -gt 0 ]; do
    if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
        # echo $1 $2 // Optional to see the parameter:value result
    fi
    shift
done

echo 'id:' $id 'seed:' $seed 'dataset:' $dataset 'space:' $space
echo 'resume_epoch:' $resume_epoch 'resume_expid' $resume_expid
echo 'gpu:' $gpu

cd ../sota/cnn
python train_search.py \
    --method sdarts-proj --perturb_alpha random \
    --search_space $space --dataset $dataset \
    --seed $seed --save $id --gpu $gpu \
    --resume_epoch $resume_epoch --resume_expid $resume_expid --dev proj \
    --edge_decision random \
    --proj_crit_normal acc --proj_crit_reduce acc --proj_crit_edge acc --proj_intv 5 \
    --fast \
    # --fast --log_tag debug \

## bash sdarts-rs-next-sota.sh --resume_expid search-sdarts-rs-sota-debug-s5-2
