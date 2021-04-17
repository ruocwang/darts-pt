#!/bin/bash
script_name=`basename "$0"`
id=${script_name%.*}
dataset=${dataset:-cifar10}
seed=${seed:-0}
gpu=${gpu:-"auto"}

resume_epoch=${resume_epoch:-100}
resume_expid=${resume_expid:-"search-blank-201-1"}

while [ $# -gt 0 ]; do
    if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
    fi
    shift
done

echo 'id:' $id 'seed:' $seed 'dataset:' $dataset
echo 'resume_epoch:' $resume_epoch 'resume_expid' $resume_expid
echo 'gpu:' $gpu


cd ../nasbench201/
python train_search.py \
    --method blank-proj \
    --dataset $dataset \
    --save $id --gpu $gpu --seed $seed \
    --resume_epoch $resume_epoch --resume_expid $resume_expid --dev proj \
    --edge_decision random --proj_crit acc \
    --proj_intv 5 \
    # --fast --log_tag debug \


## bash blank-proj-201.sh --resume_expid search-blank-201-0