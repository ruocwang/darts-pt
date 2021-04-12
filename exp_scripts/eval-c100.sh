#!/bin/bash
script_name=`basename "$0"`
id=${script_name%.*}
dataset=${dataset:-cifar100}
seed=${seed:-2}
gpu=${gpu:-"auto"}
arch=${arch:-"none"}
batch_size=${batch_size:-96}
learning_rate=${learning_rate:-0.025}
resume_expid=${resume_expid:-'none'}
resume_epoch=${resume_epoch:-0}

while [ $# -gt 0 ]; do
    if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
        # echo $1 $2 // Optional to see the parameter:value result
    fi
    shift
done

echo 'id:' $id
echo 'seed:' $seed
echo 'dataset:' $dataset
echo 'gpu:' $gpu
echo 'arch:' $arch
echo 'batch_size:' $batch_size
echo 'learning_rate:' $learning_rate


cd ../sota/cnn
python train.py \
    --arch $arch \
    --dataset $dataset \
    --auxiliary --cutout \
    --seed $seed --save $id --gpu $gpu \
    --batch_size $batch_size --learning_rate $learning_rate \
    --resume_expid $resume_expid --resume_epoch $resume_epoch \
    --init_channels 16 --layers 8 \