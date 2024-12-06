#!/usr/bin/env bash
#### command to run with retrieved images as regularization

# ARRAY[0]: target caption1
# ARRAY[1]: path to target images1
# ARRAY[2]: path where retrieved images1 are saved
# ARRAY[3]: target caption2
# ARRAY[4]: path to target images2
# ARRAY[5]: path where retrieved images2 are saved
# ARRAY[6]: target caption3
# ARRAY[7]: path to target images3
# ARRAY[8]: path where retrieved images3 are saved
# ARRAY[9]: name of the experiment
# ARRAY[10]: config name
# ARRAY[11]: pretrained model path

ARRAY=()

for i in "$@"
do 
    echo $i
    ARRAY+=("${i}")
done


# python src/retrieve_new.py --target_name "${ARRAY[0]}" --outpath ${ARRAY[2]}
# python src/retrieve_new.py --target_name "${ARRAY[3]}" --outpath ${ARRAY[5]}
# python src/retrieve_new.py --target_name "${ARRAY[6]}" --outpath ${ARRAY[8]}


python -u  train.py \
        --base configs/custom-diffusion/${ARRAY[10]}  \
        -t --gpus 0,1 \
        --resume-from-checkpoint-custom  ${ARRAY[11]} \
        --caption "<new1> ${ARRAY[0]}" \
        --datapath ${ARRAY[1]} \
        --reg_datapath "${ARRAY[2]}/images.txt" \
        --reg_caption "${ARRAY[2]}/caption.txt" \
        --caption2 "<new2> ${ARRAY[3]}" \
        --datapath2 ${ARRAY[4]} \
        --reg_datapath2 "${ARRAY[5]}/images.txt" \
        --reg_caption2 "${ARRAY[5]}/caption.txt" \
        --caption3 "<new3> ${ARRAY[6]}" \
        --datapath3 ${ARRAY[7]} \
        --reg_datapath3 "${ARRAY[8]}/images.txt" \
        --reg_caption3 "${ARRAY[8]}/caption.txt" \
        --modifier_token "<new1>+<new2>" \
        --name "${ARRAY[9]}-sdv4" \
        --batch_size 2 \
        --logdir "/tmp2/seanfu/logdir"
