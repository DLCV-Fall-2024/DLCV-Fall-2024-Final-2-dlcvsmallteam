ARRAY=()

# ARRAY[0]: target caption1
# ARRAY[1]: target caption2
# ARRAY[2]: target caption3
# ARRAY[3]: name of the experiment
# ARRAY[4]: config name

for i in "$@"
do 
    echo $i
    ARRAY+=("${i}")
done

python -u  train.py \
        --base configs/custom-diffusion/finetune_style_joint3.yaml  \
        -t --gpus 0,1 \
        --resume-from-checkpoint-custom  ./sd-v1-4.ckpt \
        --caption "<cat2> cat" \
        --datapath "Data/concept_image/cat2" \
        --reg_datapath "real_reg/cat2/images.txt" \
        --reg_caption "real_reg/cat2/caption.txt" \
        --caption2 "<wearable_glasses> wearable glasses" \
        --datapath2 "Data/concept_image/wearable_glasses" \
        --reg_datapath2 "real_reg/wearable_glasses/images.txt" \
        --reg_caption2 "real_reg/wearable_glasses/caption.txt" \
        --caption3 "<watercolor> watercolor" \
        --datapath3 "Data/concept_image/watercolor" \
        --reg_datapath3 "real_reg/watercolor/images.txt" \
        --reg_caption3 "real_reg/watercolor/caption.txt" \
        --modifier_token "<cat2>+<wearable_glasses>+<watercolor>" \
        --name "dlcv-final-custom3-sdv4" \
        --batch_size 2 \
        --logdir "/tmp2/seanfu/logdir"