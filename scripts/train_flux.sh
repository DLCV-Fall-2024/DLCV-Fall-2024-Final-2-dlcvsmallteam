export MODEL_NAME="black-forest-labs/FLUX.1-dev"
export INSTANCE_DIR="Data/concept_image/cat2"
export OUTPUT_DIR="flux-lora-trained-xl/cat2"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

CUDA_VISIBLE_DEVICES=0 accelerate launch train_dreambooth_flux.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="bf16" \
  --instance_prompt="a photo of sks cat" \
  --resolution=512 \
  --train_batch_size=1 \
  --guidance_scale=1 \
  --gradient_accumulation_steps=4 \
  --optimizer="prodigy" \
  --learning_rate=1. \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --checkpointing_steps=20 \
  --checkpoints_total_limit=2 \
  --validation_epochs=1000 \
  --num_validation_images=1 \
  --seed="0" \
  --use_8bit_adam \
  # --push_to_hub