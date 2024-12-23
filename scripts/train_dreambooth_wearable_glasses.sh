export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export INSTANCE_DIR="Data/concept_image/wearable_glasses"
export OUTPUT_DIR="lora-trained-xl/wearable_glasses"
export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

CUDA_VISIBLE_DEVICES=0 accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --pretrained_vae_model_name_or_path=$VAE_PATH \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="fp16" \
  --instance_prompt="a photo of sks glasses" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --checkpointing_steps=20 \
  --checkpoints_total_limit=2 \
  --validation_epochs=1000 \
  --seed="0" \
  --use_8bit_adam \
  # --push_to_hub