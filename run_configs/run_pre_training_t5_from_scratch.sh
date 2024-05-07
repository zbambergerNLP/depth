#!/bin/bash

# Example usage:
# sbatch run_pre_training_job_debug.sh

#SBATCH --job-name="principled-pre-training"

#SBATCH -N 1                          # number of minimum nodes
#SBATCH --gres=gpu:4                  # Request n gpus
#SBATCH --cpus-per-task=64            # number of cpus per task and per node

#SBATCH -A nlp
#SBATCH -p nlp
#SBATCH -w nlp-ada-2,nlp-a40-1

#SBATCH -o pre_training_runs/slurm_%N_%j_out.txt       # stdout goes here
#SBATCH -e pre_training_runs/slurm_%N_%j_err.txt       # stderr goes here

#SBATCH --mail-type=fail                               # send email if job fails
#SBATCH --mail-user=zachary@campus.technion.ac.il


# GPU nodes to consider using: newton3,newton4,newton5,tdk-bm4,bruno1,bruno2,dym-lab,dym-lab2

nvidia-smi
echo "Running from $(pwd)"
echo "Activating virtual environment"
source .depth/bin/activate
wandb login $WANDB_API_KEY
export DS_SKIP_CUDA_CHECK=1

################
### T5 model ###
################

#deepspeed \
#--no_local_rank \
#--master_port=42305 \
#--num_gpus=4 \
#train_encoder_decoder.py \
#mode=pt \
#num_gpus=4 \
#num_cpus=32 \
#precision=bf16 \
#model.model_implementation=hf_t5 \
#model.compile=false \
#model.random_init=true \
#dataset.validation_set.num_examples=20_000 \
#data.num_workers=32 \
#data.data_collator=custom_t5 \
#optim.name=adamw_torch \
#optim.base_lr=1e-4 \
#optim.batch_size=200 \
#optim.total_steps=1_000_000 \
#optim.warmup_steps=10_000 \
#optim.lr_scheduler=linear \
#optim.grad_acc=2 \
#evaluate.every_steps=1_000 \
#checkpoint.every_steps=2_000 \
#checkpoint.output_dir=checkpoints \
#checkpoint.save_total_limit=100 \
#logging.every_steps=100 \
#logging.wandb=true \
#deepspeed.use_deepspeed=true

# linear decay
#deepspeed \
#--no_local_rank \
#--master_port=42302 \
#--num_gpus=4 \
#train_encoder_decoder.py \
#mode=pt \
#num_gpus=4 \
#num_cpus=64 \
#precision=bf16 \
#model.model_implementation=hf_t5 \
#model.compile=false \
#model.random_init=false \
#dataset.validation_set.num_examples=20_000 \
#data.num_workers=64 \
#data.data_collator=custom_t5 \
#optim.name=adamw_torch \
#optim.base_lr=1e-4 \
#optim.batch_size=200 \
#optim.total_steps=1_000_000 \
#optim.warmup_steps=10_000 \
#optim.lr_scheduler=linear \
#optim.grad_acc=2 \
#evaluate.every_steps=1_000 \
#checkpoint.every_steps=2_000 \
#checkpoint.checkpoint_path=checkpoints/pre_train/from_scratch/hf_t5/c4_en/lr_0_0001_linear_bsz_200_shuffle_p_0_5/2024-03-12_13-26 \
#checkpoint.resume=true \
#checkpoint.save_total_limit=100 \
#logging.every_steps=100 \
#logging.wandb=true \
#deepspeed.use_deepspeed=true

# Inverse square root
deepspeed \
--no_local_rank \
--master_port=42311 \
--num_gpus=4 \
train_encoder_decoder.py \
mode=pt \
num_gpus=4 \
num_cpus=64 \
precision=bf16 \
model.model_implementation=hf_t5 \
model.compile=false \
model.random_init=true \
dataset.validation_set.num_examples=20_000 \
data.num_workers=64 \
data.data_collator=custom_t5 \
optim.name=adamw_torch \
optim.base_lr=1e-4 \
optim.batch_size=200 \
optim.total_steps=1_000_000 \
optim.warmup_steps=10_000 \
optim.lr_scheduler=inverse_sqrt \
optim.grad_acc=2 \
evaluate.every_steps=1_000 \
checkpoint.every_steps=10_000 \
checkpoint.checkpoint_path=checkpoints/pre_train/from_scratch/hf_t5/c4_en/lr_0_0001_inverse_sqrt_bsz_200_shuffle_p_0_5/2024-03-18_21-25 \
checkpoint.resume=true \
checkpoint.save_total_limit=100 \
logging.every_steps=100 \
logging.wandb=true \
deepspeed.use_deepspeed=true

# Cosine decay
#deepspeed \
#--no_local_rank \
#--master_port=42311 \
#--num_gpus=4 \
#train_encoder_decoder.py \
#mode=pt \
#num_gpus=4 \
#num_cpus=64 \
#precision=bf16 \
#model.model_implementation=hf_t5 \
#model.compile=false \
#model.random_init=true \
#dataset.validation_set.num_examples=20_000 \
#data.num_workers=64 \
#data.data_collator=custom_t5 \
#optim.name=adamw_torch \
#optim.base_lr=1e-4 \
#optim.batch_size=200 \
#optim.total_steps=1_000_000 \
#optim.warmup_steps=10_000 \
#optim.lr_scheduler=cosine \
#optim.grad_acc=2 \
#evaluate.every_steps=1_000 \
#checkpoint.every_steps=2_000 \
#checkpoint.checkpoint_path=checkpoints/pre_train/from_scratch/hf_t5/c4_en/lr_0_0001_cosine_bsz_200_shuffle_p_0_5/2024-03-18_21-56 \
#checkpoint.resume=true \
#checkpoint.save_total_limit=100 \
#logging.every_steps=100 \
#logging.wandb=true \
#deepspeed.use_deepspeed=true

# Inverse square root learning rate scheduler
#deepspeed \
#--no_local_rank \
#--master_port=42315 \
#--num_gpus=4 \
#train_encoder_decoder.py \
#mode=pt \
#num_gpus=4 \
#num_cpus=64 \
#precision=bf16 \
#model.model_implementation=hf_t5 \
#model.compile=false \
#model.random_init=true \
#dataset.validation_set.num_examples=20_000 \
#data.num_workers=64 \
#data.data_collator=custom_t5 \
#data.mlm_probability=0.15 \
#optim.name=adafactor \
#optim.base_lr=1e-3 \
#optim.batch_size=200 \
#optim.total_steps=1_000_000 \
#optim.warmup_steps=10_000 \
#optim.lr_scheduler=inverse_sqrt \
#optim.grad_acc=2 \
#evaluate.every_steps=1_000 \
#checkpoint.every_steps=2_000 \
#checkpoint.save_total_limit=100 \
#logging.every_steps=100 \
#logging.wandb=true \
#deepspeed.use_deepspeed=true


