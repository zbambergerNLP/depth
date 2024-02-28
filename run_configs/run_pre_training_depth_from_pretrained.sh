#!/bin/bash

# Example usage:
# sbatch run_pre_training_job_debug.sh

#SBATCH --job-name="principled-pre-training"

#SBATCH -N 1                          # number of minimum nodes
#SBATCH --gres=gpu:4                  # Request n gpus
#SBATCH --cpus-per-task=32            # number of cpus per task and per node

#SBATCH -A nlp
#SBATCH -p nlp
#SBATCH -w nlp-ada-[1-2]

#SBATCH -o pre_training_runs/slurm_%N_%j_out.txt       # stdout goes here
#SBATCH -e pre_training_runs/slurm_%N_%j_err.txt       # stderr goes here

#SBATCH --mail-type=fail                               # send email if job fails
#SBATCH --mail-user=zachary@campus.technion.ac.il

nvidia-smi
echo "Running from $(pwd)"
echo "Activating virtual environment"
source .depth/bin/activate
export DS_SKIP_CUDA_CHECK=1

###################
### DEPTH model ###
###################

# To run with deepspeed instead of accelerate, replace `accelerate launch` with `deepspeed` as follows:
# deepspeed \
# --no_local_rank \
# --master_port=12345 \
# --num_gpus=4 \
# train_encoder_decoder.py \
# ...
#
# Also make sure to set the flag `deepspeed.use_deepspeed` to `true`, and the optimizer to `adamw_torch` in the command
# line arguments

# Large-scale pre-training with accelerate
# Resume from HuggingFace T5 checkpoint
#accelerate launch \
#--mixed_precision bf16 \
#--num_cpu_threads_per_process 32 \
#--num_processes 4 \
#--main_process_port 29101 \
#train_encoder_decoder.py \
#mode=pt \
#num_gpus=4 \
#num_cpus=32 \
#precision=bf16 \
#model.model_implementation=depth \
#model.compile=true \
#model.random_init=false \
#dataset.validation_set.num_examples=50_000 \
#data.num_workers=32 \
#data.data_collator=depth \
#optim.name=adamwscale \
#optim.base_lr=1e-3 \
#optim.batch_size=256 \
#optim.total_steps=200_000 \
#optim.warmup_steps=10_000 \
#optim.lr_scheduler=constant_with_warmup \
#optim.grad_acc=4 \
#evaluate.every_steps=1_000 \
#checkpoint.every_steps=5_000 \
#checkpoint.output_dir=checkpoints \
#checkpoint.save_total_limit=20 \
#logging.every_steps=200 \
#logging.wandb=true \
#deepspeed.use_deepspeed=false

accelerate launch \
--mixed_precision bf16 \
--num_cpu_threads_per_process 32 \
--num_processes 4 \
--main_process_port 29102 \
train_encoder_decoder.py \
mode=pt \
num_gpus=4 \
num_cpus=32 \
precision=bf16 \
model.model_implementation=depth \
model.compile=true \
model.random_init=false \
dataset.validation_set.num_examples=50_000 \
data.num_workers=32 \
data.data_collator=depth \
optim.name=adamwscale \
optim.base_lr=1e-3 \
optim.batch_size=256 \
optim.total_steps=200_000 \
optim.warmup_steps=10_000 \
optim.lr_scheduler=constant_with_warmup \
optim.grad_acc=4 \
evaluate.every_steps=1_000 \
checkpoint.every_steps=5_000 \
checkpoint.output_dir=checkpoints \
checkpoint.checkpoint_path=pre_train/from_pretrained/depth/c4_en/lr_0_001_constant_with_warmup_bsz_256/2024-02-26_17-19 \
checkpoint.save_total_limit=20 \
checkpoint.resume=true \
logging.every_steps=200 \
logging.wandb=true \
deepspeed.use_deepspeed=false
