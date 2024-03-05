#!/bin/bash

# Example usage:
# sbatch run_pre_training_job_debug.sh

#SBATCH --job-name="principled-pre-training"

#SBATCH -N 1                          # number of minimum nodes
#SBATCH --gres=gpu:8                  # Request n gpus
#SBATCH --cpus-per-task=64            # number of cpus per task and per node

#SBATCH -w plato2

#SBATCH -o pre_training_runs/slurm_%N_%j_out.txt       # stdout goes here
#SBATCH -e pre_training_runs/slurm_%N_%j_err.txt       # stderr goes here

#SBATCH --mail-type=fail                               # send email if job fails
#SBATCH --mail-user=zachary@campus.technion.ac.il

nvidia-smi
echo "Running from $(pwd)"
echo "Activating virtual environment"
source .depth/bin/activate
export DS_SKIP_CUDA_CHECK=1

################
### T5 model ###
################

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
#accelerate launch \
#--mixed_precision bf16 \
#--num_cpu_threads_per_process 32 \
#--num_processes 4 \
#--main_process_port 29401 \
#train_encoder_decoder.py \
#mode=pt \
#num_gpus=4 \
#num_cpus=32 \
#precision=bf16 \
#model.model_implementation=hf_t5 \
#model.compile=true \
#model.random_init=true \
#dataset.validation_set.num_examples=50_000 \
#data.num_workers=32 \
#data.data_collator=custom_t5 \
#optim.name=adamwscale \
#optim.base_lr=1e-3 \
#optim.batch_size=256 \
#optim.total_steps=200_000 \
#optim.warmup_steps=10_000 \
#optim.lr_scheduler=constant \
#optim.grad_acc=4 \
#evaluate.every_steps=1_000 \
#checkpoint.every_steps=5_000 \
#checkpoint.output_dir=checkpoints \
#checkpoint.save_total_limit=20 \
#logging.every_steps=200 \
#logging.wandb=true \
#deepspeed.use_deepspeed=false

# Resume from checkpoint
#accelerate launch \
#--mixed_precision bf16 \
#--num_cpu_threads_per_process 32 \
#--num_processes 4 \
#--main_process_port 29401 \
#train_encoder_decoder.py \
#mode=pt \
#num_gpus=4 \
#num_cpus=32 \
#precision=bf16 \
#model.model_implementation=hf_t5 \
#model.compile=true \
#model.random_init=true \
#dataset.validation_set.num_examples=50_000 \
#data.num_workers=32 \
#data.data_collator=custom_t5 \
#optim.name=adamwscale \
#optim.base_lr=1e-3 \
#optim.batch_size=256 \
#optim.total_steps=200_000 \
#optim.warmup_steps=10_000 \
#optim.lr_scheduler=constant \
#optim.grad_acc=4 \
#evaluate.every_steps=1_000 \
#checkpoint.every_steps=5_000 \
#checkpoint.output_dir=checkpoints \
#checkpoint.save_total_limit=20 \
#checkpoint.checkpoint_path=pre_train/from_scratch/hf_t5/c4_en/lr_0_001_constant_bsz_256/2024-02-18_17-16 \
#checkpoint.resume=true \
#logging.every_steps=200 \
#logging.wandb=true \
#deepspeed.use_deepspeed=false

deepspeed \
--no_local_rank \
--master_port=12302 \
--num_gpus=8 \
train_encoder_decoder.py \
mode=pt \
num_gpus=8 \
num_cpus=64 \
precision=bf16 \
model.model_implementation=hf_t5 \
model.compile=false \
model.random_init=true \
dataset.validation_set.num_examples=20_000 \
data.num_workers=64 \
data.data_collator=custom_t5 \
optim.name=adamw_torch \
optim.base_lr=1e-3 \
optim.batch_size=256 \
optim.total_steps=100_000 \
optim.warmup_steps=10_000 \
optim.lr_scheduler=linear \
optim.grad_acc=4 \
evaluate.every_steps=1_000 \
checkpoint.every_steps=5_000 \
checkpoint.output_dir=/checkpoints \
checkpoint.save_total_limit=20 \
logging.every_steps=100 \
logging.wandb=true \
deepspeed.use_deepspeed=true