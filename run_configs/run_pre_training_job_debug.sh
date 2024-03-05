#!/bin/bash

# Example usage:
# sbatch run_pre_training_job_debug.sh

#SBATCH --job-name="principled-pre-training"

#SBATCH -N 1                          # number of minimum nodes
#SBATCH --gres=gpu:A40:2              # Request n gpus
#SBATCH --cpus-per-task=32            # number of cpus per task and per node

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

# Debug with deepspeed:
deepspeed \
--no_local_rank \
--master_port=54321 \
--num_gpus=2 \
train_encoder_decoder.py \
num_gpus=2 \
num_cpus=8 \
model.model_implementation=hf_t5 \
model.compile=false \
deepspeed.use_deepspeed=false \
dataset.validation_set.num_examples=500 \
logging.wandb=false \
logging.every_steps=10 \
evaluate.every_steps=20 \
checkpoint.every_steps=100 \
data.data_collator=custom_t5 \
data.num_workers=4 \
optim.total_steps=50_000 \
optim.lr_scheduler=constant \
optim.name=adamw_torch \
optim.base_lr=1e-4 \
optim.batch_size=32 \
optim.grad_acc=1

# Debug with accelerate
accelerate launch \
--mixed_precision bf16 \
--num_cpu_threads_per_process 32 \
--num_processes 2 \
train_encoder_decoder.py \
num_gpus=2 \
num_cpus=32 \
model.model_implementation=hf_t5 \
model.compile=false \
logging.wandb=false \
deepspeed.use_deepspeed=false \
dataset.validation_set.num_examples=500 \
logging.every_steps=10 \
evaluate.every_steps=20 \
checkpoint.every_steps=100 \
data.data_collator=custom_t5 \
data.num_workers=32 \
optim.total_steps=100_000 \
optim.lr_scheduler=constant \
optim.name=adamwscale \
optim.base_lr=1e-4 \
optim.batch_size=32 \
optim.grad_acc=1


###################
### DEPTH model ###
###################

# Debug with deepspeed:
deepspeed \
--no_local_rank \
--master_port=54322 \
--num_gpus=2 \
train_encoder_decoder.py \
num_gpus=2 \
num_cpus=32 \
deepspeed.use_deepspeed=true \
model.compile=true \
model.random_init=false \
model.model_implementation=depth \
dataset.validation_set.num_examples=500 \
logging.wandb=false \
logging.every_steps=10 \
evaluate.every_steps=20 \
checkpoint.every_steps=100 \
data.data_collator=depth \
data.num_workers=32 \
optim.name=adamw_torch \
optim.total_steps=100_000 \
optim.lr_scheduler=constant \
optim.base_lr=1e-4 \
optim.batch_size=32 \
optim.grad_acc=1

# Debug with accelerate on two A40 or L40 GPUs
accelerate launch \
--mixed_precision bf16 \
--num_cpu_threads_per_process 32 \
--num_processes 4 \
train_encoder_decoder.py \
num_gpus=4 \
num_cpus=32 \
model.model_implementation=depth \
model.compile=false \
model.random_init=false \
deepspeed.use_deepspeed=false \
dataset.validation_set.num_examples=500 \
logging.wandb=false \
logging.every_steps=10 \
evaluate.every_steps=20 \
checkpoint.every_steps=100 \
data.data_collator=depth \
data.num_workers=32 \
optim.total_steps=50_000 \
optim.lr_scheduler=constant \
optim.name=adamwscale \
optim.base_lr=1e-4 \
optim.batch_size=240 \
optim.grad_acc=2

# Small scale experiment (e.g., on 4 small GPUs with 16GB memory each)
accelerate launch \
--mixed_precision bf16 \
--num_cpu_threads_per_process 32 \
--num_processes 4 \
train_encoder_decoder.py \
num_gpus=4 \
num_cpus=32 \
model.model_implementation=depth \
model.compile=false \
model.random_init=false \
deepspeed.use_deepspeed=false \
dataset.validation_set.num_examples=500 \
logging.wandb=false \
logging.every_steps=10 \
evaluate.every_steps=20 \
checkpoint.every_steps=100 \
data.data_collator=depth \
data.num_workers=32 \
optim.total_steps=50_000 \
optim.lr_scheduler=constant \
optim.name=adamwscale \
optim.base_lr=1e-4 \
optim.batch_size=16 \
optim.grad_acc=1