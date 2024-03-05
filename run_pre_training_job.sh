#!/bin/bash

# Example usage:
# sbatch run_pre_training_job_debug.sh

#SBATCH --job-name="principled-pre-training"

#SBATCH -N 1                          # number of minimum nodes
#SBATCH --gres=gpu:A40:4              # Request n gpus
#SBATCH --cpus-per-task=32            # number of cpus per task and per node

#SBATCH -o pre_training_runs/slurm_%N_%j_out.txt       # stdout goes here
#SBATCH -e pre_training_runs/slurm_%N_%j_err.txt       # stderr goes here

#SBATCH --mail-type=fail                               # send email if job fails
#SBATCH --mail-user=zachary@campus.technion.ac.il

nvidia-smi

source .depth/bin/activate
export DS_SKIP_CUDA_CHECK=1

################
### T5 model ###
################

# Large-scale pre-training with deepspeed
#deepspeed \
#--no_local_rank \
#--master_port=12345 \
#--num_gpus=4 \
#pre_train_encoder_decoder.py \
#num_gpus=4 \
#num_cpus=32 \
#model.model_implementation=hf_t5 \
#model.random_init=false \
#logging.wandb=true \
#logging.every_steps=100 \
#dataset.validation_set.num_examples=10_000 \
#evaluate.every_steps=5_000 \
#checkpoint.every_steps=10_000 \
#data.data_collator=custom_t5 \
#data.num_workers=32 \
#optim.total_steps=100_000 \
#optim.lr_scheduler=constant \
#optim.name=adamw_torch \
#optim.base_lr=1e-4 \
#optim.batch_size=128 \
#optim.grad_acc=2

# Large-scale pre-training with accelerate
#accelerate launch \
#--mixed_precision bf16 \
#--num_cpu_threads_per_process 32 \
#--num_processes 4 \
#train_encoder_decoder.py \
#num_gpus=4 \
#num_cpus=32 \
#deepspeed.use_deepspeed=false \
#model.model_implementation=hf_t5 \
#model.compile=true \
#model.random_init=true \
#logging.wandb=true \
#dataset.validation_set.num_examples=20_000 \
#logging.every_steps=100 \
#evaluate.every_steps=1_000 \
#checkpoint.every_steps=10_000 \
#checkpoint.output_dir=checkpoints/t5/continuous_pre_training \
#data.data_collator=custom_t5 \
#data.num_workers=32 \
#optim.total_steps=100_000 \
#optim.lr_scheduler=constant \
#optim.name=adamwscale \
#optim.base_lr=1e-4 \
#optim.batch_size=256 \
#optim.grad_acc=4

# Debug with deepspeed:
#deepspeed \
#--no_local_rank \
#--master_port=12345 \
#--num_gpus=2 \
#pre_train_encoder_decoder.py \
#num_gpus=2 \
#num_cpus=8 \
#model.model_implementation=hf_t5 \
#model.compile=false \
#deepspeed.use_deepspeed=false \
#dataset.validation_set.num_examples=500 \
#logging.every_steps=20 \
#evaluate.every_steps=20 \
#checkpoint.every_steps=60 \
#data.data_collator=custom_t5 \
#data.num_workers=4 \
#optim.total_steps=50_000 \
#optim.lr_scheduler=constant \
#optim.name=adamw_torch \
#optim.base_lr=1e-4 \
#optim.batch_size=32 \
#optim.grad_acc=1

# Debug with accelerate
#accelerate launch \
#--mixed_precision bf16 \
#--num_cpu_threads_per_process 32 \
#--num_processes 4 \
#train_encoder_decoder.py \
#num_gpus=4 \
#num_cpus=32 \
#model.model_implementation=hf_t5 \
#model.compile=false \
#logging.wandb=true \
#deepspeed.use_deepspeed=false \
#dataset.validation_set.num_examples=500 \
#logging.every_steps=20 \
#evaluate.every_steps=20 \
#checkpoint.every_steps=60 \
#data.data_collator=custom_t5 \
#data.num_workers=32 \
#optim.total_steps=100_000 \
#optim.lr_scheduler=constant \
#optim.name=adamwscale \
#optim.base_lr=1e-4 \
#optim.batch_size=128 \
#optim.grad_acc=2


###################
### DEPTH model ###
###################

# Large-scale pre-training
#deepspeed \
#--no_local_rank \
#--master_port=12353 \
#--num_gpus=4 \
#train_encoder_decoder.py \
#num_gpus=4 \
#num_cpus=32 \
#model.model_implementation=depth \
#logging.wandb=true \
#model.compile=true \
#model.random_init=false \
#logging.every_steps=100 \
#dataset.validation_set.num_examples=20_000 \
#evaluate.every_steps=1_000 \
#checkpoint.every_steps=10_000 \
#checkpoint.output_dir=checkpoints/depth/continuous_pre_training \
#data.data_collator=depth \
#data.num_workers=32 \
#optim.total_steps=50_000 \
#optim.name=adamw_torch \
#optim.warmup_steps=5_000 \
#optim.base_lr=1e-3 \
#optim.batch_size=256 \
#optim.grad_acc=4

# Large scale pre training with accelerate
accelerate launch \
--mixed_precision bf16 \
--num_cpu_threads_per_process 32 \
--num_processes 4 \
train_encoder_decoder.py \
num_gpus=4 \
num_cpus=32 \
deepspeed.use_deepspeed=false \
model.model_implementation=depth \
model.compile=true \
model.random_init=true \
logging.wandb=true \
dataset.validation_set.num_examples=20_000 \
logging.every_steps=100 \
evaluate.every_steps=1_000 \
checkpoint.every_steps=10_000 \
checkpoint.output_dir=checkpoints/depth/from_scratch \
data.data_collator=depth \
data.num_workers=32 \
optim.total_steps=100_000 \
optim.lr_scheduler=constant \
optim.name=adamwscale \
optim.base_lr=1e-3 \
optim.batch_size=256 \
optim.grad_acc=4


## Debug with deepspeed:
#deepspeed \
#--no_local_rank \
#--master_port=12345 \
#--num_gpus=4 \
#train_encoder_decoder.py \
#num_gpus=4 \
#num_cpus=32 \
#deepspeed.use_deepspeed=true \
#model.compile=true \
#model.random_init=false \
#model.model_implementation=depth \
#dataset.validation_set.num_examples=20_000 \
#logging.every_steps=10 \
#evaluate.every_steps=100 \
#checkpoint.every_steps=1000 \
#data.data_collator=depth \
#data.num_workers=32 \
#optim.name=adamw_torch \
#optim.total_steps=100_000 \
#optim.lr_scheduler=constant \
#optim.base_lr=1e-4 \
#optim.batch_size=128 \
#optim.grad_acc=2

# Debug with accelerate
#accelerate launch \
#--mixed_precision bf16 \
#--num_cpu_threads_per_process 32 \
#--num_processes 4 \
#train_encoder_decoder.py \
#num_gpus=4 \
#num_cpus=64 \
#model.model_implementation=depth \
#model.compile=false \
#model.random_init=false \
#deepspeed.use_deepspeed=false \
#dataset.validation_set.num_examples=20_000 \
#logging.wandb=true \
#logging.every_steps=10 \
#evaluate.every_steps=50 \
#checkpoint.every_steps=100 \
#data.data_collator=depth \
#data.num_workers=32 \
#optim.total_steps=50_000 \
#optim.lr_scheduler=constant \
#optim.name=adamwscale \
#optim.base_lr=1e-4 \
#optim.batch_size=8 \
#optim.grad_acc=1
