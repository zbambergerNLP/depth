#!/bin/bash

# Example usage:
# sbatch run_pre_training_job_debug.sh

#SBATCH --job-name="principled-pre-training"

#SBATCH -N 1                          # number of minimum nodes
#SBATCH --gres=gpu:4                  # Request n gpus
#SBATCH --cpus-per-task=64            # number of cpus per task and per node

#SBATCH -p nlp
#SBATCH -A nlp
#SBATCH -w nlp-ada-1,nlp-ada-2,nlp-a40-1

#SBATCH -o pre_training_runs/slurm_%N_%j_out.txt       # stdout goes here
#SBATCH -e pre_training_runs/slurm_%N_%j_err.txt       # stderr goes here

#SBATCH --mail-type=fail                               # send email if job fails
#SBATCH --mail-user=zachary@campus.technion.ac.il

# Nodes to consider: newton3,newton4,newton5,tdk-bm4,bruno1,bruno2,dym-lab,dym-lab2,galileo1,galileo2

nvidia-smi
wandb login $WANDB_API_KEY
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
#accelerate launch \
#--mixed_precision bf16 \
#--num_cpu_threads_per_process 32 \
#--num_processes 4 \
#--main_process_port 29201 \
#train_encoder_decoder.py \
#mode=pt \
#num_gpus=4 \
#num_cpus=32 \
#precision=bf16 \
#model.model_implementation=depth \
#model.compile=true \
#model.random_init=true \
#dataset.validation_set.num_examples=50_000 \
#data.num_workers=32 \
#data.data_collator=depth \
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

# Resume from local checkpoint
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
#model.model_implementation=depth \
#model.compile=true \
#model.random_init=true \
#dataset.validation_set.num_examples=50_000 \
#data.num_workers=32 \
#data.data_collator=depth \
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
#checkpoint.checkpoint_path=pre_train/from_scratch/depth/c4_en/lr_0_001_constant_bsz_256/2024-02-24_16-35 \
#checkpoint.resume=true \
#logging.every_steps=200 \
#logging.wandb=true \
#deepspeed.use_deepspeed=false


#deepspeed \
#--no_local_rank \
#--master_port=12462 \
#--num_gpus=4 \
#train_encoder_decoder.py \
#mode=pt \
#num_gpus=4 \
#num_cpus=64 \
#precision=bf16 \
#model.model_implementation=depth \
#model.compile=false \
#model.random_init=true \
#dataset.validation_set.num_examples=20_000 \
#data.num_workers=64 \
#data.data_collator=depth \
#data.sentence_shuffling_probability=0.5 \
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
#checkpoint.resume=true \
#checkpoint.save_total_limit=100 \
#logging.every_steps=100 \
#logging.wandb=true \
#logging.wandb=true \
#deepspeed.use_deepspeed=true


# Initialize from huggingface checkpoint
#deepspeed \
#--no_local_rank \
#--master_port=12404 \
#--num_gpus=4 \
#train_encoder_decoder.py \
#mode=pt \
#num_gpus=4 \
#num_cpus=64 \
#precision=bf16 \
#model.model_implementation=depth \
#model.compile=false \
#model.random_init=false \
#dataset.validation_set.num_examples=20_000 \
#data.num_workers=64 \
#data.data_collator=depth \
#data.sentence_shuffling_probability=0.5 \
#data.mlm_probability=0.3 \
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
#checkpoint.checkpoint_path=checkpoints/pre_train/from_scratch/depth/c4_en/lr_0_0001_linear_bsz_200_shuffle_p_0_5/2024-03-11_18-50 \
#checkpoint.resume=true \
#checkpoint.save_total_limit=100 \
#logging.every_steps=100 \
#logging.wandb=true \
#deepspeed.use_deepspeed=true

# Inverse square root optimizer
#deepspeed \
#--no_local_rank \
#--master_port=12469 \
#--num_gpus=4 \
#train_encoder_decoder.py \
#mode=pt \
#num_gpus=4 \
#num_cpus=64 \
#precision=bf16 \
#model.model_implementation=depth \
#model.compile=false \
#model.random_init=true \
#dataset.validation_set.num_examples=20_000 \
#data.num_workers=64 \
#data.data_collator=depth \
#data.sentence_shuffling_probability=0.5 \
#optim.name=adamw_torch \
#optim.base_lr=1e-2 \
#optim.batch_size=512 \
#optim.total_steps=1_000_000 \
#optim.warmup_steps=10_000 \
#optim.lr_scheduler=inverse_sqrt \
#optim.grad_acc=8 \
#evaluate.every_steps=1_000 \
#checkpoint.every_steps=2_000 \
#checkpoint.output_dir=checkpoints \
#checkpoint.resume=true \
#checkpoint.save_total_limit=100 \
#logging.every_steps=100 \
#logging.wandb=true \
#logging.wandb=true \
#deepspeed.use_deepspeed=true

# Equal weights between sentence and reconstruction losses
#deepspeed \
#--no_local_rank \
#--master_port=12411 \
#--num_gpus=4 \
#train_encoder_decoder.py \
#mode=pt \
#num_gpus=4 \
#num_cpus=64 \
#precision=bf16 \
#model.model_implementation=depth \
#model.compile=false \
#model.random_init=false \
#dataset.validation_set.num_examples=20_000 \
#data.num_workers=64 \
#data.data_collator=depth \
#data.sentence_shuffling_probability=0.5 \
#data.mlm_probability=0.3 \
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
#checkpoint.checkpoint_path=checkpoints/pre_train/from_scratch/depth/allenai_c4_en/lr_0_0001_linear_bsz_200_shuffle_p_0_5/2024-04-15_18-57 \
#checkpoint.resume=true \
#checkpoint.save_total_limit=100 \
#logging.every_steps=100 \
#logging.wandb=true \
#deepspeed.use_deepspeed=true


# Sentence loss weighted 5x more than reconstruction loss
deepspeed \
--no_local_rank \
--master_port=12412 \
--num_gpus=4 \
train_encoder_decoder.py \
mode=pt \
num_gpus=4 \
num_cpus=64 \
precision=bf16 \
model.model_implementation=depth \
model.compile=false \
model.random_init=true \
dataset.validation_set.num_examples=20_000 \
data.num_workers=64 \
data.data_collator=depth \
data.sentence_shuffling_probability=0.5 \
data.mlm_probability=0.3 \
data.sentence_loss_coefficient=5.0 \
optim.name=adamw_torch \
optim.base_lr=1e-4 \
optim.batch_size=200 \
optim.total_steps=1_000_000 \
optim.warmup_steps=10_000 \
optim.lr_scheduler=linear \
optim.grad_acc=2 \
evaluate.every_steps=1_000 \
checkpoint.every_steps=2_000 \
checkpoint.output_dir=checkpoints \
checkpoint.save_total_limit=100 \
logging.every_steps=100 \
logging.wandb=true \
deepspeed.use_deepspeed=true
