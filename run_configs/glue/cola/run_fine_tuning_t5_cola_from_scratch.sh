#!/bin/bash

# Example usage:
# sbatch run_fine_tuning_job.sh

#SBATCH --job-name="principled-pre-training"

#SBATCH -N 1                                          # number of minimum nodes
#SBATCH --gres=gpu:2                                  # Request n gpus
#SBATCH --cpus-per-task=32                            # number of cpus per task and per node

#SBATCH -w bruno1,bruno2,newton3,newton4,newton5,dym-lab,dym-lab2


#SBATCH -o fine_tuning_runs/slurm_%N_%j_out.txt      # stdout goes here
#SBATCH -e fine_tuning_runs/slurm_%N_%j_err.txt      # stderr goes here

#SBATCH --mail-type=fail                              # send email if job fails
#SBATCH --mail-user=zachary@campus.technion.ac.il

nvidia-smi
echo "Running from $(pwd)"
echo "Activating virtual environment"
source .depth/bin/activate
export DS_SKIP_CUDA_CHECK=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK


# Run fine tuning on Cola with T5, half precision
# 11k examples

# Use a from-scratch pre-trained T5 model


# Use a from-scratch pre-trained T5 model
deepspeed \
--no_local_rank \
--master_port=11502 \
--num_gpus=2 \
train_encoder_decoder.py \
mode=ft \
num_gpus=2 \
num_cpus=32 \
precision=bf16 \
model.model_implementation=hf_t5 \
model.compile=false \
data.input_length=512 \
data.target_length=16 \
data.num_workers=32 \
data.data_collator=custom_t5 \
downstream.benchmark_constants=glue \
downstream.benchmark_dataset=cola \
dataset.streaming=false \
optim.name=adamw_torch \
optim.base_lr=1e-5 \
optim.batch_size=16 \
optim.total_steps=3_000 \
optim.warmup_steps=300 \
optim.grad_acc=1 \
optim.lr_scheduler=constant_with_warmup \
checkpoint.checkpoint_path=checkpoints/pre_train/from_scratch/hf_t5/c4_en/lr_0_0001_inverse_sqrt_bsz_200_shuffle_p_0_5/2024-03-18_21-25/checkpoint-1000000 \
checkpoint.resume=false \
checkpoint.every_steps=4_000 \
checkpoint.save_total_limit=3 \
logging.every_steps=10 \
logging.wandb=true \
evaluate.every_steps=100


# Baseline
#deepspeed \
#--no_local_rank \
#--master_port=11202 \
#--num_gpus=2 \
#train_encoder_decoder.py \
#mode=ft \
#num_gpus=2 \
#num_cpus=32 \
#precision=bf16 \
#model.model_implementation=hf_t5 \
#model.compile=false \
#data.input_length=512 \
#data.target_length=16 \
#data.num_workers=32 \
#data.data_collator=custom_t5 \
#downstream.benchmark_constants=glue \
#downstream.benchmark_dataset=cola \
#dataset.streaming=false \
#optim.name=adamw_torch \
#optim.base_lr=1e-5 \
#optim.batch_size=16 \
#optim.total_steps=3_000 \
#optim.warmup_steps=100 \
#optim.grad_acc=1 \
#optim.lr_scheduler=linear \
#checkpoint.resume=false \
#checkpoint.every_steps=4_000 \
#checkpoint.save_total_limit=3 \
#logging.every_steps=10 \
#logging.wandb=true \
#evaluate.every_steps=100
