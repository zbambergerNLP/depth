#!/bin/bash

# Example usage:
# sbatch run_fine_tuning_job.sh

#SBATCH --job-name="ni_principled-pre-training"

#SBATCH -N 1                                          # number of minimum nodes
#SBATCH --gres=gpu:2                                  # Request n gpus
#SBATCH --cpus-per-task=32                            # number of cpus per task and per node

#SBATCH -w bruno1,bruno2,dym-lab,dym-lab2,galileo1,galileo2,newton3,newton4,newton5,nlp-a40-1,nlp-ada-1,nlp-ada-2,tdk-bm4

#SBATCH -o fine_tuning_runs/slurm_%N_%j_out.txt      # stdout goes here
#SBATCH -e fine_tuning_runs/slurm_%N_%j_err.txt      # stderr goes here

#SBATCH --mail-type=fail                              # send email if job fails
#SBATCH --mail-user=ofek.glick@campus.technion.ac.il

nvidia-smi
echo "Running from $(pwd)"
export DS_SKIP_CUDA_CHECK=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Load from pre-trained checkpoint
deepspeed \
--no_local_rank \
--master_port=12347 \
--num_gpus=2 \
train_encoder_decoder.py \
num_gpus=2 \
num_cpus=32 \
precision=bf16 \
model.model_implementation=hf_t5 \
deepspeed.use_deepspeed=true \
logging.wandb=true \
model.compile=false \
data.data_collator=custom_t5 \
dataset.streaming=false \
optim.total_steps=10_000 \
optim.base_lr=5e-5 \
optim.batch_size=8 \
optim.grad_acc=1 \
optim.warmup_steps=0 \
evaluate.every_steps=200 \
logging.every_steps=50 \
checkpoint.every_steps=400 \
mode=ft \
downstream.benchmark_constants=ni \
optim.lr_scheduler=constant


