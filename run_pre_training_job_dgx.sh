#!/bin/bash

# Example usage:
# sbatch run_pre_training_job_dgx.sh

#SBATCH --job-name="principled-pre-training"

#SBATCH -N 1                          # number of minimum nodes
#SBATCH --gres=gpu:4                  # Request n gpus
#SBATCH --cpus-per-task=32            # number of cpus per task and per node
#SBATCH --qos=normal                  # priority

#SBATCH -o pre_training_runs/slurm_%N_%j_out.txt       # stdout goes here
#SBATCH -e pre_training_runs/slurm_%N_%j_err.txt       # stderr goes here

#SBATCH --mail-type=fail                               # send email if job fails
#SBATCH --mail-user=zachary@campus.technion.ac.il

nvidia-smi

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

srun \
--gres=gpu:4 \
--cpus_per_task 32 \
--qos=normal \
--container-image nvcr.io#nvidia/pytorch:23.12-py3 \
--container-mounts=/home/zachary/depth:/home/zachary/depth \
--container-workdir=/home/zachary/depth \
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
checkpoint.output_dir=checkpoints/depth/continuous_pre_training \
data.data_collator=depth \
data.num_workers=32 \
optim.total_steps=100_000 \
optim.lr_scheduler=constant \
optim.name=adamwscale \
optim.base_lr=1e-4 \
optim.batch_size=256 \
optim.grad_acc=4

srun \
--gres=gpu:4 \
--cpus_per_task 32 \
--qos=normal \
--container-image nvcr.io#nvidia/pytorch:23.12-py3 \
--container-mounts=/home/zachary/depth:/home/zachary/depth \
--container-workdir=/home/zachary/depth \
--pty bash
