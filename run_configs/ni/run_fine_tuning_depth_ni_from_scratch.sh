#!/bin/bash

# Example usage:
# sbatch run_fine_tuning_job.sh

#SBATCH --job-name="principled-pre-training"

#SBATCH -N 1                                          # number of minimum nodes
#SBATCH --gres=gpu:2                                  # Request n gpus
#SBATCH --cpus-per-task=32                            # number of cpus per task and per node

#SBATCH -w bruno1,bruno2,dym-lab,dym-lab2,galileo1,galileo2,nlp-a40-1,tdk-bm4,nlp-ada-1

#SBATCH -o fine_tuning_runs/slurm_%N_%j_out.txt      # stdout goes here
#SBATCH -e fine_tuning_runs/slurm_%N_%j_err.txt      # stderr goes here

#SBATCH --mail-type=fail                              # send email if job fails
#SBATCH --mail-user=zachary@campus.technion.ac.il

# To run this code, you must download the NI dataset first:
# git clone https://github.com/allenai/natural-instructions.git data

nvidia-smi
echo "Running from $(pwd)"
export DS_SKIP_CUDA_CHECK=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Load from pre-trained checkpoint
deepspeed \
--no_local_rank \
--master_port=20000 \
--num_gpus=2 \
train_encoder_decoder.py \
num_gpus=2 \
num_cpus=32 \
precision=bf16 \
model.model_implementation=depth \
deepspeed.use_deepspeed=true \
logging.wandb=true \
model.compile=false \
data.data_collator=depth \
dataset.streaming=false \
optim.epochs=3 \
optim.base_lr=1e-4 \
optim.batch_size=64 \
optim.grad_acc=1 \
optim.warmup_steps=0 \
evaluate.every_steps=100 \
logging.every_steps=50 \
checkpoint.every_steps=10_000 \
checkpoint.output_dir=./checkpoints \
checkpoint.checkpoint_path=checkpoints/pre_train/from_scratch/depth/c4_en/lr_0_0001_linear_bsz_200_shuffle_p_0_5/2024-03-11_18-50/checkpoint-8000 \
mode=ft \
downstream.benchmark_constants=ni \
optim.lr_scheduler=constant


