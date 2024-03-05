#!/bin/bash

# Example usage:
# sbatch run_fine_tuning_job.sh

#SBATCH --job-name="principled-pre-training"

#SBATCH -N 1                                          # number of minimum nodes
#SBATCH --gres=gpu:2                                  # Request n gpus
#SBATCH --cpus-per-task=32                            # number of cpus per task and per node

#SBATCH -A nlp
#SBATCH -p nlp
#SBATCH -w nlp-ada-[1,2]

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

# 70k examples
# Run fine tuning on SST-2 with Depth
# Sequence length: 268 (pad to multiple of 8, so 272)

deepspeed \
--no_local_rank \
--master_port=12100 \
--num_gpus=4 \
train_encoder_decoder.py \
num_gpus=4 \
num_cpus=32 \
data.input_length=272 \
data.target_length=8 \
dataset.streaming=false \
precision=bf16 \
model.model_implementation=depth \
deepspeed.use_deepspeed=true \
logging.wandb=true \
model.compile=false \
data.data_collator=depth \
optim.total_steps=2_000 \
optim.base_lr=1e-4 \
optim.batch_size=128 \
optim.grad_acc=1 \
optim.warmup_steps=500 \
evaluate.every_steps=100 \
logging.every_steps=10 \
checkpoint.every_steps=500 \
checkpoint.checkpoint_path=pre_train/from_pretrained/depth/c4_en/lr_0_001_constant_with_warmup_bsz_256/2024-02-26_17-19 \
mode=ft \
downstream.benchmark_dataset=sst2 \
optim.lr_scheduler=linear

