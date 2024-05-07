#!/bin/bash

# Example usage:
# sbatch run_fine_tuning_job.sh

#SBATCH --job-name="principled-pre-training"

#SBATCH -N 1                                          # number of minimum nodes
#SBATCH --gres=gpu:2                                  # Request n gpus
#SBATCH --cpus-per-task=32                            # number of cpus per task and per node

#SBATCH -p nlp
#SBATCH -A nlp
#SBATCH -w nlp-ada-1,nlp-ada-2,nlp-a40-1


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

# Run fine tuning on SST-2 with T5
# 67.3k training examples
# 872 validation examples
# Max sequence length = 268

# Inputs: sentence
# Targets: {negative: 0, positive: 1}

# Load from scratch checkpoint
deepspeed \
--no_local_rank \
--master_port=18202 \
--num_gpus=2 \
train_encoder_decoder.py \
num_gpus=2 \
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
dataset.streaming=false \
optim.total_steps=3_000 \
optim.base_lr=1e-5 \
optim.batch_size=128 \
optim.grad_acc=1 \
optim.warmup_steps=300 \
evaluate.every_steps=200 \
logging.every_steps=50 \
checkpoint.every_steps=50_000 \
checkpoint.checkpoint_path=checkpoints/pre_train/from_scratch/depth/c4_en/lr_0_0001_linear_bsz_200_shuffle_p_0_5/2024-03-11_18-50/checkpoint-1000000 \
mode=ft \
downstream.benchmark_dataset=sst2 \
optim.lr_scheduler=linear