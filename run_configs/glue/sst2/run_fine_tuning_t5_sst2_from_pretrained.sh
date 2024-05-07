#!/bin/bash

# Example usage:
# sbatch run_fine_tuning_job.sh

#SBATCH --job-name="principled-pre-training"

#SBATCH -N 1                                          # number of minimum nodes
#SBATCH --gres=gpu:2                                  # Request n gpus
#SBATCH --cpus-per-task=32                            # number of cpus per task and per node

#SBATCH -w bruno1,bruno2,galileo1,galileo2,newton3,newton4,newton5,tdk-bm4

#SBATCH -o fine_tuning_runs/slurm_%N_%j_out.txt      # stdout goes here
#SBATCH -e fine_tuning_runs/slurm_%N_%j_err.txt      # stderr goes here

#SBATCH --mail-type=fail                              # send email if job fails
#SBATCH --mail-user=zachary@campus.technion.ac.il


##############################################################################
### NOTE: This script will not work well without a pre-trained checkpoint. ###
###       Please replace the `checkpoint.checkpoint_path` argument with    ###
###       the path to a pre-trained checkpoint.                            ###
##############################################################################

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

# Load from pre-trained checkpoint
deepspeed \
--no_local_rank \
--master_port=18303 \
--num_gpus=2 \
train_encoder_decoder.py \
num_gpus=2 \
num_cpus=32 \
data.input_length=272 \
data.target_length=8 \
precision=bf16 \
model.model_implementation=hf_t5 \
deepspeed.use_deepspeed=true \
logging.wandb=true \
model.compile=false \
data.data_collator=custom_t5 \
dataset.streaming=false \
optim.total_steps=3_000 \
optim.base_lr=1e-4 \
optim.batch_size=128 \
optim.grad_acc=1 \
optim.warmup_steps=100 \
evaluate.every_steps=250 \
logging.every_steps=50 \
checkpoint.every_steps=5_000 \
checkpoint.output_dir=./checkpoints \
checkpoint.checkpoint_path=checkpoints/pre_train/from_pretrained/hf_t5/allenai/c4_en/lr_0_0001_inverse_sqrt_bsz_200_shuffle_p_0_5/2024-04-03_20-31/checkpoint-256000 \
mode=ft \
downstream.benchmark_dataset=sst2 \
optim.lr_scheduler=linear


