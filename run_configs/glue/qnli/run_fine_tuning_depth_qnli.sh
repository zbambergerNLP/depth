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


# Run fine tuning on rte with Depth, half precision
# 11k examples


# Load from pre-trained checkpoint
# Relevant checkpoints:
# 1. checkpoints/depth/from_pretrained/lr_0_001/batch_size_256/2024-02-18_12-45  (checkpoint 50_000)
#accelerate launch \
#--mixed_precision bf16 \
#--num_cpu_threads_per_process 32 \
#--num_processes 4 \
#--main_process_port 29502 \
#train_encoder_decoder.py \
#mode=ft \
#precision=bf16 \
#downstream.benchmark_constants=glue \
#downstream.benchmark_dataset=rte \
#downstream.test_results_save_dir=test_results \
#dataset.streaming=false \
#num_gpus=4 \
#num_cpus=32 \
#data.data_collator=depth \
#data.input_length=256 \
#data.target_length=8 \
#model.model_implementation=depth \
#model.compile=false \
#deepspeed.use_deepspeed=false \
#logging.wandb=true \
#logging.every_steps=10 \
#evaluate.every_steps=100 \
#checkpoint.every_steps=1_000 \
#checkpoint.checkpoint_path=checkpoints/depth/from_pretrained/lr_0_001/batch_size_256/2024-02-18_12-45 \
#checkpoint.output_dir=checkpoints \
#data.num_workers=32 \
#optim.total_steps=1_000 \
#optim.lr_scheduler=constant_with_warmup \
#optim.warmup_steps=200 \
#optim.name=adamwscale \
#optim.base_lr=1e-4 \
#optim.batch_size=128 \
#optim.grad_acc=2


# Load from scratch checkpoint
# Relevant checkpoints:
# 1. checkpoints/depth/from_scratch/lr_0_0001/batch_size_256/2024-02-19_14-30  (checkpoint 20_000)


# Run fine tuning on rte with DEPTH, half precision
# Use a continuously pre-trained DEPTH model
deepspeed \
--no_local_rank \
--master_port=11800 \
--num_gpus=2 \
train_encoder_decoder.py \
mode=ft \
num_gpus=2 \
num_cpus=32 \
precision=bf16 \
model.model_implementation=depth \
deepspeed.use_deepspeed=true \
logging.wandb=true \
model.compile=false \
data.input_length=512 \
data.target_length=16 \
data.data_collator=depth \
optim.total_steps=10_000 \
optim.warmup_steps=1_000 \
optim.base_lr=1e-4 \
optim.batch_size=64 \
optim.grad_acc=1 \
evaluate.every_steps=500 \
logging.every_steps=50 \
checkpoint.every_steps=12_000 \
checkpoint.checkpoint_path=checkpoints/pre_train/from_scratch/depth/c4_en/lr_0_0001_linear_bsz_200_shuffle_p_0_5/2024-03-11_18-50/checkpoint-2000 \
downstream.benchmark_dataset=qnli \
optim.lr_scheduler=constant_with_warmup
