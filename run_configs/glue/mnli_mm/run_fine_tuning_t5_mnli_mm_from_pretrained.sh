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

nvidia-smi
echo "Running from $(pwd)"
echo "Activating virtual environment"
source .depth/bin/activate
export DS_SKIP_CUDA_CHECK=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run fine tuning on MNLI Mismatched with T5
# 400k training examples
# 10k validation examples
# Max sequence length > 512

# Inputs: premise, hypothesis
# Targets: {entailment: 0, neutral: 1, contradiction: 2}

deepspeed \
--no_local_rank \
--master_port=12255 \
--num_gpus=2 \
train_encoder_decoder.py \
mode=ft \
num_gpus=2 \
num_cpus=32 \
precision=bf16 \
model.model_implementation=hf_t5 \
model.compile=false \
data.input_length=512 \
data.target_length=8 \
data.num_workers=32 \
data.data_collator=custom_t5 \
downstream.benchmark_constants=glue \
downstream.benchmark_dataset=mnli \
downstream.mnli_sub_dir=mismatched \
dataset.streaming=false \
optim.name=adamw_torch \
optim.base_lr=1e-5 \
optim.batch_size=200 \
optim.total_steps=12_000 \
optim.warmup_steps=1_000 \
optim.grad_acc=2 \
optim.lr_scheduler=linear \
checkpoint.resume=false \
checkpoint.every_steps=12_000 \
checkpoint.save_total_limit=3 \
checkpoint.checkpoint_path=checkpoints/pre_train/from_pretrained/hf_t5/allenai/c4_en/lr_0_0001_inverse_sqrt_bsz_200_shuffle_p_0_5/2024-04-03_20-31/checkpoint-2000 \
logging.every_steps=100 \
logging.wandb=true \
evaluate.every_steps=500


# Baseline
#deepspeed \
#--no_local_rank \
#--master_port=12269 \
#--num_gpus=2 \
#train_encoder_decoder.py \
#mode=ft \
#num_gpus=2 \
#num_cpus=32 \
#precision=bf16 \
#model.model_implementation=hf_t5 \
#model.compile=false \
#data.input_length=512 \
#data.target_length=8 \
#data.num_workers=32 \
#data.data_collator=custom_t5 \
#downstream.benchmark_constants=glue \
#downstream.benchmark_dataset=mnli \
#downstream.mnli_sub_dir=mismatched \
#dataset.streaming=false \
#optim.name=adamw_torch \
#optim.base_lr=1e-5 \
#optim.batch_size=256 \
#optim.total_steps=10_000 \
#optim.warmup_steps=1_000 \
#optim.grad_acc=2 \
#optim.lr_scheduler=linear \
#checkpoint.resume=false \
#checkpoint.every_steps=10_000 \
#checkpoint.save_total_limit=3 \
#logging.every_steps=100 \
#logging.wandb=true \
#evaluate.every_steps=500