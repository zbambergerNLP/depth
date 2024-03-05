#!/bin/bash

# Example usage:
# sbatch run_fine_tuning_job.sh

#SBATCH --job-name="principled-pre-training"

#SBATCH -N 1                                          # number of minimum nodes
#SBATCH --gres=gpu:2                                  # Request n gpus
#SBATCH --cpus-per-task=32                            # number of cpus per task and per node

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

# Run fine tuning on MNLI Matched with T5
# 400k training examples
# 10k validation examples
# Max sequence length > 512

# Inputs: premise, hypothesis
# Targets: {entailment: 0, neutral: 1, contradiction: 2}

# Baseline
deepspeed \
--no_local_rank \
--master_port=12342 \
--num_gpus=8 \
train_encoder_decoder.py \
mode=ft \
num_gpus=8 \
num_cpus=32 \
precision=bf16 \
model.model_implementation=depth \
model.compile=false \
data.input_length=512 \
data.target_length=8 \
data.num_workers=32 \
data.data_collator=depth \
downstream.benchmark_constants=glue \
downstream.benchmark_dataset=mnli \
downstream.mnli_sub_dir=matched \
dataset.streaming=false \
optim.name=adamw_torch \
optim.base_lr=1e-4 \
optim.batch_size=128 \
optim.total_steps=6_000 \
optim.warmup_steps=1_000 \
optim.grad_acc=1 \
optim.lr_scheduler=linear \
checkpoint.checkpoint_path=pre_train/from_pretrained/depth/c4_en/lr_0_0001_constant_with_warmup_bsz_64_shuffle_p_0_5/2024-03-03_16-09 \
checkpoint.resume=false \
checkpoint.every_steps=1_000 \
checkpoint.save_total_limit=3 \
logging.every_steps=10 \
logging.wandb=true \
evaluate.every_steps=500