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


# Run fine tuning on Cola with T5, half precision
# 11k examples

# Use a from-scratch pre-trained T5 model


# Use a continuously pre-trained T5 model
deepspeed \
--no_local_rank \
--master_port=12342 \
--num_gpus=8 \
train_encoder_decoder.py \
mode=ft \
num_gpus=8 \
num_cpus=32 \
precision=bf16 \
model.model_implementation=hf_t5 \
model.compile=false \
data.input_length=512 \
data.target_length=8 \
data.num_workers=32 \
data.data_collator=custom_t5 \
downstream.benchmark_constants=glue \
downstream.benchmark_dataset=cola \
dataset.streaming=false \
optim.name=adamw_torch \
optim.base_lr=1e-4 \
optim.batch_size=32 \
optim.total_steps=4_000 \
optim.warmup_steps=100 \
optim.grad_acc=1 \
optim.lr_scheduler=linear \
checkpoint.checkpoint_path=checkpoints/pre_train/from_pretrained/hf_t5/c4_en/lr_0_0001_linear_bsz_256_shuffle_p_0_5/2024-03-05_01-04/checkpoint-1/ \
checkpoint.resume=false \
checkpoint.every_steps=1_000 \
checkpoint.save_total_limit=3 \
logging.every_steps=10 \
logging.wandb=true \
evaluate.every_steps=100




#deepspeed \
#--no_local_rank \
#--master_port=12342 \
#--num_gpus=4 \
#train_encoder_decoder.py \
#mode=ft \
#num_gpus=4 \
#num_cpus=32 \
#precision=bf16 \
#model.model_implementation=depth \
#deepspeed.use_deepspeed=true \
#logging.wandb=true \
#model.compile=false \
#data.input_length=256 \
#data.target_length=8 \
#data.data_collator=depth \
#optim.total_steps=4_000 \
#optim.base_lr=1e-5 \
#optim.batch_size=128 \
#optim.grad_acc=1 \
#evaluate.every_steps=100 \
#logging.every_steps=10 \
#checkpoint.every_steps=1000 \
#checkpoint.checkpoint_path=checkpoints/depth/from_pretrained/lr_0_001/batch_size_256/2024-02-18_12-45 \
#downstream.benchmark_dataset=cola \
#optim.lr_scheduler=linear
