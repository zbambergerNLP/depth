#!/bin/bash

# Example usage:
# sbatch run_fine_tuning_job.sh

#SBATCH --job-name="principled-pre-training"

#SBATCH -N 1                                          # number of minimum nodes
#SBATCH --gres=gpu:8                                  # Request n gpus
#SBATCH --cpus-per-task=32                            # number of cpus per task and per node

#SBATCH -A nlp
#SBATCH -p nlp
#SBATCH -w nlp-2080-[1-2]

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


# Run fine tuning on SST-2 with Depth, half precision
accelerate launch \
--mixed_precision bf16 \
--num_cpu_threads_per_process 32 \
--num_processes 2 \
--main_process_port 29502 \
train_encoder_decoder.py \
mode=ft \
precision=bf16 \
downstream.benchmark_constants=glue \
downstream.benchmark_dataset=sst2 \
downstream.test_results_save_dir=test_results \
num_gpus=2 \
num_cpus=32 \
data.data_collator=depth \
model.model_implementation=depth \
model.compile=true \
deepspeed.use_deepspeed=false \
logging.wandb=false \
logging.every_steps=10 \
evaluate.every_steps=100 \
checkpoint.every_steps=1_000 \
checkpoint.checkpoint_path=checkpoints/depth/from_pretrained/lr_0_001/batch_size_256/2024-02-18_12-45 \
checkpoint.output_dir=checkpoints \
data.num_workers=32 \
optim.total_steps=20_000 \
optim.lr_scheduler=linear \
optim.name=adamwscale \
optim.base_lr=5e-5 \
optim.batch_size=64 \
optim.grad_acc=4

# Run fine tuning on SST-2 with T5, half precision
# 70k examples
accelerate launch \
--mixed_precision bf16 \
--num_cpu_threads_per_process 16 \
--num_processes 2 \
--main_process_port 29502 \
train_encoder_decoder.py \
mode=ft \
precision=bf16 \
downstream.benchmark_constants=glue \
downstream.benchmark_dataset=sst2 \
downstream.test_results_save_dir=test_results \
num_gpus=2 \
num_cpus=16 \
data.data_collator=custom_t5 \
model.model_implementation=hf_t5 \
model.compile=true \
deepspeed.use_deepspeed=false \
logging.wandb=false \
logging.every_steps=10 \
evaluate.every_steps=100 \
checkpoint.every_steps=1_000 \
checkpoint.checkpoint_path=checkpoints/hf_t5/from_scratch/lr_0_0001/batch_size_256/2024-02-18_17-13 \
checkpoint.output_dir=checkpoints \
data.num_workers=16 \
optim.total_steps=20_000 \
optim.lr_scheduler=linear \
optim.name=adamwscale \
optim.base_lr=1e-4 \
optim.batch_size=32 \
optim.grad_acc=4

# Run fine tuning on SST-2 with mixed precision
#accelerate launch \
#--mixed_precision no \
#--num_cpu_threads_per_process 32 \
#--num_processes 4 \
#train_encoder_decoder.py \
#mode=ft \
#precision=bf16 \
#downstream.benchmark_constants=glue \
#downstream.benchmark_dataset=sst2 \
#downstream.test_results_save_dir=test_results \
#num_gpus=4 \
#num_cpus=32 \
#model.model_implementation=hf_t5 \
#model.compile=true \
#deepspeed.use_deepspeed=false \
#logging.wandb=true \
#logging.every_steps=10 \
#evaluate.every_steps=100 \
#checkpoint.every_steps=1_000 \
#checkpoint.output_dir=checkpoints \
#data.num_workers=32 \
#optim.total_steps=20_000 \
#optim.lr_scheduler=constant \
#optim.name=adamwscale \
#optim.base_lr=1e-4 \
#optim.batch_size=64 \
#optim.grad_acc=4

# DeepSpeed fine tuning on SST-2 with full precision
# 70k examples
deepspeed \
--no_local_rank \
--master_port=12341 \
--num_gpus=8 \
train_encoder_decoder.py \
num_gpus=8 \
num_cpus=32 \
precision=no \
model.model_implementation=hf_t5 \
deepspeed.use_deepspeed=true \
logging.wandb=true \
model.compile=false \
data.data_collator=custom_t5 \
optim.total_steps=3_000 \
optim.base_lr=1e-4 \
optim.batch_size=64 \
optim.grad_acc=4 \
evaluate.every_steps=50 \
logging.every_steps=10 \
checkpoint.every_steps=500 \
mode=ft \
downstream.benchmark_dataset=sst2 \
optim.lr_scheduler=linear

# DeepSpeed fine tuning on cola with full precision
# 11k examples
deepspeed \
--no_local_rank \
--master_port=12342 \
--num_gpus=2 \
train_encoder_decoder.py \
num_gpus=2 \
num_cpus=32 \
precision=no \
model.model_implementation=depth \
deepspeed.use_deepspeed=true \
logging.wandb=true \
model.compile=false \
data.data_collator=depth \
optim.total_steps=500 \
optim.base_lr=1e-4 \
optim.batch_size=64 \
optim.grad_acc=8 \
evaluate.every_steps=20 \
logging.every_steps=5 \
checkpoint.every_steps=100 \
checkpoint.checkpoint_path=checkpoints/depth/from_pretrained/lr_0_001/batch_size_256/2024-02-18_12-45 \
mode=ft \
downstream.benchmark_dataset=cola \
optim.lr_scheduler=linear

# DeepSpeed fine tuning on qnli with full precision
# 116 examples
deepspeed \
--no_local_rank \
--master_port=12343 \
--num_gpus=8 \
train_encoder_decoder.py \
num_gpus=8 \
num_cpus=32 \
precision=no \
model.model_implementation=hf_t5 \
deepspeed.use_deepspeed=true \
logging.wandb=true \
model.compile=false \
data.data_collator=custom_t5 \
optim.total_steps=6_000 \
optim.base_lr=1e-4 \
optim.batch_size=64 \
optim.grad_acc=4 \
evaluate.every_steps=100 \
logging.every_steps=20 \
checkpoint.every_steps=1_000 \
mode=ft \
downstream.benchmark_dataset=qnli \
optim.lr_scheduler=linear

# DeepSpeed fine tuning on qqp with full precision
# 795k examples
deepspeed \
--no_local_rank \
--master_port=12344 \
--num_gpus=8 \
train_encoder_decoder.py \
num_gpus=8 \
num_cpus=32 \
precision=no \
model.model_implementation=hf_t5 \
deepspeed.use_deepspeed=true \
logging.wandb=true \
model.compile=false \
data.data_collator=custom_t5 \
optim.total_steps=20_000 \
optim.base_lr=1e-4 \
optim.batch_size=64 \
optim.grad_acc=4 \
evaluate.every_steps=400 \
logging.every_steps=50 \
checkpoint.every_steps=4_000 \
mode=ft \
downstream.benchmark_dataset=qqp \
optim.lr_scheduler=linear

# DeepSpeed fine tuning on mrpc with full precision
# 6k examples
deepspeed \
--no_local_rank \
--master_port=12345 \
--num_gpus=8 \
train_encoder_decoder.py \
num_gpus=8 \
num_cpus=32 \
precision=no \
model.model_implementation=hf_t5 \
deepspeed.use_deepspeed=true \
logging.wandb=true \
model.compile=false \
data.data_collator=custom_t5 \
optim.total_steps=1_000 \
optim.base_lr=1e-4 \
optim.batch_size=64 \
optim.grad_acc=4 \
evaluate.every_steps=50 \
logging.every_steps=10 \
checkpoint.every_steps=200 \
mode=ft \
downstream.benchmark_dataset=mrpc \
optim.lr_scheduler=linear

# DeepSpeed fine tuning on stsb with full precision
# 9k examples
deepspeed \
--no_local_rank \
--master_port=12346 \
--num_gpus=8 \
train_encoder_decoder.py \
num_gpus=8 \
num_cpus=32 \
precision=no \
model.model_implementation=hf_t5 \
deepspeed.use_deepspeed=true \
logging.wandb=true \
model.compile=false \
data.data_collator=custom_t5 \
optim.total_steps=1_000 \
optim.base_lr=1e-4 \
optim.batch_size=64 \
optim.grad_acc=4 \
evaluate.every_steps=50 \
logging.every_steps=10 \
checkpoint.every_steps=200 \
mode=ft \
downstream.benchmark_dataset=stsb \
optim.lr_scheduler=linear

# DeepSpeed fine tuning on wnli with full precision
# 1k examples
deepspeed \
--no_local_rank \
--master_port=12347 \
--num_gpus=8 \
train_encoder_decoder.py \
num_gpus=8 \
num_cpus=32 \
precision=no \
model.model_implementation=hf_t5 \
deepspeed.use_deepspeed=true \
logging.wandb=true \
model.compile=false \
data.data_collator=custom_t5 \
optim.total_steps=500 \
optim.base_lr=1e-4 \
optim.batch_size=64 \
optim.grad_acc=4 \
evaluate.every_steps=20 \
logging.every_steps=5 \
checkpoint.every_steps=100 \
mode=ft \
downstream.benchmark_dataset=wnli \
optim.lr_scheduler=linear

# DeepSpeed fine tuning on rte with full precision
# 6k examples
deepspeed \
--no_local_rank \
--master_port=12348 \
--num_gpus=8 \
train_encoder_decoder.py \
num_gpus=8 \
num_cpus=32 \
precision=no \
model.model_implementation=hf_t5 \
deepspeed.use_deepspeed=true \
logging.wandb=true \
model.compile=false \
data.data_collator=custom_t5 \
optim.total_steps=1_000 \
optim.base_lr=1e-4 \
optim.batch_size=64 \
optim.grad_acc=4 \
evaluate.every_steps=50 \
logging.every_steps=10 \
checkpoint.every_steps=200 \
mode=ft \
downstream.benchmark_dataset=rte \
optim.lr_scheduler=linear

