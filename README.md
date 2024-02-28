# DEPTH: Discourse Education through Pre-Training Hierarchically

## Overview
DEPTH (Discourse Education through Pre-Training Hierarchically) is an encoder-decoder model enhancing natural language processing capabilities, particularly in discourse comprehension, coherence, and compositionality. It is designed to extend the pre-training objective "Sentence Un-shuffling" to encoder-decoder models, offering advanced semantic and discourse-level representations. DEPTH supports flexible configurations for from-scratch and continuous pre-training.

## Environment Setup

### Creating a Virtual Environment
1. Ensure Python 3.10 is installed on your system.
2. Create a virtual environment:
   ```bash
   python -m venv depth_env
   ```
3. Activate the virtual environment:
   - On Windows:
     ```bash
     depth_env\Scripts\activate
     ```
   - On Unix or MacOS:
     ```bash
     source depth_env/bin/activate
     ```
4. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Distributed Training with Accelerate

We currently support only single-node multi-GPU training. To train on a single node with 4 GPUs, run:
```accelerate config```

When prompted, select the following options:
```
In which compute environment are you running? <This machine>                                                                                                                                                                                                
Which type of machine are you using? <multi-GPU>  
How many different machines will you use (use more than 1 for multi-node distributed training)? <1>
Do you wish to optimize your script with torch dynamo? <no>
Do you want to use DeepSpeed? <no>
Do you want to use FullyShardedDataParallel? <no> 
Do you want to use Megatron-LM? <no> 
How many GPUs should be used for distributed training? <4>
What GPU(s) (by id) should be used for training on this machine as a comma-seperated list? [all] <enter>
Do you wiush to use FP16 or BF16? <FP16>
```

Next, make sure you are logged into wandb so that you can track your training runs (if prompted, follow the 
instructions to create a free account):
```wandb login```

Once you've configured the accelerator, and set up wandb, you can run a training script such as:
```accelerate launch fine_tune_t5.py```


### Distributed Training with DeepSpeed

To train with DeepSpeed, you must first install it:
```pip install deepspeed```

Then, you can configure the accelerator with:
```accelerate config```

When prompted, select the following options:
```
In which compute environment are you running? <This machine>                                                                                                                                                                                                
Which type of machine are you using? <multi-GPU>                                                                                                                                                                                                   
How many different machines will you use (use more than 1 for multi-node distributed training)? <1>
Do you wish to optimize your script with torch dynamo? <no>                                                                                                                                       
Do you want to use DeepSpeed? [yes/NO]: <yes>                                                                                                                                                                 
Do you want to specify a json file to a DeepSpeed config? [yes/NO]: <yes>                                                                                                                                     
Please enter the path to the json DeepSpeed config file: <zero_stage2_config.json>
Do you want to enable `deepspeed.zero.Init` when using ZeRO Stage-3 for constructing massive models? [yes/NO]: <no>
How many GPU(s) should be used for distributed training? [1]: <4>
``` 

Once you've configured the accelerator, you can run a training script such as:
```deepspeed --no_local_rank --master_port=12345 --num_gpus=<4> pre_train_encoder_decoder.py```

Next, make sure you are logged into wandb so that you can track your training runs (if prompted, follow the
instructions to create a free account):
```wandb login```

## Pre-Training Encoder-Decoders

We use Hydra to manage the configuration of the model, data, and training parameters. The configuration files are 
located in `hydra_configs/`. The default configuration is `hydra_configs/default.yaml`.

Our main script is `pre_train_encoder_decoder.py`. This script supports the following flags:


| Parameter                      | Default Value         | Explanation                                                                                                                                                                                                                                                                         |
|--------------------------------|-----------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mode`                         | 'pt'                  | The category of experiment to run. 'pt' for pre-training and 'ft' for fine-tuning.                                                                                                                                                                                                  |
| `device`                       | 'gpu'                 | The device used for training. Options are 'cpu' and 'gpu'.                                                                                                                                                                                                                          |
| `num_gpus`                     | 8                     | The number of GPUs to use for training. Ignored if using 'cpu'.                                                                                                                                                                                                                     |
| `num_cpus`                     | 64                    | The number of CPUs to use for training.                                                                                                                                                                                                                                             |
| `precision`                    | 'bf16'                | The numerical precision to use for training. Options are 'fp32', 'fp16', and 'bf16'. Note that 'fp16' is not stable for training, and not all GPUs support 'bf16'.                                                                                                                  |
| `eval_only`                    | false                 | Whether to use a model strictly for evaluation over the provided dataset.                                                                                                                                                                                                           |
| `predict_only`                 | false                 | Whether to use a model strictly for prediction over the provided dataset.                                                                                                                                                                                                           |
| `seed`                         | 2137                  | The seed used for random initialization, corruption, shuffling, and other random operations during training.                                                                                                                                                                        |
| `model.model_implementation`   | 'local_t5'            | The model implementation to use. Options are 'local_t5', 'hf_t5', and 'depth'.                                                                                                                                                                                                      |
| `model.name`                   | 'google/t5-v1_1-base' | The name of the model. Used to retrieve the corresponding model and config from HuggingFace.                                                                                                                                                                                        |
| `model.tokenizer`              | 'google/t5-v1_1-base' | The name of the tokenizer. Used to retrieve the corresponding tokenizer from HuggingFace.                                                                                                                                                                                           |
| `model.overwrite.dropout_rate` | 0.0                   | Dropout rate to use in the model. If 0.0, dropout is disabled. If > 0.0, dropout is enabled.                                                                                                                                                                                        |
| `model.add_config.is_bf16`     | true                  | If true, the model will be trained with mixed precision. Otherwise, it will be trained with FP32.                                                                                                                                                                                   |
| `model.random_init`            | true                  | If true, the model will be randomly initialized. Otherwise, it will be initialized from the checkpoint.                                                                                                                                                                             |
| `model.compile`                | true                  | If true, optimize distributed training for the model with torch 2.0's new 'compile' feature. This option makes training generally faster, but has a few-minute overhead at the beginning of training.                                                                               |
| `dataset.path`                 | 'c4'                  | The name of the dataset to use. These correspond with the datasets available in the HuggingFace datasets library.                                                                                                                                                                   |
| `dataset.name`                 | 'en'                  | The partition of the dataset to use. For example, if using the C4 dataset, we may want to specify the 'en' partition to only use the English portion of the dataset.                                                                                                                |
| `dataset.streaming`            | true                  | Whether to stream the dataset or not. If true, the dataset will be streamed from disk. If false, the dataset will be loaded into memory.                                                                                                                                            |
| `dataset.merge_examples`       | false                 | Whether to merge examples or not. If true, dataset examples that are shorter than the model's max input length will be merged together until they are longer than the model's max input length. If false, each example from the dataset will be used as-is when fed into the model. |
| `data.input_length`            | 512                   | The maximum length of the input text. If the input text is longer than this value, it will be truncated.                                                                                                                                                                            |
| `data.target_length`           | 512                   | Target length of the output text. If the output text is longer than this value, it will be truncated.                                                                                                                                                                               |
| `data.mlm_probability`         | 0.3                   | The probability of masking a token in the input text. For example, if set to 0.15, 15% of the tokens in the input text will be masked.                                                                                                                                              |
| `data.mean_noise_span_length`  | 3.0                   | The average span length of the noise span. For example, if set to 3.0, the average noise span will be 3 tokens long.                                                                                                                                                                |
| `data.num_workers`             | 16                    | The number of CPU processes to use for data preprocessing.                                                                                                                                                                                                                          |
| `optim.name`                   | 'adamwscale'          | The optimizer to use. Options are 'adamw', 'adamwscale', and 'adafactor'.                                                                                                                                                                                                           |
| `optim.base_lr`                | 2e-2                  | The initial learning rate to use for the optimizer (not including warmup).                                                                                                                                                                                                          |
| `optim.batch_size`             | 128                   | The size of the batch to use for training. The model will only perform a step after this many examples have been processed.                                                                                                                                                         |
| `optim.total_steps`            | 65_536                | The total number of steps to train for.                                                                                                                                                                                                                                             |
| `optim.epochs`                 | -1                    | The number of epochs to train for. If this value is > 0, it will overwrite the 'total_steps' value.                                                                                                                                                                                 |
| `optim.warmup_steps`           | 10_000                | The number of warmup steps to use for the optimizer.                                                                                                                                                                                                                                |
| `optim.lr_scheduler`           | 'cosine'              | The learning rate scheduler to use. Options are 'linear', 'cosine', 'legacy', and 'constant'.                                                                                                                                                                                       |
| `optim.weight_decay`           | 0.0                   | The weight decay to use for the optimizer.                                                                                                                                                                                                                                          |
| `optim.grad_clip`              | 1.0                   | The gradient clipping value to use for the optimizer. If 0.0, gradient clipping is disabled.                                                                                                                                                                                        |
| `optim.grad_acc`               | 2                     | The number of gradient accumulation steps to use for the optimizer. If 1, gradient accumulation is disabled. Increase this value if you get OOM errors at the expense of additional training time.                                                                                  |
| `evaluate.every_steps`         | 10_000                | The number of steps to wait between evaluations.                                                                                                                                                                                                                                    |
| `evaluate.steps`               | 500                   | The number of steps to use for evaluation (i.e., steps x batch_size = number of examples to evaluate).                                                                                                                                                                              |
| `checkpoint.every_steps`       | 10_000                | The number of steps to wait between checkpoints.                                                                                                                                                                                                                                    |
| `logging.every_steps`          | 100                   | The number of steps to wait between logging.                                                                                                                                                                                                                                        |
| `deepspeed.use_deepspeed`      | false                 | Whether to use DeepSpeed or not.                                                                                                                                                                                                                                                    |



## Distributed Training with SLURM

### Using `srun` with `pre_train_encoder_decoder.py`

If you are running on SLURM, you can run an `srun` command as follows:

 ```bash
srun \
--gres=gpu:4 \
--cpus-per-task=32 \
deepspeed \
 --no_local_rank \
 --master_port=12345 \
 --num_gpus=4 \
 pre_train_encoder_decoder.py \
 num_gpus=4 \
 num_cpus=32 \
 data.num_workers=32 \
 model.model_implementation=hf_t5 \
 optim.total_steps=100_000 \
 optim.lr_scheduler=constant \
 optim.base_lr=1e-4 \
 optim.batch_size=128 \
 optim.grad_acc=2
```
   
 **Note:** 
1. The `--no_local_rank` is used in order to avoid a conflict with the `hydra` flag notation. 
2. The above command works on a GPU with >= 40GB of memory. If you are running on a GPU with less memory, 
     you can reduce the batch size and gradient accumulation steps accordingly.
3. When running 2 or more runs of this script, make sure to use different `--master_port` values for each 
     run.

### Using `sbatch` with `run_pre_training_job.sh`


Alternatively, you can use the provided `run_pre_training_job.sh` script. This is a bash script that is designed to be submitted to a SLURM job scheduler. The header of the script contains several `#SBATCH` directives that are used to specify the resources required for the job. Here's a breakdown of each line:

- `#!/bin/bash`: This line is called a shebang. It specifies that the script should be executed using the bash shell.

- `#SBATCH --job-name="principled-pre-training"`: This line sets the name of the job as it will appear in the SLURM queue. It's helpful for identifying your job among others.

- `#SBATCH -N 1`: This line requests 1 node for the job. SLURM will allocate an entire node to your job.

- `#SBATCH --gres=gpu:4`: This line requests 4 GPUs for the job. The `gres` stands for "generic resources", and it's used to request resources other than CPU and memory.

- `#SBATCH --cpus-per-task=32`: This line requests 32 CPUs for each task. In this case, the entire job is a single task, so it requests 32 CPUs for the job.

- `#SBATCH -w <node_name>`: This line specifies that the job should be run on the `<node_name>` worker node.

- `#SBATCH -o pre_training_runs/slurm_%N_%j_out.txt`: This line specifies the file where the standard output (stdout) of the job will be written. The `%N` and `%j` are placeholders that will be replaced by the node name and job ID, respectively.

- `#SBATCH -e pre_training_runs/slurm_%N_%j_err.txt`: This line specifies the file where the standard error (stderr) of the job will be written. Like the previous line, `%N` and `%j` will be replaced by the node name and job ID.

- `#SBATCH --mail-type=fail`: This line configures SLURM to send an email if the job fails.

- `#SBATCH --mail-user=<your_email>`: This line specifies the email address to which SLURM should send job notifications.

These directives allow you to customize the resources allocated to your job, where output and error logs are written, and how you're notified about the job's status.

### Understanding the Script

The script (both in the form of `sbatch` and in the form of `srun`) contains several commands that are passed to the `deepspeed` command-line tool. Here's a breakdown of the key components:

- `deepspeed`: This is the command-line tool from the DeepSpeed library, which is used to run distributed deep learning models.

- `--no_local_rank`: This flag is used to avoid a conflict with the `hydra` flag notation.

- `--master_port=12345`: This specifies the port that DeepSpeed uses for inter-process communication.

- `--num_gpus=4`: This specifies the number of GPUs to use for training.

- `pre_train_encoder_decoder.py`: This is the main Python script that contains the pre-training code for the DEPTH model.

- `num_gpus=4`, `num_cpus=32`, `data.num_workers=32`: These are arguments passed to the `pre_train_encoder_decoder.py` script. They specify the number of GPUs and CPUs to use for training, and the number of worker processes for data loading, respectively.

### File Structure:

* The `encoder_decoder_utils` package (under the root directory) contains the main code for the DEPTH model. This includes the model architecture, the data processing pipeline, and the training loop.
* The `hydra_configs` directory contains the configuration files for the DEPTH model. These files are written in the Hydra configuration language, and they specify the hyperparameters and settings for the model, the data, and the training process.
* The `pre_train_encoder_decoder.py` script is the main entry point for training the DEPTH model. It uses the Hydra configuration files to set up the model, the data, and the training process, and it runs the training loop, and potentially evaluation and prediction loops. It supports both pre-training and fine-tuning modes.
* The `run_configs` directory contains the SLURM job scripts for running DEPTH and T5 models on a SLURM cluster.
  * At the top level (`depth/run_configs`), you will find scripts for pre-training T5 and DEPTH. There are also designated sub-directories dedicated to fine-tuning the resulting pre-trained models on downstream tasks (e.g., `.../depth/run_configs/glue/sst2/run_fine_tuning_t5_sst2.sh`).
* `checkpoint` directory contains the pre-trained checkpoints for DEPTH and T5 models.
  * Checkpoints are saved into either `depth` or `hf_t5` sub-directories, depending on the model implementation used.
  * Within these sub-directories, you will find either `from_scratch` or `from_pretrained` sub-directories, depending whether pre-training was done from scratch, or from a pre-trained (HuggingFace) checkpoint.
  * Among the `from_scratch` and `from_pretrained` sub-directories, you will find sub-directories corresponding to learning rate, and under those, you will find sub-directories corresponding to batch size. 
  * Under the `batch_size` sub-directories, you will find sub-directories unique to the runtime of the script.
  * Finally, under the runtime sub-directories, you will find the actual checkpoints, saved as `checkpoint-<step>` directories.

### Extending the Script

To extend the script, you can add or modify the arguments passed to the `pre_train_encoder_decoder.py` script. For example, if you want to change the number of training steps, you can add the `optim.total_steps` argument:

```bash
deepspeed \
--no_local_rank \
--master_port=12345 \
--num_gpus=4 \
pre_train_encoder_decoder.py \
num_gpus=4 \
num_cpus=32 \
data.num_workers=32 \
optim.total_steps=200_000
```

In this example, the model will be trained for 200,000 steps.

You can also add new arguments as per the requirements of the `pre_train_encoder_decoder.py` script. Make sure to refer to the script's documentation or source code to understand the available arguments and their usage.

Remember to test the script after making any changes to ensure it runs as expected.

## Repository Maintenance

For questions or suggestions regarding the code, please contact Zachary Bamberger:

- Email: zacharybamberger1@gmail.com

### Ongoing Work

1. Make the tokenizer more efficient and interpretable. Allow the trainer to access metrics such as `length`, or the number of truncated tokens in the input for logging.
2. Add support for additional datasets (e.g., `RedPajama`) and models (e.g., `UL2`).
3. Clean up the corruption code, and add support for additional corruption types at the sentence and paragraph level.
4. Add support for additional pre-training frameworks such as `MegatronLM`

[//]: # (### Debug Mode)

[//]: # (The script includes a debug mode with more frequent logging, evaluation, and checkpointing, allowing for closer monitoring during development and testing phases.)

[//]: # (## Future Work)

[//]: # (- Extend training with larger models and datasets.)

[//]: # (- Evaluate DEPTH on comprehensive generative tasks.)

[//]: # (- Explore the application of DEPTH framework on decoder-only models.)

[//]: # (- Consider additional baselines and benchmarks for further validation.)

