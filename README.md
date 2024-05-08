# DEPTH: Discourse Education through Pre-Training Hierarchically

## Overview

DEPTH (Discourse Education through Pre-Training Hierarchically) is an encoder-decoder model that enhances natural language processing capabilities, particularly in discourse comprehension, coherence, and compositionality. It extends the pre-training objective "Sentence Un-shuffling" to encoder-decoder models, offering advanced semantic and discourse-level representations. DEPTH supports flexible configurations for from-scratch and continuous pre-training.

Key features of DEPTH include:

- Hierarchical sentence representations combined with span corruption and sentence un-shuffling objectives
- Ability to learn semantic and discourse-level representations faster than the T5 baseline
- Improved discourse capabilities when continuously pre-trained from a T5 checkpoint
- Strong performance on GLUE and DiscoEval benchmarks, demonstrating syntactic, semantic, and discourse understanding

## Environment Setup

### Creating a Virtual Environment
1. Ensure Python 3.10 is installed on your system.
2. Create a virtual environment:
   ```bash
   python -m venv .depth
   ```
3. Activate the virtual environment:
   - On Windows:
     ```bash
     .depth\Scripts\activate
     ```
   - On Unix or MacOS:
     ```bash
     source .depth/bin/activate
     ```
4. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Distributed Training with DeepSpeed

DEPTH supports distributed training using DeepSpeed. To use DeepSpeed, install it with:
```pip install deepspeed```

Then, configure the accelerator with:
```accelerate config```

When prompted, select the appropriate options for your setup, including using DeepSpeed and specifying the path to the DeepSpeed configuration file (e.g., `zero_stage2_config.json`).

Make sure to log in to Weights & Biases to track your training runs:
```wandb login```

To run a training script with DeepSpeed, use:
```
deepspeed \
--no_local_rank \
--master_port=<port> \
--num_gpus=<num_gpus> \
train_encoder_decoder.py \
<training_args>
```

Replace `<port>`, `<num_gpus>`, and `<training_args>` with the appropriate values for your setup and desired training configuration.

## Pre-Training and Fine-Tuning

DEPTH uses Hydra for managing the configuration of the model, data, and training parameters. The configuration files are located in `hydra_configs/`, with the default configuration in `hydra_configs/default.yaml`.

The main training script is `train_encoder_decoder.py`, which supports various command-line arguments for configuring the training process, model architecture, dataset, optimization, evaluation, checkpointing, and logging. Refer to the `default.yaml` file for a complete list of available arguments and their default values.

DEPTH supports two main training modes:
1. Pre-training from scratch (FS): Both T5 and DEPTH models are randomly initialized and pre-trained on the C4 dataset with their respective objectives.
2. Continuous pre-training (CPT): Both T5 and DEPTH models are initialized from a pre-trained T5 checkpoint and continue pre-training on the C4 dataset with their respective objectives.

After pre-training, the models can be fine-tuned on downstream tasks such as GLUE and DiscoEval benchmarks to evaluate their performance on natural language understanding and discourse comprehension.

Example pre-training and fine-tuning scripts for DEPTH and T5 models can be found in the `run_configs` directory. These scripts demonstrate how to configure and run the training process using SLURM job schedulers.

## Experiment Results

Experiments comparing DEPTH to the T5 baseline demonstrate that:

1. During pre-training, DEPTH consistently achieves a lower validation loss than a comparably trained T5 model, both in the from-scratch and continuous pre-training settings.
2. DEPTH outperforms T5 in span-corruption loss despite the additional sentence un-shuffling objective.
3. In the from-scratch setting, DEPTH improves faster than T5 on GLUE tasks and consistently outperforms T5 on DiscoEval tasks, indicating its robustness in understanding discourse.
4. In the continuous pre-training setting, DEPTH and T5 obtain comparable performance on GLUE tasks, with DEPTH occasionally outperforming T5 in later stages of pre-training. On DiscoEval tasks, DEPTH still outperforms T5 despite being initialized from a checkpoint pre-trained with a different objective.

These results highlight the effectiveness of DEPTH's hierarchical representations and pre-training objectives in learning semantic and discourse-level information, leading to improved performance on downstream tasks.

## File Structure

- `encoder_decoder_utils`: Contains the main code for the DEPTH model, including the model architecture, data processing pipeline, and training loop.
- `hydra_configs`: Contains the configuration files for the DEPTH model, specifying hyperparameters and settings for the model, data, and training process.
- `train_encoder_decoder.py`: The main entry point for training the DEPTH model, supporting both pre-training and fine-tuning modes.
- `run_configs`: Contains SLURM job scripts for running DEPTH and T5 models on a SLURM cluster, including scripts for pre-training and fine-tuning on downstream tasks.
- `checkpoints`: Contains pre-trained checkpoints for DEPTH and T5 models, organized by model implementation, initialization strategy, learning rate, batch size, and runtime.
- `results`: Contains the results of experiments comparing DEPTH and T5 models on pre-training and downstream tasks, including validation loss, GLUE scores, and DiscoEval scores.
- `requirements.txt`: Contains the required Python packages for running the DEPTH codebase.
- `zero_stage2_config.json`: Contains the DeepSpeed configuration for distributed training with DeepSpeed.
- `fine_tune_constants`: A directory that contains the constants used for fine-tuning the DEPTH model on downstream tasks. 

## Limitations and Future Work

Current limitations of the DEPTH codebase include:

1. Limited computational resources, leading to pre-training with smaller batch sizes and fewer tokens compared to the original T5 model.
2. Lack of support for example packing during pre-training, which can impact training efficiency and stability.
3. Reliance on the NLTK sentence segmentation tool for the sentence un-shuffling objective, which may introduce noise.

Future work may involve:

1. Extending DEPTH to larger models and datasets to further explore its scalability and performance.
2. Applying DEPTH to other task-specific architectures, such as RAG, and higher-level discourse units like paragraphs and documents.
3. Investigating the impact of example packing on DEPTH's training dynamics and performance.
4. Exploring alternative sentence segmentation methods to reduce noise in the sentence un-shuffling objective.

## Repository Maintenance

For questions or suggestions regarding the code, please contact:

- Zachary Bamberger (zacharybamberger1@gmail.com)

We appreciate your interest in DEPTH and welcome any feedback or contributions to improve the model and codebase.