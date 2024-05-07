import csv
import random

import datasets
import omegaconf
import hydra
import typing

from transformers.trainer_utils import get_last_checkpoint

from encoder_decoder_utils import (
    constants,
    setup_utils,
    model_utils,
    optimizer_utils,
    metric_utils,
)
from accelerate import Accelerator
import os
import transformers

from encoder_decoder_utils.trainer import EncoderDecoderTrainer
from fine_tune_constants import glue_constants, disco_eval_constants

"""
source .depth/bin/activate

T5 example usage: 

deepspeed \
--no_local_rank \
--master_port=12345 \
train_encoder_decoder.py \
num_gpus=4 \
num_cpus=32 \
model.model_implementation=hf_t5 \
model.compile=false \
data.data_collator=custom_t5 \
optim.total_steps=100_000 \
optim.base_lr=1e-4 \
optim.batch_size=128 \
optim.grad_acc=2

Depth example usage:

deepspeed \
--no_local_rank \
--master_port=12345 \
pre_train_encoder_decoder.py \
num_gpus=4 \
num_cpus=32 \
model.compile=true \
model.model_implementation=depth \
model.tokenizer=depth \
data.data_collator=depth \
data.num_workers=32 \
dataset.validation_set.num_examples=5_000 \
optim.base_lr=1e-4 \
optim.batch_size=128 \
optim.grad_acc=2 \
evaluate.every_steps=50 \
checkpoint.every_steps=50 \
logging.every_steps=10
"""


@hydra.main(
    config_path="hydra_configs",
    config_name="default",
    version_base='1.3',
)
def main(dict_config: omegaconf.DictConfig):
    """
    Main function for training the encoder-decoder model. This function initializes the accelerator, logger, tokenizer,
    model, optimizer, learning rate scheduler, and Trainer, and then trains the model.

    Training Phase (mode):
    - Pre-training (PT): Pre-train the model from scratch or from a pre-trained model.
    - Fine-tuning (FT): Fine-tune the model on a downstream task.

    Random Initialization (model.random_init):
    - True: The pre-training checkpoint is initialized from scratch. During fine-tuning, the model is initialized from
          the from-scratch pre-training checkpoint.
    - False: The pre-training checkpoint is initialized from a pre-trained model. During fine-tuning, the model is
          initialized from the continuously pre-trained model.

    Model Implementation (model.model_implementation):
    - T5: Use the T5 model implementation.
    - DEPTH: Use the DEPTH model implementation.

    Dataset (downstream.benchmark_constants, downstream.benchmark_dataset, dataset.path, dataset.name):
    - GLUE: Use the GLUE benchmark dataset. Specify via the downstream.benchmark_constants flag. Use during
          fine-tuning. Specify the dataset via the downstream.benchmark_dataset flag.
    - C4: Use the C4 dataset. Specify via the dataset.path flag. Use during pre-training. Specify the dataset
          partition via the dataset.name flag.

    Hyper-parameters:
    - Learning rate (optim.base_lr): The base learning rate used for training.
    - Batch size (optim.batch_size): The batch size used for training.
    - Scheduler (optim.lr_scheduler): The learning rate scheduler used for training.

    Experiment Name:
    {training_phase}/{random_init}/{model_implementation}/{dataset}/{learning_rate}/{batch_size}/{date}

    Note that if we are fine-tuning, but no checkpoint path is provided, we will fine-tune from a HuggingFace
    checkpoint. In this case, the experiment `random_init` will be set to `baseline`.

    :param dict_config: A dictionary containing the configuration parameters for the training script. See the
    `default.yaml` file in the `hydra_configs` directory for the default configuration parameters.
    """
    # The below environment variable setting prevents `SSLError: HTTPSConnectionPool(host='huggingface.co', port=443)`
    # See https://github.com/huggingface/transformers/issues/17611
    os.environ['CURL_CA_BUNDLE'] = ''

    experiment = setup_utils.Experiment(dict_config=dict_config)

    # if checkpoint load_directory is specified, then resume training from that checkpoint.
    # Default value is empty string, which means training from scratch, or continuing from a HuggingFace checkpoint.
    if dict_config.checkpoint.checkpoint_path and dict_config.checkpoint.resume:
        # We are resuming pre-training from a checkpoint, and continuing to save checkpoints in the same directory.
        dict_config.checkpoint.output_dir = dict_config.checkpoint.checkpoint_path
        experiment.set_path(dict_config.checkpoint.checkpoint_path)
    else:
        # We are either starting a new pre-training experiment (i.e., a new 'from_pretrained' or 'from_finetuned'
        # pre-training run) or a new fine-tuning experiment. In either case, we will save the checkpoints in a new
        # directory.
        dict_config.checkpoint.output_dir = experiment.path

    # Set the monitoring platform to report to.
    report_to: typing.List[str] = []
    if dict_config.logging.wandb:
        report_to.append(constants.MonitoringPlatform.WANDB.value)

    # Initialize the accelerator and logger.
    accelerator = Accelerator(
        project_dir=dict_config.checkpoint.output_dir,
        cpu=dict_config.device == constants.Device.CPU.value,
        mixed_precision=dict_config.precision,
        log_with=report_to,
        split_batches=True,
        gradient_accumulation_steps=dict_config.optim.grad_acc,
    )

    # Initialize the logger.
    logger = setup_utils.setup_basics(
        accelerator=accelerator,
        args=dict_config,
        experiment=experiment,
    )
    logger.log_args(args=dict_config)
    logger.log_message("Initialized logger successfully.")

    # Create the output directory if it does not exist.
    if not os.path.exists(dict_config.checkpoint.output_dir) and accelerator.is_main_process:
        os.makedirs(dict_config.checkpoint.output_dir)
    logger.log_message(f"Saving checkpoints and metadata to {dict_config.checkpoint.output_dir}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(dict_config.checkpoint.checkpoint_path):
        logger.log_message(f"Searching for model in checkpoint directory {dict_config.checkpoint.checkpoint_path}.")
        if dict_config.checkpoint.checkpoint_path.split("/")[-1].startswith("checkpoint-"):
            # If the checkpoint step is specified, then we will load the checkpoint from that step.
            last_checkpoint = dict_config.checkpoint.checkpoint_path
        else:
            # Otherwise, we will load the last checkpoint in the directory.
            last_checkpoint = get_last_checkpoint(dict_config.checkpoint.checkpoint_path)
        logger.log_message(f"Using checkpoint: {last_checkpoint}")
    elif dict_config.checkpoint.checkpoint_path and not os.path.isdir(dict_config.checkpoint.checkpoint_path):
        # If the user specifies a checkpoint path, but the path does not exist, then raise an error.
        raise FileNotFoundError(f"Checkpoint path {dict_config.checkpoint.checkpoint_path} does not exist.")
    else:
        logger.log_message(
            "No checkpoint path specified. Starting from "
            f"{'scratch' if dict_config.model.random_init else 'HuggingFace checkpoint'}.")


    # Set the number of training steps per device.
    per_device_train_batch_size = (
            dict_config.optim.batch_size // dict_config.optim.grad_acc // dict_config.num_gpus
    )

    # Load the tokenizer
    tokenizer = model_utils.get_tokenizer(
        args=dict_config,
        logger=logger,
    )

    # Get the model configuration
    config = model_utils.get_config(
        args=dict_config,
        logger=logger,
        last_checkpoint=last_checkpoint,
    )

    # Load the dataset splits
    dataset_splits = model_utils.load_dataset_splits(
        args=dict_config,
        logger=logger,
    )

    # Process the dataset splits (e.g., tokenization, padding, etc.)
    dataset_splits = model_utils.process_dataset(
        dataset_splits=dataset_splits,
        args=dict_config,
        tokenizer=tokenizer,
        logger=logger,
    )

    # Get the data collator. Note that T5 and DEPTH each have 2 different data collators: one for pre-training and one
    # for fine-tuning.
    # During pre-training, the data collator is responsible for corrupting the input data and masking tokens, as well
    # as creating the corresponding labels.
    # During fine-tuning, the data collator is responsible for padding and truncating the provided input **and** labels
    data_collator = model_utils.get_data_collator(
        tokenizer=tokenizer,
        config=config,
        args=dict_config,
        logger=logger,
    )

    # Get the model (either T5 or DEPTH)
    model = model_utils.get_model(
        args=dict_config,
        config=config,
        logger=logger,
        tokenizer=tokenizer,
        last_checkpoint=last_checkpoint,
    )

    # Get the optimizer and learning rate scheduler. Note that when using DeepSpeed, the optimizer and learning rate
    # scheduler are not directly used by the Trainer. Instead, they are passed to the DeepSpeed engine, which handles
    # the optimization and learning rate scheduling. In practice, therefore, we pass DummyOptimizer and DummyLRScheduler
    # to the Trainer when using DeepSpeed.
    optimizer = optimizer_utils.get_optimizer(model=model, args=dict_config, logger=logger)
    logger.log_message(f"Optimizer:\n{optimizer}")
    lr_scheduler = optimizer_utils.get_lr_scheduler(optimizer=optimizer, args=dict_config, logger=logger)
    logger.log_message(f"Learning rate scheduler:\n{lr_scheduler}")
    optimizers = (None, None) if dict_config.deepspeed.use_deepspeed else (optimizer, lr_scheduler)
    # Each fine-tuning task has its own set of constants. We need to pass these constants to the compute_metrics
    # function in order to compute the fi ne-tuning metrics. We will use the GLUE constants for the GLUE benchmark
    # datasets, and the DiscoEval constants for the DiscoEval dataset.
    if dict_config.downstream.benchmark_constants == constants.DownstreamDataset.GLUE.value:
        ft_constants = glue_constants.GlueConstants()
    elif dict_config.downstream.benchmark_constants == constants.DownstreamDataset.DISCO_EVAL.value:
        ft_constants = disco_eval_constants.DiscoEvalConstants()
    else:
        ft_constants = 'ni'

    assert not (dict_config.optim.total_steps == -1 and dict_config.optim.epochs == -1), (
        "Either total_steps or epochs should be set to -1, but not both."
    )

    # Set the number of training examples if the evaluation/save strategy is set to "epoch":
    # total_steps = (num_batches // grad_acc // num_gpus) * epochs
    if (
            dict_config.optim.epochs > 0 and
            not isinstance(dataset_splits[constants.DatasetSplit.TRAIN.value], datasets.IterableDataset)
    ):
        batches_per_epoch = len(dataset_splits[constants.DatasetSplit.TRAIN.value]) // dict_config.optim.batch_size
        dict_config.optim.total_steps = batches_per_epoch * dict_config.optim.epochs

    # Set the training arguments.
    training_arguments = transformers.Seq2SeqTrainingArguments(
        # The location where checkpoints are saved
        output_dir=dict_config.checkpoint.output_dir,
        # Overwrite the output directory if it exists for continuous training
        overwrite_output_dir=dict_config.checkpoint.resume,
        remove_unused_columns=False,  # Data collator is responsible for removing unused columns.

        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size * 4,  # Batch size can be larger for evaluation
        gradient_accumulation_steps=dict_config.optim.grad_acc,
        eval_accumulation_steps=dict_config.optim.grad_acc,

        # Training duration
        max_steps=(
            dict_config.optim.total_steps
        ),
        # max_steps overrides num_train_epochs
        num_train_epochs=dict_config.optim.epochs,

        # Optimizer (only used if using DeepSpeed)
        learning_rate=dict_config.optim.base_lr,
        weight_decay=dict_config.optim.weight_decay,
        optim=dict_config.optim.name if dict_config.deepspeed.use_deepspeed else 'adamw_torch',  # Default on HuggingFace
        lr_scheduler_type=dict_config.optim.lr_scheduler,
        warmup_steps=min(dict_config.optim.warmup_steps, dict_config.optim.total_steps * 0.1),

        # Logging
        report_to=report_to,
        logging_steps=dict_config.logging.every_steps,
        include_inputs_for_metrics=True,
        length_column_name=constants.DEPTHTokenizerConstants.INPUT_LENGTH,
        logging_first_step=True,
        include_num_input_tokens_seen=True,

        # TODO: Incorporate evaluation/prediction with generation so that the code commented below works for both
        #  pre-training and fine-tuning.
        # Evaluation/Prediction with generation
        # predict_with_generate=True if dict_config.mode == constants.TrainingPhase.FT.value else False,
        # generation_config=transformers.GenerationConfig(
        #     max_length=dict_config.generation.max_length,
        #     num_beams=dict_config.generation.num_beams,
        #     decoder_start_token_id=tokenizer.pad_token_id,
        #     eos_token_id=tokenizer.eos_token_id,
        #     pad_token_id=tokenizer.pad_token_id,
        # ),

        # Checkpointing
        save_total_limit=dict_config.checkpoint.save_total_limit,
        save_steps=dict_config.checkpoint.every_steps,
        save_strategy="steps" if dict_config.optim.total_steps != -1 else "epoch",
        ignore_data_skip=True if dict_config.checkpoint.resume else False,

        # Evaluation
        eval_steps=dict_config.evaluate.every_steps,
        evaluation_strategy="steps" if dict_config.optim.total_steps != -1 else "epoch",
        load_best_model_at_end=True,
        label_names=[constants.TokenizerConstants.LABELS],

        # Determinism
        seed=dict_config.seed,
        # Loading the model from a checkpoint will not be deterministic.
        data_seed=dict_config.seed if not dict_config.checkpoint.resume else random.randint(100, 1000),

        # Precision
        bf16=dict_config.precision == constants.NumericalPrecision.BF16.value,

        # Distributed training optimization
        # split_batches=True,
        dispatch_batches=False,
        ddp_find_unused_parameters=False,
        torch_compile=dict_config.model.compile,
        dataloader_num_workers=dict_config.data.num_workers,
        dataloader_pin_memory=True,
        deepspeed="./zero_stage2_config.json" if dict_config.deepspeed.use_deepspeed else None,  # TODO: Pass in flag.
    )

    logger.log_message(f"Defining a metric function for {dict_config.mode} mode.")

    # Define the compute_metrics function. This function is used by the Trainer to compute the evaluation metrics.
    def compute_metrics(eval_preds):
        if dict_config.mode == constants.TrainingPhase.PT.value:
            return metric_utils.compute_metrics(
                eval_preds=eval_preds,
                tokenizer=tokenizer,
            )
        else:
            if dict_config.downstream.benchmark_constants == constants.DownstreamDataset.GLUE.value:
                return metric_utils.compute_fine_tune_metrics(
                    eval_preds=eval_preds,
                    benchmark=dict_config.downstream.benchmark_constants,
                    dataset=dict_config.downstream.benchmark_dataset,
                    tokenizer=tokenizer,
                    ft_constants=ft_constants,
                )
            elif dict_config.downstream.benchmark_constants == constants.DownstreamDataset.DISCO_EVAL.value:
                return metric_utils.compute_fine_tune_metrics(
                    eval_preds=eval_preds,
                    metric=constants.Metric.ACCURACY.value,
                    benchmark=dict_config.downstream.benchmark_constants,
                    dataset=dict_config.downstream.benchmark_dataset,
                    tokenizer=tokenizer,
                    ft_constants=ft_constants,
                )
            else:
                return metric_utils.compute_fine_tune_metrics_ni(
                    eval_preds=eval_preds,
                    tokenizer=tokenizer,
                )

    # Initialize the Trainer
    logger.log_message(f"dataset_splits: {dataset_splits}")
    print(f"dataset_splits: {dataset_splits}")
    trainer = EncoderDecoderTrainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset_splits[constants.DatasetSplit.TRAIN.value],
        eval_dataset=dataset_splits[
            constants.DatasetSplit.TEST.value if (
                    dict_config.mode == constants.TrainingPhase.PT.value or
                    dict_config.downstream.benchmark_constants == 'ni'
            )
            else constants.DatasetSplit.VALIDATION.value
        ],
        tokenizer=tokenizer,
        optimizers=optimizers,
        data_collator=data_collator,
        # Only use the compute_discourse_metrics function during pre-training if the model is a DEPTH model.
        compute_discourse_metrics=(
            metric_utils.compute_metrics_depth if
            dict_config.mode == constants.TrainingPhase.PT.value and
            dict_config.model.model_implementation == constants.ModelImplementation.DEPTH.value else
            None
        ),
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=metric_utils.preprocess_logits_for_metrics,
        callbacks=[]
    )

    # Train the model
    trainer.train(
        resume_from_checkpoint=last_checkpoint if last_checkpoint and dict_config.checkpoint.resume else None,
    )

    test_predictions, test_labels, test_metrics = trainer.predict(
        test_dataset=dataset_splits[constants.DatasetSplit.TEST.value],
        metric_key_prefix=constants.DatasetSplit.TEST.value,
    )
    logger.log_message(f"Test metrics: {test_metrics}")

    # Save the test predictions to a file
    if dict_config.downstream.benchmark_constants == constants.DownstreamDataset.GLUE.value:
        ft_constants = glue_constants.GlueConstants()
        file_name = ft_constants[dict_config.downstream.benchmark_dataset].SUBMISSION_NAME
        # Find the indices of the labels that are not -100, eos_token_id, or pad_token_id (Only prediction).
        label_ids_mask = (test_labels != -100) & (test_labels != tokenizer.eos_token_id) & (
                test_labels != tokenizer.pad_token_id)
        # Reshape the predictions to match the mask.
        test_predictions = test_predictions[label_ids_mask].reshape(-1)
        # Create a dictionary to map the label to the prediction id.
        label_to_id = {label: idx for idx, label in ft_constants[dict_config.downstream.benchmark_dataset].LABELS.items()}
        test_predictions_for_tsv = []
        for idx, pred in enumerate(test_predictions):
            if tokenizer.decode(pred) in label_to_id.keys():
                test_predictions_for_tsv.append((idx, label_to_id[tokenizer.decode(pred)]))
            else:
                test_predictions_for_tsv.append((idx, ft_constants.OTHER))
        try:
            os.mkdir(dict_config.downstream.test_results_save_dir)
        except FileExistsError:
            logger.log_message(f"Directory {dict_config.downstream.test_results_save_dir} already exists. Continuing...")
        if dict_config.downstream.benchmark_dataset == constants.GLUEConstants.MNLI:
            if dict_config.downstream.mnli_sub_dir == constants.GLUEConstants.MISMATCHED:
                file_names = file_name.split(".")
                file_name = file_names[0] + "-mm." + file_names[1]
            else:
                file_names = file_name.split(".")
                file_name = file_names[0] + "-m." + file_names[1]
        with open(os.path.join(dict_config.downstream.test_results_save_dir, file_name), 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow([constants.GLUEConstants.ID, constants.GLUEConstants.LABEL])
            for row in test_predictions_for_tsv:
                writer.writerow(row)
    else:
        logger.log_message(test_metrics)


if __name__ == '__main__':
    main()