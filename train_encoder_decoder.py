import logging
import time
import csv
import evaluate
import omegaconf
import hydra
import typing

import wandb

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
    report_to: typing.List[str] = []
    if dict_config.logging.wandb:
        report_to.append(constants.MonitoringPlatform.WANDB.value)

    # Initialize the accelerator and logger.
    accelerator = Accelerator(
        project_dir=os.getcwd(),
        cpu=dict_config.device == constants.Device.CPU.value,
        mixed_precision=dict_config.precision,
        log_with=report_to,
        split_batches=True,
        gradient_accumulation_steps=dict_config.optim.grad_acc,
    )

    logger = setup_utils.setup_basics(
        accelerator=accelerator,
        args=dict_config,
    )
    logger.log_message("Initialized logger successfully.")

    tokenizer = model_utils.get_tokenizer(args=dict_config, logger=logger)
    config = model_utils.get_config(args=dict_config, logger=logger)
    model = model_utils.get_model(args=dict_config, config=config, logger=logger, tokenizer=tokenizer)

    optimizer = optimizer_utils.get_optimizer(model=model, args=dict_config, logger=logger)
    logger.log_message(f"Optimizer: {optimizer}")
    lr_scheduler = optimizer_utils.get_lr_scheduler(optimizer=optimizer, args=dict_config, logger=logger)
    logger.log_message(f"Learning rate scheduler: {lr_scheduler}")

    dataset_splits = model_utils.load_dataset_splits(args=dict_config, logger=logger)
    dataset_splits = model_utils.process_dataset(
        dataset_splits=dataset_splits,
        args=dict_config,
        tokenizer=tokenizer,
        logger=logger,
    )

    logger.log_message(f"Dataset splits: {dataset_splits.keys()}")
    data_collator = model_utils.get_data_collator(
        tokenizer=tokenizer,
        config=config,
        args=dict_config,
        logger=logger,
    )
    logger.log_message(f"Data collator: {type(data_collator)}")
    logger.log_args(args=dict_config)
    per_device_train_batch_size = (
            dict_config.optim.batch_size // dict_config.optim.grad_acc // dict_config.num_gpus
    )

    # Create an output directory specific to the execution time of the script.
    date_time = time.strftime("%Y-%m-%d_%H-%M")
    dict_config.checkpoint.output_dir = os.path.join(
        dict_config.checkpoint.output_dir, f"{date_time}_{dict_config.model.model_implementation}"
    )

    training_arguments = transformers.Seq2SeqTrainingArguments(
        output_dir=dict_config.checkpoint.output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size * 2,  # Batch size can be larger for evaluation
        gradient_accumulation_steps=dict_config.optim.grad_acc,
        eval_accumulation_steps=dict_config.optim.grad_acc,
        max_steps=dict_config.optim.total_steps,
        remove_unused_columns=False,  # Data collator is responsible for removing unused columns.

        # Optimizer
        # learning_rate=dict_config.optim.base_lr if dict_config.deepspeed.use_deepspeed else None,
        # weight_decay=dict_config.optim.weight_decay,
        # optim=dict_config.optim.name,
        # # lr_scheduler_type=dict_config.optim.lr_scheduler if dict_config.deepspeed.use_deepspeed else lr_scheduler,
        # lr_scheduler_type=dict_config.optim.lr_scheduler,
        # warmup_steps=dict_config.optim.warmup_steps,

        # Logging
        report_to=report_to,
        logging_steps=dict_config.logging.every_steps,
        # include_num_input_tokens_seen=True,  # TODO: Consider un-commenting this.
        include_inputs_for_metrics=True,
        length_column_name=constants.DEPTHTokenizerConstants.INPUT_LENGTH,

        # Checkpointing
        save_total_limit=10,  # TODO: Make this a flag. Default assumes 65k steps, with a checkpoint every 10k steps.
        save_steps=dict_config.checkpoint.every_steps,

        # Evaluation
        eval_steps=dict_config.evaluate.every_steps,
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        label_names=["labels"],

        # Determinism
        seed=dict_config.seed,
        data_seed=dict_config.seed,

        # Precision
        bf16=dict_config.precision == constants.NumericalPrecision.BF16.value,

        # Distributed training optimization
        torch_compile=dict_config.model.compile,
        dataloader_num_workers=dict_config.data.num_workers,
        dataloader_pin_memory=True,
        deepspeed="./zero_stage2_config.json" if dict_config.deepspeed.use_deepspeed else None,
    )

    optimizers = (None, None) if dict_config.deepspeed.use_deepspeed else (optimizer, lr_scheduler)

    ft_constants = glue_constants.GlueConstants() if dict_config.data.benchmark_constants == 'glue' \
        else disco_eval_constants.DiscoEvalConstants()

    def compute_metrics(eval_preds):
        if dict_config.mode == constants.TrainingPhase.PT:
            return metric_utils.compute_metrics(
                eval_preds=eval_preds,
                tokenizer=tokenizer,
            )
        else:
            if dict_config.model.model_implementation == constants.ModelImplementation.DEPTH.value:
                raise NotImplementedError("Depth model does not support evaluation metrics in fine tuning.")
            else:
                if dict_config.data.benchmark_constants == 'glue':
                    return metric_utils.compute_fine_tune_metrics(
                        eval_preds=eval_preds,
                        benchmark=dict_config.data.benchmark_constants,
                        dataset=dict_config.data.benchmark_dataset,
                        tokenizer=tokenizer,
                        ft_constants=ft_constants,
                    )
                else:
                    # TODO: Use constants instead of literal strings
                    return metric_utils.compute_fine_tune_metrics(
                        eval_preds=eval_preds,
                        metric='accuracy',
                        tokenizer=tokenizer,
                        ft_constants=ft_constants,
                    )

    trainer = EncoderDecoderTrainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset_splits[constants.DatasetSplit.TRAIN.value],
        eval_dataset=dataset_splits[
            constants.DatasetSplit.TEST.value if dict_config.mode == constants.TrainingPhase.PT
            else constants.DatasetSplit.VALIDATION.value],
        tokenizer=tokenizer,
        optimizers=optimizers,
        data_collator=data_collator,
        compute_discourse_metrics=(
            metric_utils.compute_metrics_depth if
            dict_config.model.model_implementation == constants.ModelImplementation.DEPTH.value else
            None
        ),
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=metric_utils.preprocess_logits_for_metrics,
        callbacks=[]
    )

    trainer.train()

    test_predictions, test_labels, test_metrics = trainer.predict(
        test_dataset=dataset_splits[constants.DatasetSplit.TEST.value],
        metric_key_prefix=constants.DatasetSplit.TEST.value,
    )
    logger.log_message(f"Test metrics: {test_metrics}")
    if dict_config.data.benchmark_constants == 'glue':
        """
        The following code is specific to the GLUE benchmark. It creates
        """
        # Load the constants for the GLUE benchmark.
        file_name = ft_constants[dict_config.data.benchmark_dataset].SUBMISSION_NAME
        # Find the indices of the labels that are not -100, eos_token_id, or pad_token_id (Only prediction).
        label_ids_mask = (test_labels != -100) & (test_labels != tokenizer.eos_token_id) & (
                test_labels != tokenizer.pad_token_id)
        # Reshape the predictions to match the mask.
        test_predictions = test_predictions[label_ids_mask].reshape(-1)
        # Create a dictionary to map the label to the prediction id.
        label_to_id = {label: idx for idx, label in ft_constants[dict_config.data.benchmark_dataset].LABELS.items()}
        test_predictions_for_tsv = []
        for idx, pred in enumerate(test_predictions):
            if tokenizer.decode(pred) in label_to_id.keys():
                test_predictions_for_tsv.append((idx, label_to_id[tokenizer.decode(pred)]))
            else:
                test_predictions_for_tsv.append((idx, ft_constants.OTHER))
        try:
            os.mkdir(dict_config.data.test_results_save_dir)
        except FileExistsError:
            logger.log_message(f"Directory {dict_config.data.test_results_save_dir} already exists. Continuing...")
        if dict_config.data.benchmark_dataset == 'mnli':
            if dict_config.data.mnli_sub_dir == 'mismatched':
                file_names = file_name.split(".")
                file_name = file_names[0] + "-mm." + file_names[1]
            else:
                file_names = file_name.split(".")
                file_name = file_names[0] + "-m." + file_names[1]
        with open(os.path.join(dict_config.data.test_results_save_dir, file_name), 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['id', 'label'])
            for row in test_predictions_for_tsv:
                writer.writerow(row)
    else:
        logger.log_message(test_metrics)


if __name__ == '__main__':
    main()
