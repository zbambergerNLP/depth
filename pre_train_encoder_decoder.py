import omegaconf
import hydra
import typing
from encoder_decoder_utils import (
    constants,
    setup_utils,
    model_utils,
    optimizer_utils,
    metric_utils,
)

from accelerate import Accelerator
import os
import torch
import transformers
import evaluate

from encoder_decoder_utils.trainer import EncoderDecoderTrainer

"""
T5 example usage: 

deepspeed \
--no_local_rank \
--master_port=12345 \
pre_train_encoder_decoder.py \
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

    config = model_utils.get_config(args=dict_config, logger=logger)
    model = model_utils.get_model(args=dict_config, config=config, logger=logger)

    # TODO: Account for model checkpoint loading in trainer
    # Load the model from a defined checkpoint
    # if dict_config.model.checkpoint_path:
    #     logger.log_message(f'Loading model from checkpoint: {dict_config.model.checkpoint_path}')
    #     accelerator.load_state(dict_config.model.checkpoint_path)

    tokenizer = model_utils.get_tokenizer(args=dict_config, logger=logger)

    optimizer, lr_scheduler = None, None
    if not dict_config.deepspeed.use_deepspeed:
        optimizer = optimizer_utils.get_optimizer(model=model, args=dict_config, logger=logger)
        lr_scheduler = optimizer_utils.get_lr_scheduler(optimizer=optimizer, args=dict_config, logger=logger)

    dataset_splits = model_utils.load_dataset_splits(args=dict_config, logger=logger)
    dataset_splits = model_utils.process_dataset(
        dataset_splits=dataset_splits, args=dict_config, tokenizer=tokenizer, logger=logger)

    data_collator = model_utils.get_data_collator(tokenizer=tokenizer, config=config, args=dict_config, logger=logger)
    logger.log_args(args=dict_config)
    per_device_train_batch_size = (
            dict_config.optim.batch_size // dict_config.optim.grad_acc // dict_config.num_gpus
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
        learning_rate=dict_config.optim.base_lr if dict_config.deepspeed.use_deepspeed else None,
        optim=dict_config.optim.name if dict_config.deepspeed.use_deepspeed else None,
        lr_scheduler_type=dict_config.optim.lr_scheduler if dict_config.deepspeed.use_deepspeed else None,


        # Logging
        report_to=report_to,
        logging_steps=dict_config.logging.every_steps,
        # include_num_input_tokens_seen=True,  # TODO: Consider un-commenting this.
        include_inputs_for_metrics=True,
        length_column_name=constants.DEPTHTokenizerConstants.INPUT_LENGTH,

        # Checkpointing
        # save_total_limit=3,  # TODO: Make this a flag. Default assumes 65k steps, with a checkpoint every 10k steps.
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
        sortish_sampler=True,
        dataloader_num_workers=dict_config.data.num_workers,
        dataloader_persistent_workers=True if dict_config.data.num_workers > 0 else False,
        dataloader_pin_memory=True,
        deepspeed="./zero_stage2_config.json" if dict_config.deepspeed.use_deepspeed else None,
        auto_find_batch_size=dict_config.optim.auto_find_batch_size,  # TODO: Consider removing this.
        # deepspeed=str(dict_config.deepspeed.deepspeed_config_path) if dict_config.deepspeed.use_deepspeed else None,
    )

    # TODO: Add metrics specifically for DEPTH
    def compute_metrics(eval_preds: transformers.EvalPrediction) -> typing.Dict[str, float]:
        """
        Compute the metrics for the evaluation set. This function is called after every evaluation step.

        :param eval_preds: The evaluation predictions. This is an EvalPrediction object, which contains the
            predictions, labels, and sometimes the input ids as well (if specified in the training arguments).
        :return: A dictionary of the metrics. The keys are the metric names, and the values are the metric values.
        """
        predictions = eval_preds.predictions
        labels = eval_preds.label_ids

        batch_size = predictions.shape[0]
        target_length = predictions.shape[1]

        # Flatten the predictions and labels from (batch_size, target_length) to (batch_size * target_length)
        predictions = predictions.reshape([batch_size * target_length])
        labels = labels.reshape([batch_size * target_length])

        clf_metrics = evaluate.combine(
            [
                constants.Metric.ACCURACY.value,
            ]
        )
        return clf_metrics.compute(predictions=predictions, references=labels)

    def preprocess_logits_for_metrics(
            logits: torch.Tensor,  # (batch_size, target_length, vocab_size)
            labels: torch.Tensor,  # (batch_size, target_length)
    ):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)

    trainer = EncoderDecoderTrainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset_splits[constants.DatasetSplit.TRAIN.value],
        eval_dataset=dataset_splits[constants.DatasetSplit.TEST.value],
        tokenizer=tokenizer,
        optimizers=(optimizer, lr_scheduler) if not dict_config.deepspeed.use_deepspeed else (None, None),
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    trainer.train()


if __name__ == '__main__':
    main()
