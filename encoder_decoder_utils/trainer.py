from typing import (
    Optional,
    List,
    Dict,
    Union,
    Tuple,
    Mapping,
    Callable,
    Any,
)

import numpy as np
import transformers
import torch
from datasets import Dataset
from torch import nn
from transformers import EvalPrediction
from transformers.deepspeed import deepspeed_init
from transformers.trainer_pt_utils import (
    nested_numpify,
    IterableDatasetShard,
    nested_concat,
    find_batch_size,
    nested_detach,
)

from transformers.trainer_utils import EvalLoopOutput, denumpify_detensorize, has_length
from transformers.utils import logging
from torch.utils.data import DataLoader
from encoder_decoder_utils import t5_model
from encoder_decoder_utils import constants

logger = logging.get_logger(__name__)


class EncoderDecoderTrainer(transformers.Seq2SeqTrainer):
    def __init__(
            self,
            model: Union[transformers.PreTrainedModel, nn.Module] = None,
            args: transformers.TrainingArguments = None,
            data_collator: Optional[transformers.DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
            tokenizer: Optional[transformers.PreTrainedTokenizer] = None,
            model_init: Optional[Callable[[], transformers.PreTrainedModel]] = None,
            compute_metrics: Optional[
                Callable[[transformers.EvalPrediction], Mapping[str, float]]] = None,
            compute_discourse_metrics: Optional[
                Callable[
                    [transformers.EvalPrediction, transformers.PreTrainedTokenizer, torch.Tensor],
                    Mapping[str, float]
                ]] = None,
            callbacks: Optional[List[transformers.TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler] = (None, None),
            preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        self.compute_discourse_metrics = compute_discourse_metrics
        self.total_train_tokens = 0
        self.total_validation_tokens = 0

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        batch_size = self._train_batch_size

        return self.accelerator.prepare(
            torch.utils.data.DataLoader(
                train_dataset,
                collate_fn=data_collator,
                batch_size=batch_size,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                drop_last=False,
            )
        )

    def get_eval_dataloader(
            self,
            eval_dataset: Optional[Dataset] = None,
    ) -> DataLoader:
        """
        Create the evaluation dataloader to be used during training.

        :param eval_dataset: The dataset to use. If None, will use the dataset passed during init.
        :return: A DataLoader instance to be used during evaluation.
        """

        assert eval_dataset is not None or self.eval_dataset is not None, \
            "Trainer: evaluation requires an eval_dataset."

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self.data_collator

        dataloader_params = {
            constants.DataLoaderConstants.BATCH_SIZE: self.args.eval_batch_size,
            constants.DataLoaderConstants.COLLATE_FN: data_collator,
            constants.DataLoaderConstants.NUM_WORKERS: (
                self.args.dataloader_num_workers if self.args.dataloader_num_workers <= 8 else 8
            ),
            constants.DataLoaderConstants.PIN_MEMORY: self.args.dataloader_pin_memory,
            constants.DataLoaderConstants.PERSISTENT_WORKERS: self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params[constants.DataLoaderConstants.SAMPLER] = self._get_eval_sampler(eval_dataset)
            dataloader_params[constants.DataLoaderConstants.DROP_LAST] = self.args.dataloader_drop_last

        return self.accelerator.prepare(DataLoader(eval_dataset, **dataloader_params))

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        data_collator = self.data_collator
        return self.accelerator.prepare(
            torch.utils.data.DataLoader(
                test_dataset,
                # shuffle=False,
                collate_fn=data_collator,
                batch_size=self.args.eval_batch_size,
                pin_memory=self.args.dataloader_pin_memory,
                drop_last=False,
                num_workers=self.args.dataloader_num_workers,
                # num_workers=8,  # The maximum number of workers on the eval set
            )
        )

    def evaluation_loop(
            self,
            dataloader: DataLoader,
            description: str,
            prediction_loss_only: Optional[bool] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only
        )

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        is_shuffled_host = None
        losses_host = None
        sequence_losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None

        # losses/preds/labels on CPU (final containers)
        all_is_shuffled = None
        all_losses = None
        all_sequence_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            if isinstance(model, t5_model.DepthForConditionalGeneration):
                loss, sequence_losses, logits, labels = self.depth_prediction_step(
                    model,
                    inputs,
                    prediction_loss_only,
                    ignore_keys=ignore_keys,
                )
                # 1 if batch is shuffled, 0 otherwise
                is_shuffled = inputs[constants.DepthDataCollatorConstants.IS_SHUFFLED].clone().detach().to(args.device)
            else:  # Using a model other than Depth (e.g., T5)
                loss, logits, labels = self.prediction_step(
                    model,
                    inputs,
                    prediction_loss_only,
                    ignore_keys=ignore_keys,
                )
                sequence_losses = None
                is_shuffled = None

            main_input_name = getattr(self.model, "main_input_name", constants.T5TokenizerConstants.INPUT_IDS)
            inputs_decode = self._prepare_input(inputs[main_input_name]) if args.include_inputs_for_metrics else None

            # Update containers on host
            if is_shuffled is not None:
                is_shuffled = self._nested_gather(is_shuffled)
                is_shuffled_host = is_shuffled if is_shuffled_host is None else torch.cat(
                    (is_shuffled_host, is_shuffled),
                    dim=0)
            if loss is not None:
                losses = self.gather_function((loss.repeat(batch_size)))
                losses_host = losses if losses_host is None else nested_concat(losses_host, losses, padding_index=-100)
            if sequence_losses is not None:
                sequence_losses = self.accelerator.pad_across_processes(sequence_losses)
                sequence_losses = self._nested_gather(sequence_losses)
                sequence_losses_host = (
                    sequence_losses if sequence_losses_host is None else torch.cat(
                        (sequence_losses_host, sequence_losses),
                        dim=0)
                )
            if labels is not None:
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
                inputs_decode = self.gather_function((inputs_decode))
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                )
            if logits is not None:
                logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.gather_function((logits))
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)

            if labels is not None:
                labels = self.gather_function((labels))
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if is_shuffled_host is not None:
                    is_shuffled = nested_numpify(is_shuffled_host)
                    all_is_shuffled = (
                        is_shuffled if all_is_shuffled is None else np.concatenate(
                            (all_is_shuffled, is_shuffled), axis=0)
                    )
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if sequence_losses_host is not None:
                    sequence_losses = nested_numpify(sequence_losses_host)
                    all_sequence_losses = (
                        sequence_losses
                        if all_sequence_losses is None
                        else np.concatenate((all_sequence_losses, sequence_losses), axis=0)
                    )
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if inputs_host is not None:
                    inputs_decode = nested_numpify(inputs_host)
                    all_inputs = (
                        inputs_decode
                        if all_inputs is None
                        else nested_concat(all_inputs, inputs_decode, padding_index=-100)
                    )
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )
                # Set back to None to begin a new accumulation
                (is_shuffled_host, losses_host, sequence_losses_host, preds_host, inputs_host, labels_host) = (
                    None, None, None, None, None, None)

        # After all calls to `.gather_function`, reset to `gather_for_metrics`:
        self.gather_function = self.accelerator.gather_for_metrics
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if is_shuffled_host is not None:
            is_shuffled = nested_numpify(is_shuffled_host)
            all_is_shuffled = (
                is_shuffled if all_is_shuffled is None else np.concatenate(
                    (all_is_shuffled, is_shuffled), axis=0)
            )
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate(
                (all_losses, losses), axis=0)
        if sequence_losses_host is not None:
            sequence_losses = nested_numpify(sequence_losses_host)
            all_sequence_losses = (
                sequence_losses
                if all_sequence_losses is None
                else nested_concat(all_sequence_losses, sequence_losses, padding_index=-100)
            )
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
            all_inputs = (
                inputs_decode if all_inputs is None else nested_concat(all_inputs, inputs_decode, padding_index=-100)
            )
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        print(f'compute_discourse_metrics: {self.compute_discourse_metrics}')

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            if self.compute_discourse_metrics is not None:
                metrics = self.compute_discourse_metrics(
                    EvalPrediction(
                        predictions=all_preds,
                        label_ids=all_labels,
                        inputs=all_inputs,
                    ),
                    self.tokenizer,
                    all_sequence_losses,
                )
            else:
                metrics = self.compute_metrics(
                    EvalPrediction(
                        predictions=all_preds,
                        label_ids=all_labels,
                    ),
                )
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def depth_prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on 'model' using 'inputs'.

        :param model: The model to evaluate.
        :param inputs: The inputs and targets of the model. The dictionary will be unpacked before being fed to the
            model. Most models expect the targets under the argument 'labels'. Check your model's documentation for all
            accepted
        :param prediction_loss_only: Whether or not to return the loss only.
        :param ignore_keys: A list of keys in the output of 'model.forward()' to ignore when gathering predictions.
        :return: A tuple containing the loss, sequence_losses, logits and labels (each being optional).
        """
        inputs = self._prepare_inputs(inputs)

        labels = nested_detach(inputs.get(constants.DepthDataCollatorConstants.LABELS, None))

        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            if isinstance(outputs, t5_model.HierarchicalSeq2SeqLMOutput):
                loss = outputs.get('loss', None)
                sequence_losses = outputs.get('sequence_losses', None)
                logits = outputs.get('logits', None)
            else:
                raise RuntimeError(f'Model outputted an instance of unexpected type {type(outputs)}')

            if prediction_loss_only:
                return loss, None, None, None

            logits = nested_detach(logits)
            sequence_losses = nested_detach(sequence_losses)

            return loss, sequence_losses, logits, labels
