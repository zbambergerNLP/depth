import typing
import nltk
import numpy as np
import torch
import transformers
import evaluate
import random
from encoder_decoder_utils import constants, tokenizer_utils
from encoder_decoder_utils.constants import Metric


def postprocess_text(
        preds: typing.List[str],
        labels: typing.List[str],
) -> typing.Tuple[typing.List[str], typing.List[str]]:
    """
    Postprocess the predictions and labels to ensure that they are in the correct format for computing metrics.
    :param preds: The predictions from the model, decoded from the logits via the tokenizer.
    :param labels: The labels for the model, decoded from the label ids via the tokenizer.
    :return: A 2-tuple containing the postprocessed predictions and labels. The postprocessed predictions and labels
        should be in the correct format for computing metrics.
    """
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    return preds, labels

# TODO: Add metrics specifically for DEPTH
def compute_metrics_depth(
        eval_preds: transformers.trainer_utils.EvalPrediction,
        tokenizer: tokenizer_utils.DepthTokenizer,
        sequence_losses: np.ndarray,  # A tensor of shape (batch_size, sequence_length)
) -> typing.Mapping[str, float]:
    """
    Return a collection of evaluation metrics given a DiscourseT5 model.

    :param eval_preds: A 2-tuple of the form [logits, labels]. Labels is a collection of integers representing the label
        of the input. Logits is a collection of tensors corresponding to the model's logits for each input in the batch.
        It is possible for eval_preds to contain a third item in the tuple, which is a collection of tensors
        corresponding to the model's input ids for each input in the batch.
    :param tokenizer: The tokenizer used to encode the inputs and targets.
    :param sequence_losses: A tensor of shape (batch_size, sequence_length) containing the loss for each token in the
        target sequence. Used to decompose the loss into different components (e.g. sentence-level loss, word-level
        loss, etc.).
    :return: A dictionary of metrics (mapping string metric names to float metric values).
    """
    predictions: np.ndarray = eval_preds.predictions
    labels: np.ndarray = eval_preds.label_ids
    inputs: np.ndarray = eval_preds.inputs if eval_preds.inputs is not None else None
    metrics = {}

    # Sentence tokens include <sent_i> (where i is an integer) tokens as well as <eosen> tokens.
    sentence_token_ids = np.array(tokenizer.get_sentence_token_and_eosen_ids())
    is_sentence_token = np.isin(labels, sentence_token_ids)

    # Find average loss on each example, and then the average loss across examples in the batch
    sentence_losses = np.ma.masked_array(
        sequence_losses,
        np.logical_not(is_sentence_token),  # Mask out non-sentence tokens
    )
    average_loss_on_sentence_tokens = sentence_losses.mean()
    is_padding_token = np.equal(labels, -100)  # -100 is the ignore index within the labels  # TODO: Use constant instead of magic number
    non_sentence_losses = np.ma.masked_array(
        sequence_losses,
        np.logical_or(is_sentence_token, is_padding_token)  # Mask out sentence, padding, and eos tokens
    )
    average_loss_on_non_sentence_tokens = non_sentence_losses.mean()

    prediction_is_correct = np.equal(predictions, labels)
    sentence_accuracy = np.ma.masked_array(
        prediction_is_correct,
        np.logical_not(is_sentence_token),  # Mask out non-sentence tokens
    ).mean()

    reconstruction_accuracy = np.ma.masked_array(
        prediction_is_correct,
        np.logical_or(is_sentence_token, is_padding_token)  # Mask out sentence and padding tokens
    ).mean()

    sentence_tokens_per_example = np.sum(is_sentence_token, axis=1).mean()

    metrics.update({
        constants.Metric.AVERAGE_LOSS_ON_SENTENCE_TOKENS.value: average_loss_on_sentence_tokens,
        constants.Metric.AVERAGE_LOSS_ON_NON_SENTENCE_TOKENS.value: average_loss_on_non_sentence_tokens,
        constants.Metric.SENTENCE_ACCURACY.value: sentence_accuracy,
        constants.Metric.RECONSTRUCTION_ACCURACY.value: reconstruction_accuracy,
        constants.Metric.NUM_SENTENCE_TOKENS.value: sentence_tokens_per_example,
    })
    return metrics


def compute_metrics(
        eval_preds: transformers.EvalPrediction,
        tokenizer: typing.Union[transformers.PreTrainedTokenizer, tokenizer_utils.DepthTokenizer],
) -> typing.Dict[str, float]:
    """
    Compute the metrics for the evaluation set. This function is called after every evaluation step.

    :param eval_preds: The evaluation predictions. This is an EvalPrediction object, which contains the
        predictions, labels, and sometimes the input ids as well (if specified in the training arguments).
    :param tokenizer: The tokenizer used to encode the inputs and labels.
    :return: A dictionary of the metrics. The keys are the metric names, and the values are the metric values.
    """
    predictions = eval_preds.predictions
    labels = eval_preds.label_ids
    input_ids = eval_preds.inputs

    target_padding_tokens = np.equal(labels, -100)
    prediction_is_correct = np.equal(predictions, labels)
    sentinel_token_ids = np.array(tokenizer.get_sentinel_token_ids())

    sentinel_tokens_in_labels = np.isin(labels, sentinel_token_ids)
    padding_tokens_in_labels = np.equal(labels, tokenizer.pad_token_id)
    non_padding_tokens_in_labels = np.not_equal(labels, tokenizer.pad_token_id)

    # TODO: Determine the number of tokens in between consecutive sentinel tokens

    metrics = {
        Metric.NUM_SENTINEL_TOKENS_IN_LABELS.value: np.sum(sentinel_tokens_in_labels, axis=1).mean(),
        Metric.PADDING_TOKENS_IN_LABELS.value: np.sum(padding_tokens_in_labels, axis=1).mean(),
        Metric.NON_PADDING_TOKENS_IN_LABELS.value: np.sum(non_padding_tokens_in_labels, axis=1).mean(),
    }

    if input_ids is not None:
        sentinel_tokens_in_inputs = np.isin(input_ids, sentinel_token_ids)
        padding_tokens_in_inputs = np.equal(input_ids, tokenizer.pad_token_id)
        non_padding_tokens_in_inputs = np.not_equal(input_ids, tokenizer.pad_token_id)
        metrics.update({
            Metric.NUM_SENTINEL_TOKENS_IN_INPUTS.value: np.sum(sentinel_tokens_in_inputs, axis=1).mean(),
            Metric.PADDING_TOKENS_IN_INPUTS.value: np.sum(padding_tokens_in_inputs, axis=1).mean(),
            Metric.NON_PADDING_TOKENS_IN_INPUTS.value: np.sum(non_padding_tokens_in_inputs, axis=1).mean(),
        })

    if isinstance(tokenizer, tokenizer_utils.DepthTokenizer):
        sentence_token_ids = tokenizer.get_sentence_token_ids()
        target_id_sentence_tokens = np.isin(labels, sentence_token_ids)
        sentence_accuracy = np.ma.masked_array(
            prediction_is_correct,
            np.logical_not(target_id_sentence_tokens),  # Mask out non-sentence tokens
        ).mean()
        reconstruction_accuracy = np.ma.masked_array(
            prediction_is_correct,
            # Mask out sentence and padding tokens
            np.logical_or(target_id_sentence_tokens, target_padding_tokens)
        ).mean()
        metrics[Metric.SENTENCE_ACCURACY.value] = sentence_accuracy
    else:
        reconstruction_accuracy = np.ma.masked_array(
            np.equal(predictions, labels),
            # Mask out padding tokens
            target_padding_tokens
        ).mean()
    metrics[Metric.RECONSTRUCTION_ACCURACY.value] = reconstruction_accuracy
    return metrics


def preprocess_logits_for_metrics(
        logits: torch.Tensor,  # (batch_size, target_length, vocab_size)
        labels: torch.Tensor,  # (batch_size, target_length)
):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_fine_tune_metrics(
        eval_preds: transformers.EvalPrediction,
        metric: str = None,
        benchmark: str = None,
        dataset: str = None,
        ft_constants: typing.Any = None,
        tokenizer: transformers.PreTrainedTokenizer = None,
) -> typing.Dict[str, float]:
    """
    Computes fine-tuning metrics for a given set of evaluation predictions.

    This function takes as input evaluation predictions, and optionally a metric, benchmark, and dataset.
    It then computes the specified metric for the given predictions. If no metric is specified,
    it loads the appropriate metric based on the provided benchmark and dataset.

    Args:
        eval_preds (transformers.EvalPrediction): The evaluation predictions to compute metrics for.
        metric (str, optional): The name of the metric to compute. If not provided, a metric is loaded based on the benchmark and dataset.
        benchmark (str, optional): The name of the benchmark to use for loading the metric, if no metric is provided.
        dataset (str, optional): The name of the dataset to use for loading the metric, if no metric is provided.
        ft_constants (typing.Any): The fine-tuning constants for the model.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer used to encode the inputs and labels.
    Returns:
        dict[str, float]: A dictionary mapping metric names to their computed values.

    Raises:
        ValueError: If no metric is provided and either benchmark or dataset is None.
    """
    if metric is not None:
        metric_fn = evaluate.load(metric)
    elif benchmark is not None and dataset is not None:
        metric_fn = evaluate.load(benchmark, dataset)
    else:
        raise ValueError("Either a metric or both a benchmark and dataset must be provided.")
    preds, labels = eval_preds
    if len(preds.shape) == 1:
        preds = preds.reshape(-1, 1)

    # Replace -100s used for padding as we can't decode them
    preds = np.where(labels != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    # Create a dictionary to map the label to the prediction id.
    label_to_id = {label: idx for idx, label in ft_constants[dataset].LABELS.items()}
    possible_labels = set(label_to_id.keys())
    predictions_converted = []
    labels_converted = []
    for pred, label in zip(decoded_preds, decoded_labels):

        if dataset == constants.GLUEConstants.STS_B:
            possible_labels = set(label_to_id.keys())
            labels_converted.append(round(float(label) / 0.2) * 0.2)
            try:
                # If the prediction is a number, we round it to the nearest 0.2, as was proposed in the T5 paper.
                predictions_converted.append(round(float(pred) / 0.2) * 0.2)
            except ValueError:
                # If the prediction is not a number, we select the label that is farthest from the correct label.
                # This is done to ensure that the metric is as low as possible so as to convey that the model is
                # not performing well.
                predictions_converted.append(5.0 if float(label) < 2.5 else 0.0)

        else:
            # In classification datasets (excluding STS-B), we remove the "unknown" label from the possible labels
            #  (the unknown label corresponds with the ID -1, whereas other labels start at 0).
            try:
                possible_labels -= {ft_constants[dataset].LABELS[-1]}
            except KeyError:
                pass

            if pred in label_to_id.keys():
                predictions_converted.append(label_to_id[pred])
            else:
                # In the case that the model predicts a label that is not in the possible labels, we will randomly
                # select another label from the possible labels that isn't the correct label.
                other_labels = list(possible_labels - {label})
                wrong_label = random.sample(other_labels, 1)[0]
                predictions_converted.append(label_to_id[wrong_label])
            labels_converted.append(label_to_id[label])

    return metric_fn.compute(
        predictions=predictions_converted,
        references=labels_converted,
    )
