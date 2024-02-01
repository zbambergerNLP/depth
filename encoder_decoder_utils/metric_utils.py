import typing

import numpy as np
import torch
from sklearn.metrics import precision_score, f1_score, recall_score, matthews_corrcoef, accuracy_score
import transformers
from torch.nn import CrossEntropyLoss

from encoder_decoder_utils.constants import Metric, DepthMetric


def trainer_compute_metrics(
        eval_pred: transformers.EvalPrediction,
        tokenizer: transformers.PreTrainedTokenizer,
) -> typing.Dict[str, float]:
    labels = eval_pred.label_ids
    logits = eval_pred.predictions
    sequence_loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
    sequence_losses = sequence_loss_fct(
        logits.view(-1, logits.size(-1)),
        labels.view(-1)
    ).reshape(labels.shape)
    return compute_loss_metrics(
        input_ids=eval_pred.input_ids,
        labels=labels,
        sequence_losses=sequence_losses,
        pad_token_id=eval_pred.label_ids[0, 0],
        sentence_token_ids=None,
    )


def compute_loss_metrics(
        input_ids: np.ndarray,  # Shape: (batch_size, input_length
        labels: np.ndarray,  # Shape: (batch_size, target_length)
        sequence_losses: np.ndarray,  # Shape: (batch_size, target_length)
        pad_token_id: int,
        sentence_token_ids: typing.List[int] = None,
) -> typing.Dict[str, float]:
    """
    Compute loss-related metrics on the given batch.

    :param input_ids: A tensor of shape (batch_size, input_length) containing the input ids.
    :param labels: A tensor of shape (batch_size, target_length) containing the labels.
    :param sequence_losses: A tensor of shape (batch_size, target_length) containing the loss on each token.
    :param pad_token_id: The id of the padding token.
    :param sentence_token_ids: When using a Depth model, this is a list of ids of sentence tokens. Sentence tokens are
        tokens which are the first token in a sentence. Additionally, there is a sentence token at the end of the
        sequence (<eosen>).
    :return: A dictionary mapping from metric name to metric value.
    """
    stats = {}
    average_non_padding_tokens_per_example_input = np.not_equal(input_ids, pad_token_id).sum(axis=1).mean()
    variance_non_padding_tokens_per_example_input = np.not_equal(input_ids, pad_token_id).sum(axis=1).var()
    # -100 is the ignore index within the labels. It is used to ignore padding tokens when computing the loss.
    average_non_padding_tokens_per_example_label = np.not_equal(labels, -100).sum(axis=1).mean()
    variance_non_padding_tokens_per_example_label = np.not_equal(labels, -100).sum(axis=1).var()
    num_non_padding_tokens_in_batch_input = np.not_equal(input_ids, pad_token_id).sum(axis=1).sum()
    num_non_padding_tokens_in_batch_label = np.not_equal(labels, pad_token_id).sum(axis=1).sum()
    stats[Metric.AVERAGE_NON_PADDING_TOKENS_PER_EXAMPLE_INPUT.value] = average_non_padding_tokens_per_example_input
    stats[Metric.VARIANCE_NON_PADDING_TOKENS_PER_EXAMPLE_INPUT.value] = variance_non_padding_tokens_per_example_input
    stats[Metric.AVERAGE_NON_PADDING_TOKENS_PER_EXAMPLE_LABEL.value] = average_non_padding_tokens_per_example_label
    stats[Metric.VARIANCE_NON_PADDING_TOKENS_PER_EXAMPLE_LABEL.value] = variance_non_padding_tokens_per_example_label
    stats[Metric.NUM_NON_PADDING_TOKENS_IN_BATCH_INPUT.value] = num_non_padding_tokens_in_batch_input
    stats[Metric.NUM_NON_PADDING_TOKENS_IN_BATCH_LABEL.value] = num_non_padding_tokens_in_batch_label

    if sentence_token_ids:
        is_sentence_token = np.isin(labels, sentence_token_ids)

        # Find average loss on each example, and then the average loss across examples in the batch
        sentence_losses = np.ma.masked_array(
            sequence_losses,
            np.logical_not(is_sentence_token),  # Mask out non-sentence tokens
        )
        average_loss_on_sentence_tokens = sentence_losses.mean()
        variance_loss_on_sentence_tokens = sentence_losses.var()
        is_padding_token = np.equal(labels, -100)  # -100 is the ignore index within the labels
        non_sentence_losses = np.ma.masked_array(
            sequence_losses,
            np.logical_or(is_sentence_token, is_padding_token)  # Mask out sentence, padding, and eos tokens
        )
        average_loss_on_non_sentence_tokens = non_sentence_losses.mean()
        variance_loss_on_non_sentence_tokens = non_sentence_losses.var()

        stats[DepthMetric.AVERAGE_LOSS_ON_SENTENCE_TOKENS.value] = average_loss_on_sentence_tokens
        stats[DepthMetric.VARIANCE_LOSS_ON_SENTENCE_TOKENS.value] = variance_loss_on_sentence_tokens
        stats[DepthMetric.AVERAGE_LOSS_ON_NON_SENTENCE_TOKENS.value] = average_loss_on_non_sentence_tokens
        stats[DepthMetric.VARIANCE_LOSS_ON_NON_SENTENCE_TOKENS.value] = variance_loss_on_non_sentence_tokens
    return stats

METRIC_NAME_TO_FUNC = {
    'accuracy': accuracy_score,
    'f1': lambda labels, prediction: f1_score(
        y_true=labels,
        y_pred=prediction,
        average='micro',
    ),
    'precision': lambda labels, prediction: precision_score(
        y_true=labels,
        y_pred=prediction,
        average='micro',
    ),
    'recall': lambda labels, prediction: recall_score(
        y_true=labels,
        y_pred=prediction,
        average='micro',
    ),
    'mcc': matthews_corrcoef,
    # 'pearson': stats.pearsonr,
    # 'spearman': stats.spearmanr,
}


def compute_metrics(
        eval_pred: transformers.EvalPrediction,
        metric_names: typing.List[str],
        tokenizer: transformers.PreTrainedTokenizer,
) -> typing.Dict[str, float]:
    """Compute the accuracy of the model.

    Args:
        eval_pred: A namedtuple containing the model predictions and labels.
        metric_names: The names of the metrics to be used for evaluation on a benchmark task.
        tokenizer: The tokenizer used to encode the inputs and labels.

    Returns:
        A dictionary containing the accuracy of the model.
    """
    predictions, labels = eval_pred
    predictions: np.ndarray  # Shape is [batch_size, target_sequence_length]
    labels: np.ndarray       # Shape is [batch_size, target_sequence_length]
    metrics = {}
    labels[labels == -100] = tokenizer.pad_token_id

    if predictions[:, 0].max() == tokenizer.pad_token_id:  # Check if the first token in the predictions is the padding token
        # Skip the first token in the predictions (i.e., the decoder start token), and add a padding token at the end
        predictions = np.concatenate(
            [predictions[:, 1:],
             np.full(
                 (predictions.shape[0], 1),
                 tokenizer.pad_token_id)
             ],
            axis=1,
        )

    is_correct = np.equal(predictions, labels)
    num_correct_per_example = is_correct.sum(axis=1)
    ideal_num_correct_per_example = np.ones_like(num_correct_per_example) * labels.shape[1]
    example_is_correct = np.equal(num_correct_per_example, ideal_num_correct_per_example)

    predictions = predictions[(labels != tokenizer.pad_token_id) & (labels != tokenizer.eos_token_id)]
    labels = labels[(labels != tokenizer.pad_token_type_id) & (labels != tokenizer.eos_token_id)]

    # Get the metrics!
    for metric_name in metric_names:
        # Metrics from scipy return `statistic` and `pvalue`, but we are only interested in the statistic.
        if metric_name == 'pearson' or metric_name == 'spearman':
            # Get the statistic (not the pvalue)
            metrics[f'token_{metric_name}'] = METRIC_NAME_TO_FUNC[metric_name](labels, predictions)[0]
            metrics[f'example_{metric_name}'] = METRIC_NAME_TO_FUNC[metric_name](
                example_is_correct, np.ones_like(example_is_correct))[0]
        # Multiply mcc by 100 to remain consistent with the original T5 implementation:
        # https://github.com/google-research/text-to-text-transfer-transformer/blob/main/t5/data/glue_utils.py#L140
        elif metric_name == 'mcc':
            metrics[f'token_{metric_name}'] = METRIC_NAME_TO_FUNC[metric_name](labels, predictions) * 100
            metrics[f'example_{metric_name}'] = METRIC_NAME_TO_FUNC[metric_name](
                example_is_correct, np.ones_like(example_is_correct)) * 100
        else:
            metrics[f'token_{metric_name}'] = METRIC_NAME_TO_FUNC[metric_name](labels, predictions)
            metrics[f'example_{metric_name}'] = METRIC_NAME_TO_FUNC[metric_name](
                example_is_correct, np.ones_like(example_is_correct))
    return metrics


def preprocess_logits_for_metrics(
        logits: torch.Tensor,  # Shape is [batch_size, target_sequence_length, vocab_size]
        labels: torch.Tensor,  # Shape is [batch_size, target_sequence_length]
) -> torch.Tensor:
    """
    Original Trainer may have a memory leak.

    This is a workaround to avoid storing too many tensors that are not needed (which may cause a memory leak).

    Args:
        logits: The logits output by the model.
        labels: The labels for the model.

    Returns:
        The predictions of the model (i.e., the argmax of the logits). Shape is [batch_size, target_sequence_length].
    """
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]

    return logits.argmax(dim=-1)