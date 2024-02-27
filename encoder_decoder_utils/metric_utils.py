import typing
import nltk
import numpy as np
import torch
import transformers
import evaluate
from encoder_decoder_utils import constants, tokenizer_utils
import random
from encoder_decoder_utils.constants import Metric


# TODO: Add metrics specifically for DEPTH
def compute_metrics_depth(
        eval_preds: transformers.trainer_utils.EvalPrediction,
        tokenizer: transformers.PreTrainedTokenizer,
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

    average_non_padding_tokens_per_example_input = np.not_equal(inputs, tokenizer.pad_token_id).sum(axis=1).mean()
    # -100 is the ignore index within the labels. It is used to ignore padding tokens when computing the loss.
    average_non_padding_tokens_per_example_label = np.not_equal(labels, -100).sum(
        axis=1).mean()  # TODO: Use constant instead of magic number
    num_non_padding_tokens_in_batch_input = np.not_equal(inputs, tokenizer.pad_token_id).sum(axis=1).sum()
    num_non_padding_tokens_in_batch_label = np.not_equal(labels, tokenizer.pad_token_id).sum(axis=1).sum()

    # Sentence tokens include <sent_i> (where i is an integer) tokens as well as <eosen> tokens.
    sentence_tokens = list(
        filter(lambda token: f'<{constants.DEPTHTokenizerConstants.SENT}' in token, tokenizer.all_special_tokens))
    sentence_tokens.append(constants.DEPTHTokenizerConstants.END_OF_SENTENCE_TOKEN)

    sentence_token_ids = tokenizer.convert_tokens_to_ids(sentence_tokens)
    sentence_token_ids = np.array(sentence_token_ids, dtype=np.int32)
    is_sentence_token = np.isin(labels, sentence_token_ids)

    # Find average loss on each example, and then the average loss across examples in the batch
    sentence_losses = np.ma.masked_array(
        sequence_losses,
        np.logical_not(is_sentence_token),  # Mask out non-sentence tokens
    )
    average_loss_on_sentence_tokens = sentence_losses.mean()
    variance_loss_on_sentence_tokens = sentence_losses.var()
    is_padding_token = np.equal(labels,
                                -100)  # -100 is the ignore index within the labels  # TODO: Use constant instead of magic number
    non_sentence_losses = np.ma.masked_array(
        sequence_losses,
        np.logical_or(is_sentence_token, is_padding_token)  # Mask out sentence, padding, and eos tokens
    )
    average_loss_on_non_sentence_tokens = non_sentence_losses.mean()
    variance_loss_on_non_sentence_tokens = non_sentence_losses.var()
    metrics.update({
        constants.Metric.AVERAGE_NON_PADDING_TOKENS_PER_EXAMPLE_INPUT.value:
            average_non_padding_tokens_per_example_input,
        constants.Metric.AVERAGE_NON_PADDING_TOKENS_PER_EXAMPLE_LABEL.value:
            average_non_padding_tokens_per_example_label,
        constants.Metric.NUM_NON_PADDING_TOKENS_IN_BATCH_INPUT.value: num_non_padding_tokens_in_batch_input,
        constants.Metric.NUM_NON_PADDING_TOKENS_IN_BATCH_LABEL.value: num_non_padding_tokens_in_batch_label,
        constants.Metric.AVERAGE_LOSS_ON_SENTENCE_TOKENS.value: average_loss_on_sentence_tokens,
        constants.Metric.VARIANCE_LOSS_ON_SENTENCE_TOKENS.value: variance_loss_on_sentence_tokens,
        constants.Metric.AVERAGE_LOSS_ON_NON_SENTENCE_TOKENS.value: average_loss_on_non_sentence_tokens,
        constants.Metric.VARIANCE_LOSS_ON_NON_SENTENCE_TOKENS.value: variance_loss_on_non_sentence_tokens,
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

    batch_size = predictions.shape[0]
    target_length = predictions.shape[1]
    sentinel_tokens = tokenizer.get_sentinel_token_ids()

    average_non_padding_tokens_per_example_input = np.not_equal(
        input_ids, tokenizer.pad_token_id).mean()
    average_non_padding_tokens_per_example_label = np.not_equal(labels, -100).mean()

    input_id_sentinel_tokens = np.isin(input_ids, sentinel_tokens)
    target_id_sentinel_tokens = np.isin(labels, sentinel_tokens)
    target_padding_tokens = np.equal(labels, -100)

    metrics = {
        Metric.AVERAGE_NON_PADDING_TOKENS_PER_EXAMPLE_INPUT.value: average_non_padding_tokens_per_example_input,
        Metric.AVERAGE_NON_PADDING_TOKENS_PER_EXAMPLE_LABEL.value: average_non_padding_tokens_per_example_label,
    }

    if isinstance(tokenizer, tokenizer_utils.DepthTokenizer):
        sentence_token_ids = tokenizer.get_sentence_token_ids()
        target_id_sentence_tokens = np.isin(labels, sentence_token_ids)

        prediction_is_correct = np.equal(predictions, labels)
        sentence_accuracy = np.ma.masked_array(
            prediction_is_correct,
            np.logical_not(target_id_sentence_tokens),  # Mask out non-sentence tokens
        ).mean()
        reconstruction_accuracy = np.ma.masked_array(
            prediction_is_correct,
            np.logical_or(target_id_sentence_tokens, target_padding_tokens)
            # Mask out sentence, padding, and eos tokens
        ).mean()
        metrics[Metric.SENTENCE_ACCURACY.value] = sentence_accuracy
        metrics[Metric.RECONSTRUCTION_ACCURACY.value] = reconstruction_accuracy

    # Flatten the predictions and labels from (batch_size, target_length) to (batch_size * target_length)
    predictions = predictions.reshape([batch_size * target_length])
    labels = labels.reshape([batch_size * target_length])

    # TODO: Make sure accuracy does not account for padding tokens.
    # flattened_target_padding_tokens = target_padding_tokens.reshape([batch_size * target_length])
    # accuracy = evaluate.load(
    #     "accuracy",
    #     sample_weight=np.logical_not(flattened_target_padding_tokens, dtype=np.int8))
    clf_metrics = evaluate.combine(
        [
            constants.Metric.ACCURACY.value,
        ]
    )
    metrics.update(clf_metrics.compute(predictions=predictions, references=labels))
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


# TODO: Use the following code to compute metrics for seq2seq model during evaluation
# METRIC_NAME_TO_FUNC = {
#     'accuracy': accuracy_score,
#     'f1': lambda labels, prediction: f1_score(
#         y_true=labels,
#         y_pred=prediction,
#         average='micro',
#     ),
#     'precision': lambda labels, prediction: precision_score(
#         y_true=labels,
#         y_pred=prediction,
#         average='micro',
#     ),
#     'recall': lambda labels, prediction: recall_score(
#         y_true=labels,
#         y_pred=prediction,
#         average='micro',
#     ),
#     'mcc': matthews_corrcoef,
#     # 'pearson': stats.pearsonr,
#     # 'spearman': stats.spearmanr,
# }

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    return preds, labels


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
    # Replace -100s used for padding as we can't decode them
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    # TODO: use a constant

    label_to_id = {label: idx for idx, label in ft_constants[dataset].LABELS.items() if
                   label != ft_constants[dataset].OTHER}
    possible_labels = set(label_to_id.keys()) - {'other'}
    predictions_converted = []
    labels_converted = []
    for pred, label in zip(decoded_preds, decoded_labels):
        # TODO: use a constant
        if dataset == 'stsb':
            labels_converted.append(round(float(label) / 0.2) * 0.2)
            try:
                predictions_converted.append(round(float(pred) / 0.2) * 0.2)
            except Exception:
                predictions_converted.append(random.sample([0.0, 5.0], 1)[0])
        else:
            if pred in label_to_id.keys():
                predictions_converted.append(label_to_id[pred])
            else:
                # In the case that the model predicts a label that is not in the possible labels, we will randomly select
                # another label from the possible labels that isn't the correct label.
                other_labels = list(possible_labels - {pred})
                wrong_label = random.sample(other_labels, 1)[0]
                predictions_converted.append(label_to_id[wrong_label])
                # try:
                #     predictions_converted.append(label_to_id[ft_constants.OTHER])
                # except Exception:
                #     predictions_converted.append(-1)
                labels_converted.append(label_to_id[label])
    return metric_fn.compute(
        predictions=predictions_converted,
        references=labels_converted,
    )

# TODO: Use the following code to compute metrics for seq2seq model during evaluation
# def compute_metrics(
#         eval_pred: transformers.EvalPrediction,
#         metric_names: typing.List[str],
#         tokenizer: transformers.PreTrainedTokenizer,
# ) -> typing.Dict[str, float]:
#     """Compute the accuracy of the model.
#
#     Args:
#         eval_pred: A namedtuple containing the model predictions and labels.
#         metric_names: The names of the metrics to be used for evaluation on a benchmark task.
#         tokenizer: The tokenizer used to encode the inputs and labels.
#
#     Returns:
#         A dictionary containing the accuracy of the model.
#     """
#     predictions, labels = eval_pred
#     predictions: np.ndarray  # Shape is [batch_size, target_sequence_length]
#     labels: np.ndarray       # Shape is [batch_size, target_sequence_length]
#     metrics = {}
#     labels[labels == -100] = tokenizer.pad_token_id
#
#     if predictions[:, 0].max() == tokenizer.pad_token_id:  # Check if the first token in the predictions is the padding token
#         # Skip the first token in the predictions (i.e., the decoder start token), and add a padding token at the end
#         predictions = np.concatenate(
#             [predictions[:, 1:],
#              np.full(
#                  (predictions.shape[0], 1),
#                  tokenizer.pad_token_id)
#              ],
#             axis=1,
#         )
#
#     is_correct = np.equal(predictions, labels)
#     num_correct_per_example = is_correct.sum(axis=1)
#     ideal_num_correct_per_example = np.ones_like(num_correct_per_example) * labels.shape[1]
#     example_is_correct = np.equal(num_correct_per_example, ideal_num_correct_per_example)
#
#     predictions = predictions[(labels != tokenizer.pad_token_id) & (labels != tokenizer.eos_token_id)]
#     labels = labels[(labels != tokenizer.pad_token_type_id) & (labels != tokenizer.eos_token_id)]
#
#     # Get the metrics!
#     for metric_name in metric_names:
#         # Metrics from scipy return `statistic` and `pvalue`, but we are only interested in the statistic.
#         if metric_name == 'pearson' or metric_name == 'spearman':
#             # Get the statistic (not the pvalue)
#             metrics[f'token_{metric_name}'] = METRIC_NAME_TO_FUNC[metric_name](labels, predictions)[0]
#             metrics[f'example_{metric_name}'] = METRIC_NAME_TO_FUNC[metric_name](
#                 example_is_correct, np.ones_like(example_is_correct))[0]
#         # Multiply mcc by 100 to remain consistent with the original T5 implementation:
#         # https://github.com/google-research/text-to-text-transfer-transformer/blob/main/t5/data/glue_utils.py#L140
#         elif metric_name == 'mcc':
#             metrics[f'token_{metric_name}'] = METRIC_NAME_TO_FUNC[metric_name](labels, predictions) * 100
#             metrics[f'example_{metric_name}'] = METRIC_NAME_TO_FUNC[metric_name](
#                 example_is_correct, np.ones_like(example_is_correct)) * 100
#         else:
#             metrics[f'token_{metric_name}'] = METRIC_NAME_TO_FUNC[metric_name](labels, predictions)
#             metrics[f'example_{metric_name}'] = METRIC_NAME_TO_FUNC[metric_name](
#                 example_is_correct, np.ones_like(example_is_correct))
#     return metrics
