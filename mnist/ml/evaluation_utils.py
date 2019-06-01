import json

from tqdm import tqdm

import mnist.config as config
import mnist.ml.training_utils as train_utils
from mnist.custom_utils.logger import DISABLE_PROGRESS_BAR
from mnist.custom_utils.logger import LOGGER_LEVEL_NAME


def evaluation_accumulator(batches):
    """Runs the input batch generator and accumulates the evaluation
    results.

    In this implementation, the `batches` iterable must yield tuples like:
        (
            <0-based index of the batch>,
            <batch loss>,
            <number of true positives the batch>,
            <size of the batch>,
        ).

    Args:
        batches: An iterable of compatible tuples.

    Returns:
        A tuple (<average loss>, <accuracy>).
    """
    true_positives = 0
    tot_batch_loss = 0
    num_samples = 0
    for batch_idx, batch_loss, batch_tp, batch_size in batches:
        tot_batch_loss += batch_loss
        true_positives += batch_tp
        num_samples += batch_size

    avg_loss = tot_batch_loss / num_samples
    accuracy = true_positives / num_samples

    return avg_loss, accuracy


def run_evaluation(evaluation_engine, dataset_def_path, dataset_name=None):
    """Evaluates the best trained model on a test set.

    Args:
        evaluation_engine (:obj:`EvaluationEngine`): EvaluationEngine instance
            to run to evaluate the model on the dataset.
        dataset_def_path (str): Path to the dataset definition.
        dataset_name (str, optional): Identifier associated to the current
            dataset. Defaults to None.

    Returns:
        A tuple (<average loss>, <accuracy>).
    """
    with open(dataset_def_path, 'r') as f:
        dataset_def = json.load(f)

    num_batches = train_utils.batches_per_epoch(
        dataset_size=len(dataset_def),
        batch_size=config.ExperimentConfig.BATCH_SIZE_TEST,
        drop_last=False)

    desc = '[{}] Evaluating'.format(LOGGER_LEVEL_NAME)
    if dataset_name:
        desc += ' on {}'.format(dataset_name)
    pbar = tqdm(
        evaluation_engine.evaluate_best_model_on_dataset(dataset_def),
        desc=desc,
        total=num_batches,
        leave=True,
        disable=DISABLE_PROGRESS_BAR)

    avg_loss, accuracy = evaluation_accumulator(pbar)
    return avg_loss, accuracy
