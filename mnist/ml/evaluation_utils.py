import json

from tqdm import tqdm

import mnist.config as config
import mnist.ml.training_utils as train_utils
from mnist.custom_utils.logger import DISABLE_PROGRESS_BAR
from mnist.custom_utils.logger import PROGRESS_BAR_PREFIX


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
    for _batch_idx, batch_loss, batch_tp, batch_size in batches:
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

    desc = '{} Evaluating'.format(PROGRESS_BAR_PREFIX)
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


def save_evaluation_results(eval_results_path, eval_results):
    """Saves the evaluation results on a file on disk.

    The evaluation results are stored as a list of dictionaries. Every time
    a new result is to be saved, it is appended to the existing list.

    Args:
        eval_results_path (str): Path to the evaluation results file.
        eval_results (dict): Dict of results to store.
    """
    try:
        # Try to open the evaluation list from an existing file.
        with open(eval_results_path, 'r') as f:
            evaluations_list = json.load(f)
    except FileNotFoundError:
        # If the file does not exist yet, create a new list.
        evaluations_list = []

    evaluations_list.append(eval_results)

    with open(eval_results_path, 'w+') as f:
        json.dump(evaluations_list, f, indent=4)
