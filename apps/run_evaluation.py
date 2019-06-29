import time

import mnist.constants as constants
import mnist.ml.engines.evaluation_engine as eval_eng
import mnist.ml.evaluation_utils as eval_utils
import mnist.paths as paths

_CNST = constants.Constants


def main():
    dataset_name = _CNST.TEST_SET_NAMES[0]
    dataset_def_path = paths.DatasetDefinitions.TEST[dataset_name]

    evaluation_engine = eval_eng.EvaluationEngine()

    avg_loss, accuracy = eval_utils.run_evaluation(
        evaluation_engine=evaluation_engine,
        dataset_def_path=dataset_def_path,
        dataset_name=dataset_name
    )

    print('Average loss: {:.3f}\n'
          'Accuracy: {:.3f}\n'.format(avg_loss, accuracy))

    # Append the new evaluation results.
    eval_results = {
        _CNST.EVAL_RESULTS_TIMESTAMP_KEY: time.strftime('%a %d %b %Y %H:%M:%S'),
        _CNST.EVAL_RESULTS_DATASET_KEY: dataset_name,
        'avg_loss': avg_loss,
        'accuracy': accuracy,
    }
    eval_utils.save_evaluation_results(
        eval_results_path=paths.EvaluationResults.PATH,
        eval_results=eval_results)


if __name__ == '__main__':
    main()