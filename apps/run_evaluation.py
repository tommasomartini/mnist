import mnist.constants as constants
import mnist.ml.engines.evaluation_engine as eval_eng
import mnist.ml.evaluation_utils as eval_utils
import mnist.paths as paths


def main():
    dataset_name = constants.Constants.TEST_SET_NAMES[0]
    dataset_def_path = paths.DatasetDefinitions.TEST[dataset_name]

    evaluation_engine = eval_eng.EvaluationEngine()

    avg_loss, accuracy = eval_utils.run_evaluation(
        evaluation_engine=evaluation_engine,
        dataset_def_path=dataset_def_path,
        dataset_name=dataset_name
    )

    print('Average loss: {:.3f}\n'
          'Accuracy: {:.3f}\n'.format(avg_loss, accuracy))


if __name__ == '__main__':
    main()