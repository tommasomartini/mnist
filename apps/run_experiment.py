import numpy as np
import tensorflow as tf

import mnist.config as config
import mnist.ml.experiment_scheduler as scheduler
import mnist.paths as paths


def _set_random_seeds(random_seed):
    np.random.seed(random_seed)
    tf.set_random_seed(random_seed)


def main():
    random_seed = config.GeneralConfig.RANDOM_SEED
    log_dir = paths.BasePaths.EXP_LOG_DIR

    _set_random_seeds(random_seed)
    scheduler.ExperimentScheduler(log_dir).run()


if __name__ == '__main__':
    main()
