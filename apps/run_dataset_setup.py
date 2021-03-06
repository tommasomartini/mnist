import os

import mnist.config as config
import mnist.dataset_setup.data_download as d_dload
import mnist.dataset_setup.generate_datasets as gd
import mnist.paths as paths
from mnist.custom_utils.logger import DISABLE_PROGRESS_BAR


def main():
    data_dir = paths.BasePaths.DATA_DIR
    if not os.path.exists(data_dir):
        raise IOError('Base data folder {} does not exist'.format(data_dir))

    dataset_def_dir = paths.BasePaths.DATASET_DEF_DIR
    if not os.path.exists(dataset_def_dir):
        raise IOError('Dataset definition folder {} '
                      'does not exist'.format(dataset_def_dir))

    dry = config.SetupConfig.DRY
    silent = DISABLE_PROGRESS_BAR

    d_dload.download_mnist_data(data_dir=data_dir,
                                silent=silent,
                                dry=dry)

    data_split_weights = config.SetupConfig.DATA_SPLIT_WEIGHTS
    gd.generate_datasets(data_dir=data_dir,
                         dataset_def_dir=dataset_def_dir,
                         data_split_weights=data_split_weights,
                         silent=silent,
                         dry=dry)


if __name__ == '__main__':
    main()
