import os

import mnist.config as config
import mnist.paths as paths
import mnist.preparation.data_download as d_dload
import mnist.preparation.generate_datasets as gd


def main():
    data_dir = paths.BasePaths.DATA_DIR
    if not os.path.exists(data_dir):
        raise IOError('Base data folder {} does not exist'.format(data_dir))

    dataset_def_dir = paths.BasePaths.DATASET_DEF_DIR
    if not os.path.exists(dataset_def_dir):
        raise IOError('Dataset definition folder {} '
                      'does not exist'.format(dataset_def_dir))

    d_dload.download_mnist_data(data_dir=data_dir,
                                silent=False,
                                dry=False)

    data_split_weights = config.GeneralConfig.DATA_SPLIT_WEIGHTS
    gd.generate_datasets(data_dir=data_dir,
                         dataset_def_dir=dataset_def_dir,
                         data_split_weights=data_split_weights,
                         silent=False,
                         dry=False)


if __name__ == '__main__':
    main()
