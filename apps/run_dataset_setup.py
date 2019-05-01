import os

import mnist.preparation.data_download as d_dload
import mnist.preparation.generate_datasets as gd
from mnist.config import GeneralConfig


def main():
    data_dir = GeneralConfig.DATA_DIR
    if not os.path.exists(data_dir):
        raise IOError('Base data folder {} does not exist'.format(data_dir))

    d_dload.download_mnist_data(data_dir=data_dir,
                                silent=False,
                                dry=True)

    gd.generate_datasets(data_dir)


if __name__ == '__main__':
    main()
