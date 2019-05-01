import mnist.preparation.data_download as d_dload
from mnist.config import GeneralConfig


def main():
    d_dload.download_mnist_data(data_dir=GeneralConfig.DATA_DIR,
                                silent=False,
                                dry=True)


if __name__ == '__main__':
    main()
