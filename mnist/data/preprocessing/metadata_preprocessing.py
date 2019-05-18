import mnist.constants as constants


def preprocess_label(label):
    label_type = constants.Constants.LABEL_DATA_TYPE
    label = label_type(label)
    return label
