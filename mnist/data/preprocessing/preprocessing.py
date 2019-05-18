import mnist.data.preprocessing.image_preprocessing as img_pp
import mnist.data.preprocessing.metadata_preprocessing as meta_pp


def load_and_preprocess_sample(image_path, label):
    preprocessed_image = img_pp.load_and_preprocess_image(image_path)
    preprocessed_label = meta_pp.preprocess_label(label)
    return preprocessed_image, preprocessed_label
