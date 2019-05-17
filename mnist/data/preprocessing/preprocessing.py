import mnist.data.preprocessing.image_preprocessing as img_pp


def load_and_preprocess_sample(image_path, label):
    preprocessed_image = img_pp.load_and_preprocess_image(image_path)
    return preprocessed_image, label
