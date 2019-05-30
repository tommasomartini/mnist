import json

import mnist.file_interface as fi
import mnist.ml.engines.inference_engine as inf_eng


def main():
    image_path = '/home/tom/Data/mnist/4d/7a/53848.png'
    metadata_path = '/home/tom/Data/mnist/4d/7a/53848.json'

    with open(metadata_path, 'r') as f:
        meta = json.load(f)
    label = fi.MetadataReader.get_label(meta)

    with inf_eng.InferenceEngine() as inference_engine:
        predictions, probs = \
            inference_engine.load_preprocess_and_predict([image_path])

    title = 'True: {}\n' \
            'Predicted: {}\n' \
            '  with prob: {:.3f}'.format(label,
                                         predictions[0],
                                         probs[0][predictions[0]])
    print(title)


if __name__ == '__main__':
    main()