import argparse
import os
from math import sqrt
import numpy as np
from tensorflow.keras.models import load_model

from ..wb_attack_federated_generator import WBFederatedAttackGenerator

assert os.getcwd().endswith("FIA"), "script should be started from home folder"

"""
    This script only extracts features
"""


def extract_features(plain_indices, batch_size, output, experiment, models, data, num_classes):
    """
    Extract features from a set of target models
    """
    print("loading indices")
    indices = np.load(plain_indices, allow_pickle=True).item()
    train_indices = indices['train']
    test_indices = indices['test']
    print("loading data")
    data = np.load(data, allow_pickle=True)
    X = data['x']
    Y = data['y']
    if "lfw" in output:
        X = X.astype(np.float32) / 255
        l = int(sqrt(len(X[0])))  # in some LDP cases we work with smaller images
        X = np.reshape(X, (-1, l, l, 1))
        X = np.tile(X, (1, 1, 1, 3))
        Y = Y.astype(np.int8)
        Y = Y.flatten()

    print("creating attack data for single epochs")
    wb_generator = WBFederatedAttackGenerator(models, X, Y, batch_size, train_indices, test_indices,
                                              num_classes, last_layer_only=True, one_hot=True)
    wb_generator.generate(output, experiment)
    return wb_generator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Target model training")
    parser.add_argument("--batch_size", "-b", default=128, type=int, help="Batch Size")
    parser.add_argument("--num_classes", "-n", default=100, type=int, required=True, help="Number of labels")
    parser.add_argument("--experiment", "-e", type=str, required=True, help="Experiment name")
    parser.add_argument("--indices", "-i", type=str, required=True, help="Path to indices")
    parser.add_argument("--data", "-d", type=str, required=True, help="Path to data")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output Path")
    parser.add_argument('--models', type=str, required=True,
                        help='path to models (for T epochs). Order matters!')

    args = parser.parse_args()
    args.models = args.models.split(" ")
    assert os.path.exists(args.data), "data path should point to a file"
    assert os.path.exists(args.indices), "indices path should point to a file"
    for x in args.models:
        assert os.path.exists(x), "model path should point to a file"
    assert os.path.isdir(args.output), "output path should be a directory"
    args.num_classes = int(args.num_classes)
    args.batch_size = int(args.batch_size)
    args.models = list(map(lambda x: load_model(x), args.models))

    extract_features(args.indices, args.batch_size, args.output, args.experiment, args.models,
                     args.data, args.num_classes)
