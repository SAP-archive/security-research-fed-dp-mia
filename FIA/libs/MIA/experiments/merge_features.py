import argparse
import os

from math import sqrt
import numpy as np

from ..wb_attack_federated_generator import WBFederatedAttackGenerator

assert os.getcwd().endswith("FIA"), "script should be started from home folder"

"""
    This script only merges features
"""


def merge_features(batch_size, files, experiment, output, plain_indices, num_classes):
    """
    Merge features by a set of attack datasets
    """
    print("loading indices")
    indices = np.load(plain_indices, allow_pickle=True).item()
    train_indices = indices['train']
    test_indices = indices['test']
    print("merging the attack data")
    # we dont have to set models and data here
    wb_generator = WBFederatedAttackGenerator([], [], [], batch_size, train_indices, test_indices,
                                              num_classes, last_layer_only=True, one_hot=True)
    wb_generator.merge(output, experiment, files=files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Target model training")
    parser.add_argument("--batch_size", "-b", default=128, type=int, help="Batch Size")
    parser.add_argument("--num_classes", "-n", default=100, type=int, required=True, help="Number of labels")
    parser.add_argument("--experiment", "-e", type=str, required=True, help="Experiment name")
    parser.add_argument("--indices", "-i", type=str, required=True, help="Path to indices")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output Path")
    parser.add_argument('--files', "-f", nargs='+', required=True, help='the json attack data info files')

    args = parser.parse_args()

    assert os.path.exists(args.indices), "indices path should point to a file"
    assert os.path.isdir(args.output), "output path should be a directory"
    args.num_classes = int(args.num_classes)
    args.batch_size = int(args.batch_size)
    merge_features(args.batch_size, args.files, args.experiment,
                   args.output, args.indices, args.num_classes)
