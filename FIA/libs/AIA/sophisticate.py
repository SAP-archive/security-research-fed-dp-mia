"""
This is the main attribute inference attack model
"""
import argparse
import os
import pathlib
import random
import tensorflow.keras as keras
from tensorflow.keras.layers import LeakyReLU
from ..MIA.wb_attack_layer import WBAttackLayer
from ..MIA.wb_attack_federated_generator import WBFederatedAttackGenerator

import numpy as np
from ..MIA.experiments.train_wb import train
from tensorflow.keras.models import load_model

assert os.getcwd().endswith("FIA"), "script should be started from home folder"


def get_test_indices(path):
    """
    Retrieve the test indices
    """
    indices = np.load(path, allow_pickle=True).item()
    return indices['train']


def arreq_in_list(myarr, list_arrays):
    return next((True for elem in list_arrays if np.array_equal(elem, myarr)), False)


def variate_dataset(data, attribut_index, test_indices):
    """
    Variate the dataset |pi(x)| times for every possible attribut value. For texas/purchases |pi(x)|==2 because all values are boolean
    """
    data = np.load(data, allow_pickle=True)
    X_original = data['x'][test_indices]
    Y_original = data['y'][test_indices]
    X_fake = np.copy(X_original)
    Y_fake = np.copy(Y_original)
    todelete = []
    for i in range(len(X_fake)):
        X_fake[i][attribut_index] ^= 1
        if arreq_in_list(X_fake[i], X_original):
            todelete.append(i)
    print(len(X_original), len(Y_original), len(X_fake), len(Y_fake), "delete:", len(todelete))

    X_original = np.delete(X_original, todelete, axis=0)
    X_fake = np.delete(X_fake, todelete, axis=0)
    Y_original = np.delete(Y_original, todelete, axis=0)
    Y_fake = np.delete(Y_fake, todelete, axis=0)

    X = np.concatenate((X_original, X_fake))
    Y = np.concatenate((Y_original, Y_fake))
    print(len(X_original), len(Y_original), len(X_fake), len(Y_fake))
    return X, Y


def generate(X, Y, models, batch_size, num_classes, output, experiment):
    wb_generator = WBFederatedAttackGenerator(models, X, Y, batch_size, list(range(len(X) // 2)),
                                              list(range(len(X) // 2, len(X))),
                                              num_classes, last_layer_only=True, one_hot=True)
    wb_generator.generate(output, experiment)
    wb_generator.merge(output, experiment)


if __name__ == "__main__":
    """
        This script runs MIA WB experiments from MULTIPLE target models
    """
    parser = argparse.ArgumentParser(description="Target model training")
    parser.add_argument("--batch_size", "-target_b", default=128, type=int, help="Batch Size")
    parser.add_argument("--num_classes", "-n", default=100, type=int, required=True, help="Number of labels")
    parser.add_argument("--experiment", "-e", type=str, required=True, help="Experiment name")
    parser.add_argument("--seed", "-sd", type=str, required=True, help="Seed")
    parser.add_argument("--indices", "-i", type=str, required=True, help="Path to indices")
    parser.add_argument("--data", "-d", type=str, required=True, help="Path to data")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output Path")
    parser.add_argument('--models', required=True, type=str, help='path to models (for T epochs)')
    parser.add_argument("--workers", "-w", default=4, type=int, help="Amount of cpus used to train")
    parser.add_argument("--save_epochs", "-s", default=4, required=True, type=int,
                        help="List of epochs a model should be saved")
    parser.add_argument("--optimize", "-opt", default=0, required=False, type=int, help="Execute bayesian optimization")
    parser.add_argument("--learning_rate", "-target_lr", type=str, default="0.0001", help="learning rate")
    parser.add_argument("--index", "-ai", required=True, type=str,
                        help="Index of sensitive attribute to be attacked (comma seperated)")

    args = parser.parse_args()
    assert args.workers > 0, "workers should be positive number"
    assert os.path.exists(args.data), "data path should point to a file"
    assert os.path.exists(args.indices), "indices path should point to a file"
    args.models = args.models.split(" ")
    for x in args.models:
        assert os.path.exists(x), f'{x} model path should point to a file'
    assert os.path.isdir(args.output), "output path should be a directory"
    num_classes = int(args.num_classes)
    batch_size = int(args.batch_size)
    learning_rate = float(args.learning_rate)
    models = list(map(lambda x: load_model(x, compile=False), args.models))
    np.random.seed(int(args.seed))
    random.seed(int(args.seed))
    save_epochs = args.save_epochs

    data = args.data
    experiment = args.experiment

    sensitive_indices = args.index.split(",")
    for index in sensitive_indices:
        output = f"{args.output}/{index}"
        index = int(index)
        pathlib.Path(output).mkdir(parents=True, exist_ok=True)
        x, y = variate_dataset(data, index, get_test_indices(args.indices))
        generate(x, y, models, batch_size, num_classes, output, experiment)
        train(batch_size, data, f'{output}/{experiment}_merged_data_inf.json', output,
              list(range(len(x))), save_epochs, 8, num_classes, args.optimize, learning_rate)
