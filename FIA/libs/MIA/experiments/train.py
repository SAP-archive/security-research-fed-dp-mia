import argparse
import os
import numpy as np
import random
import tensorflow as tf

from tensorflow.keras.models import load_model

from .train_wb import train
from .extract_features import extract_features
from .merge_features import merge_features

assert os.getcwd().endswith("FIA"), "script should be started from home folder"

if __name__ == "__main__":
    """
        This script runs MIA WB experiments from MULTIPLE target models
    """
    parser = argparse.ArgumentParser(description="Target model training")
    parser.add_argument("--batch_size", "-b", default=128, type=int, help="Batch Size")
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
    parser.add_argument("--learning_rate", "-lr", type=str, default="0.0001", help="learning rate")

    args = parser.parse_args()
    assert args.workers > 0, "workers should be positive number"
    assert os.path.exists(args.data), "data path should point to a file"
    assert os.path.exists(args.indices), "indices path should point to a file"
    args.models = args.models.split(" ")
    for x in args.models:
        assert os.path.exists(x), f'{x} model path should point to a file'
    assert os.path.isdir(args.output), "output path should be a directory"
    args.num_classes = int(args.num_classes)
    args.batch_size = int(args.batch_size)
    args.learning_rate = float(args.learning_rate)
    args.models = list(map(lambda x: load_model(x, compile=False), args.models))
    np.random.seed(int(args.seed))
    random.seed(int(args.seed))
    generator = extract_features(args.indices, args.batch_size, args.output, args.experiment, args.models,
                                 args.data, args.num_classes)

    merge_features(args.batch_size, generator.files, args.experiment,
                   args.output, args.indices, args.num_classes)

    train(args.batch_size, args.data, f'{args.output}/{args.experiment}_merged_data_inf.json', args.output,
          args.indices, args.save_epochs, args.workers, args.num_classes, args.optimize, args.learning_rate)
