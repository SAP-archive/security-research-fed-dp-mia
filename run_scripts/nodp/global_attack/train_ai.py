# PARAMETERS #
import os
import pathlib
import subprocess
import json
import numpy as np
import argparse


def execute_command(cmd):
    subprocess.Popen(cmd).wait()


parser = argparse.ArgumentParser(description="Target model training")
parser.add_argument("--path", "-p", type=str, default="./experiments/purchases100")
parser.add_argument("--dataset", "-d", type=str, required=True)
parser.add_argument("--seed", "-s", type=str, required=True)
parser.add_argument("--output_size", "-os", type=int, default=100)
parser.add_argument("--optimize", "-opt", default=0, required=False, type=int, help="Execute bayesian optimization")
parser.add_argument("--learning_rate", "-target_lr", type=str, default="0.0008")
parser.add_argument("--batch_size", "-bs", type=str, default="64")
parser.add_argument("--index", "-ai", type=str, default="0,1,2,3,4,5,6")
args = parser.parse_args()

num_classes = args.output_size  # for purchases 100
workers = 4  # parallel workers
learning_rate = args.learning_rate
optimize = args.optimize
batch_size = args.batch_size
# File pathes. Should all be relative to the Project root
path = args.path
input = f"{path}/target"
seed = args.seed
data_file = args.dataset

with open(f"{input}/training_information.json", 'r') as f:
    data = json.load(f)
    data_owners = data["clients"]
    last_round = int(np.array(data["validation_accuracy"])[-1, 0])
    last_round = last_round - (last_round % 5)
    epochs = [x for x in range(last_round, 0, -5)]
    if len(epochs) > 5:
        epochs = epochs[-5:]
    for data_owner in data_owners:
        output = f"{path}/aia"
        experiment_name = f"attack_{data_owner}"
        pathlib.Path(output).mkdir(parents=True, exist_ok=True)

        models = reversed(list(map(lambda epoch: f"{input}/{data_owner}_{epoch}_local_model.h5", epochs)))
        indices = f"{input}/{data_owner}_indices.npy"

        execute_command(
            ["python3", "-m", "libs.AIA.sophisticate", "--save_epochs", str(len(epochs)), "--num_classes",
             str(num_classes),
             "--experiment", experiment_name, "--indices", indices, "--output", output, "--batch_size", str(batch_size),
             "--models", " ".join(models), "--data", data_file, "--workers", str(workers), "--seed", str(seed),
             "--learning_rate", str(learning_rate), "--optimize", str(optimize), "--index", args.index])
        exit(1)
