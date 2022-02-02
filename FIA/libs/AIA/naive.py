"""
This attribute inference attack model is not used!
"""
import json

import tensorflow.keras as keras
from tensorflow.keras.layers import LeakyReLU
from ..MIA.wb_attack_layer import WBAttackLayer
from ..MIA.wb_attack_federated_generator import WBFederatedAttackGenerator
from tensorflow.keras.models import load_model
from ..MIA.wb_attack_loader import WBAttackLoader
from ..MIA.experiments.train_wb import get_scores
from sklearn.metrics import accuracy_score, precision_score, recall_score

import numpy as np


def get_test_indices(path):
    """
    Retrieve the test indices
    """
    data = np.load(f"{path}/predictions.npz", allow_pickle=True)
    return data["train_indices"]


def load_mia_model(path):
    """
    Load the MIA model. For some reason, it does not detect the LeakyReLu layer automatically
    """
    return keras.models.load_model(f"{path}/attacker_model.h5",
                                   custom_objects={"WBAttackLayer": WBAttackLayer, "LeakyReLU": LeakyReLU})


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def variate_dataset(data, attribut_index, test_indices):
    """
    Variate the dataset |pi(x)| times for every possible attribut value. For texas/purchases |pi(x)|==2 because all values are boolean
    """
    data = np.load(data, allow_pickle=True)
    X_original = data['x'][test_indices]
    Y_original = data['y'][test_indices]
    X_fake = np.copy(X_original)
    Y_fake = np.copy(Y_original)
    for i in range(len(X_fake)):
        X_fake[i][attribut_index] ^= 1
    X = np.concatenate((X_original, X_fake))
    Y = np.concatenate((Y_original, Y_fake))
    return X, Y


def generate(X, Y, models, batch_size, num_classes, output, experiment):
    wb_generator = WBFederatedAttackGenerator(models, X, Y, batch_size, list(range(len(X) // 2)),
                                              list(range(len(X) // 2, len(X))),
                                              num_classes, last_layer_only=True, one_hot=True)
    wb_generator.generate(output, experiment)
    wb_generator.merge(output, experiment)


models = reversed(
    ["./data/global/purchases100_s_42/target/9523ccac5de84caea837360fd45bbae8_55_local_model.h5",
     "./data/global/purchases100_s_42/target/9523ccac5de84caea837360fd45bbae8_50_local_model.h5",
     "./data/global/purchases100_s_42/target/9523ccac5de84caea837360fd45bbae8_45_local_model.h5",
     "./data/global/purchases100_s_42/target/9523ccac5de84caea837360fd45bbae8_40_local_model.h5",
     "./data/global/purchases100_s_42/target/9523ccac5de84caea837360fd45bbae8_35_local_model.h5"
     ])
models = list(map(lambda x: load_model(x, compile=False), models))
batch_size = 100
scores = []
num_classes = 100
path = "./data/global/purchases100_s_42/attack/9523ccac5de84caea837360fd45bbae8"
data = f"./models/shokri_purchases_{num_classes}_classes.npz"
index = 12
x, y = variate_dataset(data, index, get_test_indices(path))
size = (len(x) // batch_size) * batch_size
x = x[:size]
y = y[:size]
#generate(x, y, models, batch_size, num_classes, "/home/ubuntu/FIA/aia", f"aia_test_naive{index}")
file = open(f"./data/aia/aia_test_naive{index}_merged_data_inf.json", 'r')
attack_data_inf = json.load(file)
wb_loader_test = WBAttackLoader(attack_data_inf["attack_train_file"],
                                attack_data_inf["attack_test_file"],
                                list(range(size // 2)), list(range(size // 2)), batch_size, shuffle=False)
model = load_mia_model(path)
conf = np.array(model.predict_generator(wb_loader_test).ravel())
y_conf = []
for i in range(len(conf) // 2):
    y_conf.append(softmax(np.array([conf[i], conf[i + size // 2]]))[0])
y = [1] * (size // 2)
print(get_scores(y, y_conf))
scores.append(get_scores(y, y_conf))
print(scores)
