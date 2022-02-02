"""
This is yeoms attribute inference attack model and is not used!
"""
import numpy as np
from tensorflow.keras.models import load_model
from ..MIA.experiments.train_wb import get_scores


def get_test_indices(path):
    """
    Retrieve the test indices
    """
    indices = np.load(path, allow_pickle=True).item()
    #    return np.concatenate((indices['train'],indices['test']))
    return indices["train"]


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
    for i in range(len(X_fake)):
        X_fake[i][attribut_index] ^= 1

    X = np.concatenate((X_original, X_fake))
    Y = np.concatenate((Y_original, Y_fake))
    print(len(X_original), len(Y_original), len(X_fake), len(Y_fake))
    return X, Y


models = ["./data/experiments/global/texas100_s_42/target/2e6b7d3509284da39d2389532b9bace4_10_local_model.h5",
          ]
models = list(map(lambda x: load_model(x, compile=True), models))

# loss = SparseCategoricalCrossentropy(from_logits=False, reduction=tf.compat.v1.losses.Reduction.NONE)
# models[0].compile(optimizer=Adam(0.001), loss=loss, metrics=['accuracy'])
num_classes = 100
index = 2

path = "./data/global/texas100_s_42/target/2e6b7d3509284da39d2389532b9bace4_indices.npy"
data = f"./models/shokri_texas_{num_classes}_classes.npz"
X, Y = variate_dataset(data, index, get_test_indices(path))
conf = []
size = len(X) // 2
for i in range(len(X)):
    conf.append(models[0].evaluate(X[i:i + 1], Y[i:i + 1], batch_size=1)[0])
pred_y = []
scores = []
true_y = []
for i in range(size):
    print(X[i][index], X[i + size][index])
    true_y.append(X[i][index])
    if conf[i] < conf[i + size]:
        pred_y.append(X[i][index])
    else:
        pred_y.append(X[i + size][index])

# true_y = X[:size, index]

scores.append(get_scores(true_y, pred_y))
print(list(true_y))
print("bla")
print(pred_y)
print(scores)
