from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
import os


def get_data():
    """
    Retrieve exemplary the MNIST model to test te overall MIA Framework
    """
    model = load_model(f'{os.getcwd()}/libs/MIA/tests/fixtures/1058f0b567c04c488d7f8085c92d9b40_local_model.h5')
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    y_train = to_categorical(y_train, 10)[:100]
    y_test = to_categorical(y_test, 10)[:100]
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))[:100]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))[:100]
    return model, x_train, x_test, y_train, y_test
