import pytest
import os
import numpy as np

from .utils import get_data
from ..wb_attack_data_generator import WBAttackGenerator


def test_data():
    """
    The Data Generator should not crash
    """
    model, x_train, x_test, y_train, y_test = get_data()
    X = np.concatenate((x_train, x_test))
    Y = np.concatenate((y_train, y_test))
    wb_generator_train = WBAttackGenerator(model, X, Y,
                                           range(0, len(x_train) // 2), range(len(x_train) // 2, len(x_train)),
                                           10, 10, last_layer_only=True)
    wb_generator_train.write_attack_info(f'{os.getcwd()}/libs/MIA/tests/fixtures/', "mnist_train")

    assert os.path.exists(f'{os.getcwd()}/libs/MIA/tests/fixtures/mnist_train_data_inf.json')
    assert os.path.exists(f'{os.getcwd()}/libs/MIA/tests/fixtures/mnist_train_target_train_attack_data.h5')
    assert os.path.exists(f'{os.getcwd()}/libs/MIA/tests/fixtures/mnist_train_target_test_attack_data.h5')
