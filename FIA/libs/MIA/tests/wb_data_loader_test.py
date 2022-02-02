import json

import pytest
import os

from ..wb_attack_loader import WBAttackLoader


def test_load():
    """
    Overall this test checks that WBAttackLoader does not crash
    """
    with open(f'{os.getcwd()}/libs/MIA/tests/fixtures/mnist_train_data_inf.json', 'r') as file:
        attack_data_inf = json.load(file)
        WBAttackLoader(attack_data_inf["attack_train_file"],
                       attack_data_inf["attack_test_file"],
                       range(0, 10), range(10, 20), 100)
        assert attack_data_inf["attack_train_file"] is not None
        assert attack_data_inf["attack_test_file"] is not None
        assert attack_data_inf['layers_used'][0] == 128
        assert attack_data_inf['layers_used'][1] == 10
