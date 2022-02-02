import os

import tensorflow.keras as keras

from ..flask.client import FederatedClient
from ..experiments.mnist.mnist import Mnist


def test_client():
    """
    Simply test if things dont crash
    """
    optimizer = {'loss': keras.losses.categorical_crossentropy,
                 'optimizer': keras.optimizers.Adadelta(),
                 'metrics': ['accuracy']}
    mnist = Mnist()
    indices = mnist.disjoint_dataset_indices(0, 1, 1337)

    client = FederatedClient("127.0.0.1", 5000,
                             optimizer, f"{os.getcwd()}/models", mnist, indices)
    assert client.server_host == "127.0.0.1"
    assert client.server_port == 5000
