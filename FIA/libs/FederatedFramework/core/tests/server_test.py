from ..experiments.mnist.mnist import GlobalModel_MNIST_CNN
from ..flask.server import FLServer


def test_server():
    """
    Simply check if class instantiation doesnt crash
    """
    server = FLServer(GlobalModel_MNIST_CNN, "127.0.0.1", 5000, None, [])

    assert server.current_round == -1
    assert server.ready_client_sids.__len__() <= 0
