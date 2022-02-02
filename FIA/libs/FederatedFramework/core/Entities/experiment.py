import multiprocessing

import numpy as np

from ..flask.adversarial_client import AdversarialFederatedClient
from ..flask.client import FederatedClient
from ..flask.server import FLServer


class Experiment:
    """
    A Experiment consists of a MODEL and a DATASOURCE
    Here a federated system will be parametrized and orchestrated to train a model
    collaboratively
    """

    def __init__(self, optimizer, datasource, model):
        assert self.__class__.__name__.endswith("Experiment"), "experiment name should end with 'Experiment'"
        self.model = model
        self.all_indices = []
        self.processes = []
        self.optimizer = optimizer
        self.datasource = datasource

    def stop_all(self):
        """
        Terminating each client
        """
        print("TERMINATING")
        for p in self.processes:
            p.terminate()
            p.join()

    def start_client(self, num_clients, seed, path, index, port=5000):
        """
        Start a specific client
        """
        indices = self.datasource.disjoint_dataset_indices(index, num_clients, seed)
        assert set(self.all_indices).isdisjoint(indices)
        client = FederatedClient("127.0.0.1", port,
                                 self.optimizer, path, self.datasource, indices)
        p = multiprocessing.Process(target=client.start)
        p.start()
        self.processes.append(p)
        self.all_indices = np.concatenate([self.all_indices, indices])

    def start_clients(self, num_clients, seed, path,
                      hasAdversary=False, save_epochs=[], port=5000):
        """
        start the clients in seperate processes
        """
        for i in range(1, (num_clients - 1) if hasAdversary else num_clients):
            self.start_client(num_clients, seed, path, i, port)

        if hasAdversary:
            self.start_adversarial(num_clients, seed, path, save_epochs, port)

    def start_server(self, num_clients, seed, path, saved_epochs, parallel_workers=2, min_connected=2, batch_size=128,
                     epochs=1, port=5000):
        """
        start the server
        """
        indices = self.datasource.disjoint_dataset_indices(0, num_clients, seed)
        assert set(self.all_indices).isdisjoint(indices)
        server = FLServer(self.model, "0.0.0.0", port, self.datasource, indices, parallel_workers=parallel_workers,
                          min_connected=min_connected, path=path, saved_epochs=saved_epochs, batch_size=batch_size,
                          epoch_per_round=epochs)
        server.start()

    def start_adversarial(self, num_clients, seed, path, save_epochs, port=5000):
        """
        start an adversarial client saves multiple models
        """
        print("starting adversarial")
        indices = self.datasource.disjoint_dataset_indices(num_clients - 1, num_clients, seed)
        assert set(self.all_indices).isdisjoint(indices)
        client = AdversarialFederatedClient("127.0.0.1", port,
                                            self.optimizer, path, self.datasource, indices, save_epochs)
        p = multiprocessing.Process(target=client.start)
        self.processes.append(p)
        p.start()
