import json
import time
from threading import Lock

import numpy as np
import tensorflow as tf


class GlobalModel(object):
    """
    The Global model is kept by the server.
    all weights from the clients local models will be averaged and updated to the global
    afterwards it will get sent back to the clients
    """

    def __init__(self, output_size=100):
        """
        Initialize the global model
        """
        self.output_size = output_size
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.session = tf.Session()
            with self.session.as_default():
                self.model = self.build_model()
                self.model._make_predict_function()
                self.current_weights = self.model.get_weights()
                self.client_model = self.build_model()

        # for convergence check
        self.prev_valid_loss = 1e-04
        self.prev_round = -1
        self.evaluate_lock = Lock()
        self.update_lock = Lock()
        # all rounds; losses[i] = [round#, timestamp, loss]
        # round# could be None if not applicable
        self.train_losses = []
        self.valid_losses = []
        self.train_accuracies = []
        self.valid_accuracies = []
        self.test_accuracies = []
        self.current_loss = 1e-03
        self.current_accuracy = 0
        self.training_start_time = int(round(time.time()))

    def build_model(self):
        """
        Build the model
        """
        raise NotImplementedError()

    def evaluate(self, X, y):
        """
        evaluate the global model
        important are the locks, so no race condition can appear
        also tensorflow 1.15 needs the same graph and session for all Threads/Processes
        """
        with self.update_lock:
            with self.graph.as_default():
                with self.session.as_default():
                    loss, acc = self.model.evaluate(X, y, verbose=1)
                    return loss, acc

    def update_weights(self, client_weights, client_sizes):
        """
        Update the weights according to McMahan FedAVG algorithm
        updates are weighted by the training sizes of the respective data owners
        """
        with self.update_lock:
            with self.graph.as_default():
                with self.session.as_default():
                    """
                     Update the global models weight by summing over the averaged local model weights
                    """
                    new_weights = [np.zeros(w.shape) for w in self.current_weights]
                    total_size = np.sum(client_sizes)
                    for c in range(len(client_weights)):
                        for i in range(len(new_weights)):
                            new_weights[i] += np.float64(client_weights[c][i]) * np.float64(
                                client_sizes[c]) / total_size
                    self.current_weights = new_weights
                    self.model.set_weights(new_weights)

    def aggregate_loss_accuracy(self, client_losses, client_accuracies, client_sizes):
        """
        Calculate the global loss accuracy by averaging over the local ones
        """
        total_size = np.sum(client_sizes)
        # weighted sum
        aggr_loss = np.sum(client_losses[i] / total_size * client_sizes[i]
                           for i in range(len(client_sizes)))
        aggr_accuracies = np.sum(client_accuracies[i] / total_size * client_sizes[i]
                                 for i in range(len(client_sizes)))
        return aggr_loss, aggr_accuracies

    def aggregate_train_loss_accuracy(self, client_losses, client_accuracies, client_sizes, cur_round):
        """
        Calculate the train loss accuracy
        """
        cur_time = int(round(time.time())) - self.training_start_time
        aggr_loss, aggr_accuraries = self.aggregate_loss_accuracy(client_losses, client_accuracies, client_sizes)
        self.train_losses += [[cur_round, cur_time, aggr_loss]]
        self.train_accuracies += [[cur_round, cur_time, aggr_accuraries]]
        return aggr_loss, aggr_accuraries

    def aggregate_test_loss_accuracy(self, client_accuracies, client_sizes, valid_loss, cur_round):
        """
        Calculate the train loss accuracy
        """
        total_size = np.sum(client_sizes)
        cur_time = int(round(time.time())) - self.training_start_time
        aggr_accuracies = np.sum(client_accuracies[i] / total_size * client_sizes[i]
                                 for i in range(len(client_sizes)))
        self.test_accuracies += [[cur_round, cur_time, aggr_accuracies]]
        self.current_loss = np.sum(valid_loss[i] / total_size * client_sizes[i]
                                   for i in range(len(client_sizes)))
        self.current_accuracy = aggr_accuracies
        return aggr_accuracies

    def aggregate_valid_loss_accuracy(self, loss, accuracy, cur_round):
        """
        Calculate the validation loss accuracy
        """
        cur_time = int(round(time.time())) - self.training_start_time
        self.valid_losses += [[cur_round, cur_time, str(loss)]]
        self.valid_accuracies += [[cur_round, cur_time, str(accuracy)]]

    def get_stats(self):
        return {
            "train_loss": self.train_losses,
            "valid_loss": self.valid_losses,
            "train_accuracy": self.train_accuracies,
            "valid_accuracy": self.valid_accuracies,
            "test_accuracy": self.test_accuracies
        }

    def save_validation_indices(self, path, save_indices):
        """
        save the validation indeces from the aggregator
        """
        np.save(f'{path}/aggregator_indices.npy', save_indices)

    def save_client_model(self, path, cid, epoch, weights):
        """
        save a client model at a specific epoch
        the result will be used for inference attack to examine the privaccy gains (see Nasr et al)
        """
        print(f'saving local model from client {cid} for epoch {epoch} to {path}/{cid}_{epoch}_local_model.h5')
        with self.update_lock:
            with self.graph.as_default():
                with self.session.as_default():
                    self.client_model.set_weights([np.float64(x) for x in weights])
                    self.client_model.save(f'{path}/{cid}_{epoch}_local_model.h5')
