import codecs
import json
import pickle
import random
import uuid
from threading import Lock, currentThread
from time import sleep

from flask import *
from flask_socketio import *

from .server import FLServer, pickle_string_to_obj


class FLServerAggregatedLoss(FLServer):
    """
    Federated Averaging algorithm with the server pulling from clients
    The Server synchronizes and instructs the Clients
    It uses the aggregated loss for early stopping
    """
    LOSS_EPS = .0001  # used for convergence
    UNRESPONSIVE_CLIENT_TOLERANCE = .9
    EARLY_STOPPING_TOLERANCE = 15
    MAX_NUM_ROUNDS = 200
    ROUNDS_BETWEEN_VALIDATIONS = 1
    MIN_ROUNDS = 5

    def __init__(self, global_model, host, port, datasource, parallel_workers=2, min_connected=2, path="./",
                 saved_epochs=None, batch_size=128, epoch_per_round=1):
        """
        Initialize global model
        :param global_model:
        :param host:
        :param port:
        """
        super().__init__(global_model, host, port, datasource, [], parallel_workers, min_connected, path,
                         saved_epochs, batch_size, epoch_per_round)


    def handle_client_update(self, data):
        """
        On gathered update , average the weights, loss and accuracies.
        :param data:
        :return:
        """

        print("handle client_update", request.sid, " thread ", currentThread().name)

        self.current_round_client_updates += [data]
        self.current_round_client_updates[-1]['weights'] = pickle_string_to_obj(data['weights'])
        if data['round_number'] != self.current_round:
            return
        if not self.saved_epochs is None and int(data['round_number']) % self.saved_epochs == 0:
            with self.evaluate_lock:
                self.global_model.save_client_model(self.path, request.sid, data['round_number'],
                                                    self.current_round_client_updates[-1]['weights'])
        if len(self.current_round_client_updates) >= \
                self.parallel_workers * FLServer.UNRESPONSIVE_CLIENT_TOLERANCE:
            self.global_model.update_weights(
                [x['weights'] for x in self.current_round_client_updates],
                [float(x['train_size']) for x in self.current_round_client_updates],
            )

            if 'train_loss' in self.current_round_client_updates[0]:
                aggr_train_loss, aggr_train_accuracy = self.global_model.aggregate_train_loss_accuracy(
                    [float(x['train_loss']) for x in self.current_round_client_updates],
                    [float(x['train_accuracy']) for x in self.current_round_client_updates],
                    [float(x['train_size']) for x in self.current_round_client_updates],
                    self.current_round
                )
                self.global_model.prev_train_loss = aggr_train_loss

            if 'valid_loss' in self.current_round_client_updates[0]:
                self.global_model.aggregate_test_loss_accuracy(
                    [float(x['valid_accuracy']) for x in self.current_round_client_updates],
                    [float(x['valid_size']) for x in self.current_round_client_updates],
                    [float(x['valid_loss']) for x in self.current_round_client_updates],
                    self.current_round
                )
            # Early Stopping
            if data['round_number'] > FLServer.MIN_ROUNDS and data[
                'round_number'] > self.global_model.prev_round and \
                    (self.global_model.prev_valid_loss - self.global_model.current_loss) < FLServer.LOSS_EPS:
                # converges
                if self.early_stopping_triggered >= FLServer.EARLY_STOPPING_TOLERANCE:
                    print("converges! starting test phase..")
                    self.stop_and_eval()
                    return
                else:
                    self.early_stopping_triggered += 1

            self.global_model.aggregate_valid_loss_accuracy(self.global_model.current_loss,
                                                            self.global_model.current_accuracy,
                                                            self.current_round)
            self.global_model.prev_valid_loss = self.global_model.current_loss
            self.global_model.prev_round = data['round_number']
            self.client_rounds[request.sid] = self.client_rounds[request.sid] + 1
            if self.current_round >= FLServer.MAX_NUM_ROUNDS:
                print("max round num reached")
                self.stop_and_eval()
            else:
                self.train_next_round()
        else:
            return
