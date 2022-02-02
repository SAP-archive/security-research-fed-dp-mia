import sys

from sklearn.model_selection import train_test_split
import socketio
import gc
from .server import obj_to_pickle_string, pickle_string_to_obj, LOGGER
from ..entities.local_model import LocalModel
from random import randint
from time import sleep
import tensorflow as tf


class FederatedClient(object):
    """
    The Flask Client in a Federated ML Setup
    """

    def __init__(self, server_host, server_port, optimizer, model_path, datasource, indices):
        """
        Initialize the upstream configuration
        :param server_host:
        :param server_port:
        :param optimizer: A dict containing a metric, optimizer and loss function
        :param model_path: Path to Save models
        :param X:   Dataset
        :param y: Labels
        """
        self.local_model = None
        self.path = model_path
        self.optimizer = optimizer
        self.X, self.y = datasource.get_data()
        self.indices = indices
        self.stopped = False
        self.train_indices = []
        self.test_indices = []
        self.client_id = 0
        self.server_host = server_host
        self.server_port = server_port
        self.socketio = None

    def start(self):
        """
        Start the Socket client
        """
        sleep(randint(15, 20))  # wait server to be started
        assert self.socketio is None, "socketio cannot be initialized again"
        self.socketio = socketio.Client(engineio_logger=LOGGER, request_timeout=1000000000, logger=LOGGER)
        self.register_handles()
        self.socketio.connect(f"http://{self.server_host}:{self.server_port}/")

    def split_data(self, data_split):
        """
        Prepare the Data
        :param data_split:
        :return:
        """
        self.train_indices, self.test_indices = train_test_split(self.indices, test_size=data_split)
        assert (set(self.train_indices).isdisjoint(self.test_indices)), "train and test indices are not disjoint"
        return {
            "x_train": self.X[self.train_indices],
            "y_train": self.y[self.train_indices],
            "x_test": self.X[self.test_indices],
            "y_test": self.y[self.test_indices],
        }

    ########## Socket IO messaging ##########
    def on_connect(self):
        print('connect')
        print("sent wakeup")
        self.socketio.emit('client_wake_up')

    def on_init(self, *args):
        """
        Setup the local modal from the serialized verison from the server
        :param args:
        :return:
        """
        model_config = args[0]
        print('my id is: ', model_config['client_id'])
        self.client_id = model_config['client_id']
        print('preparing local data based on server model_config')
        self.local_model = LocalModel(model_config,
                                      self.split_data(model_config['data_split']),
                                      self.optimizer)
        # ready to be dispatched for training
        self.socketio.sleep(1)
        self.socketio.emit('client_ready', {
            'train_size': self.local_model.x_train.shape[0]})

    def on_disconnect(self):
        print('disconnect')
        if self.stopped:
            self.socketio.disconnect()
            sys.exit(0)

    def on_reconnect(self):
        print('reconnect')
        if self.stopped:
            self.socketio.disconnect()
            sys.exit(0)

    def on_request_update(self, *args):
        """
        On Request Update we train one round of the local model
        and send the parameters to the server
        :param args:
        :return:
        """
        print("update requested")
        req = args[0]
        weights = pickle_string_to_obj(req['current_weights'])
        self.local_model.set_weights(weights)
        my_weights, train_loss, train_accuracy = self.local_model.train_one_round()
        resp = {
            'round_number': req['round_number'],
            'weights': obj_to_pickle_string(my_weights),
            'train_size': self.local_model.x_train.shape[0],
            'valid_size': self.local_model.x_test.shape[0],
            'train_loss': str(train_loss),
            'train_accuracy': str(train_accuracy),
        }
        if req['run_validation']:
            valid_loss, valid_accuracy = self.local_model.evaluate()
            resp['valid_loss'] = str(valid_loss)
            resp['valid_accuracy'] = str(valid_accuracy)
        self.socketio.sleep(randint(1, 10))
        self.socketio.emit('client_update', resp)
        gc.collect()
        return

    def on_stop_and_eval(self, *args):
        """
        If model form server converged we evaluate our local model
        :param args:
        :return:
        """
        req = args[0]
        weights = pickle_string_to_obj(req['current_weights'])
        self.local_model.set_weights(weights)
        test_loss, test_accuracy = self.local_model.evaluate()
        resp = {
            'test_size': self.local_model.x_test.shape[0],
            'test_loss': str(test_loss),
            'test_accuracy': str(test_accuracy)
        }
        self.socketio.emit('client_eval', resp)
        self.local_model.save(self.path, self.client_id, self.train_indices, self.test_indices)
        self.stopped = True
        self.socketio.disconnect()
        sys.exit(0)

    def register_handles(self):
        self.socketio.on('connect', self.on_connect)
        self.socketio.on('disconnect', self.on_disconnect)
        self.socketio.on('reconnect', self.on_reconnect)
        self.socketio.on('init', self.on_init)
        self.socketio.on('request_update', self.on_request_update)
        self.socketio.on('stop_and_eval', self.on_stop_and_eval)
