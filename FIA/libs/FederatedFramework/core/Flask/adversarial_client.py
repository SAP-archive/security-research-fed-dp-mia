import json
import sys
from random import randint
from time import sleep

from .client import FederatedClient
from .server import pickle_string_to_obj, obj_to_pickle_string


class AdversarialFederatedClient(FederatedClient):
    """
    The adverserial client gets the global updates and saves them
    """

    def __init__(self, server_host, server_port, optimizer, model_path, datasource, indices, save_epochs):
        super().__init__(server_host, server_port, optimizer, model_path, datasource, indices)
        self.save_epochs = save_epochs
        assert isinstance(save_epochs, int) and save_epochs > 0, "save epochs should be int and larger than zero"

    def on_request_update(self, *args):
        """
        On Request Update we train one round of the local model
        and send the parameters to the server.
        The adversarial thus saves its observed global models over time T
        :param args:
        :return:
        """
        print("Adversary: update requested")
        req = args[0]
        weights = pickle_string_to_obj(req['current_weights'])
        self.local_model.set_weights(weights)
        if int(req['round_number']) % self.save_epochs == 0:
            self.local_model.save_model(self.path, f"{self.client_id}_{req['round_number']}")
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


        self.socketio.emit('client_update', resp)

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
        print("create do_training_information for local attacker model")
        with open(f'{self.path}/do_training_information.json', 'w') as outfile:
            json.dump({
                "client": self.client_id,
                "save_epochs": self.save_epochs,
            }, outfile, indent=4, sort_keys=True)
        sys.exit(0)
