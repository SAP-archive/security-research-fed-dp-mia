from threading import Lock

import numpy as np
import sklearn
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import tensorflow.keras.backend as K


class LocalModel(object):
    """
    Local Model
    Each Client has its own model. The Weights will be sent to the Server.
    The Server updates a Global Model and sends this back to the clients.
    """

    def __init__(self, model_config, data_collected, optimizer):
        """
        Create Local Modal.
        Retrieved Configuration from server and local client data are applied here.
        :param model_config:
        :param data_collected:
        :param optimizer:
        """
        assert optimizer is not None, "please prove a optimizer dict"
        assert optimizer['loss'] is not None, "a loss function must be set in the optimizer dict"
        assert optimizer['metrics'] is not None, "a metric must be defined in the optimizer dict"
        assert optimizer['optimizer'] is not None, "a optimizer must be defined in the optimizer dict"
        assert data_collected["x_train"] is not None, "X matrix for training must be provided"
        assert data_collected["y_train"] is not None, "y vector for training must be provided"
        assert data_collected["x_test"] is not None, "X matrix for testing must be provided"
        assert data_collected["y_test"] is not None, "y vector for testing must be provided"
        self.model_config = model_config
        self.update_lock = Lock()
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.session = tf.Session()
            with self.session.as_default():
                self.model = model_from_json(model_config['model_json'])
                if len(self.model.layers) >= 32: # hardcoded if model is lfw
                    for l in self.model.layers[:-4]:
                        l.trainable = False
                self.optimizer = optimizer
                self.model.compile(loss=optimizer['loss'],
                                   optimizer=optimizer['optimizer'](),
                                   metrics=optimizer['metrics'])
                self.model._make_predict_function()

        self.x_train = np.array(data_collected["x_train"])
        self.y_train = np.array(data_collected["y_train"])
        self.x_test = np.array(data_collected["x_test"])
        self.y_test = np.array(data_collected["y_test"])

    def get_weights(self):
        """
        Get Keras Model Weights
        :return: weights
        """
        return self.model.get_weights()

    def set_weights(self, new_weights):
        """
        Sets the Keras model Weights
        :param new_weights:
        :return:
        """
        with self.update_lock:
            with self.graph.as_default():
                with self.session.as_default():
                    self.model.set_weights(new_weights)

    def get_batch(self, x, y):
        """
        Returns a random training batch
        :return: Training Batch
        """
        x, y = sklearn.utils.shuffle(x, y)
        residual = (len(x) % self.model_config['batch_size'])
        return x[:-residual], y[:-residual]

    def train_one_round(self):
        """
        Train one round
        :return: weights and score
        """

        x_train, y_train = self.get_batch(self.x_train, self.y_train)
        with self.update_lock:
            with self.graph.as_default():
                with self.session.as_default():
                    self.model.fit(x_train, y_train,
                                   epochs=self.model_config['epoch_per_round'],
                                   batch_size=self.model_config['batch_size'],
                                   verbose=1,
                                   validation_data=(x_train, y_train))
                    score = self.model.evaluate(x_train, y_train, batch_size=self.model_config['batch_size'], verbose=0)
                    score[0] = np.mean(score[0])
                    print('Train loss:', score[0])
                    print('Train accuracy:', score[1])
                    return self.model.get_weights(), score[0], score[1]

    def evaluate(self):
        """
        Evaluation fo Test set after global model converged
        :return:
        """
        with self.update_lock:
            with self.graph.as_default():
                with self.session.as_default():
                    x_test, y_test = self.get_batch(self.x_test, self.y_test)
                    score = self.model.evaluate(x_test, y_test, batch_size=self.model_config['batch_size'], verbose=0)
                    score[0] = np.mean(score[0])
                    print('Test loss:', score[0])
                    print('Test accuracy:', score[1])
                    return score

    def save_model(self, path, cid):
        print(f'saving local model to {path}/{cid}_local_model.h5')
        with self.update_lock:
                with self.graph.as_default():
                    with self.session.as_default():
                        self.model.save(f'{path}/{cid}_local_model.h5')

    def save(self, path, cid, train_indices, test_indices):
        """
        Save Global Model wrt. to client
        :param path:
        :param cid:
        :param train_indices:
        :param test_indices:
        :param validation_indices:
        :return:
        """
        save_indices = {"train": train_indices, "test": test_indices}
        print(f'saving indices to {path}/{cid}_indices.npy')
        np.save(f'{path}/{cid}_indices.npy', save_indices)
        self.save_model(path, cid)
