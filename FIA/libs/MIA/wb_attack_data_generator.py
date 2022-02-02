import json
from pathlib import Path

import h5py
import numpy as np
from tensorflow.keras import backend as K
import tensorflow.keras as keras
import tensorflow as tf
import os
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam


class WBAttackGenerator:
    """
    Class to generate WB Train/Test Data
    WB-Input layer consists of several components including:
    - gradient
    - Output Hidden Layers
    - Loss
    - Label
    """
    DS_NAME = 'attack_data'

    def __init__(self, model, X, Y,
                 train_indices, test_indices, batch_size,
                 num_classes, last_layer_only=True, one_hot=False):
        """
        train_data_file: Train data from target model
        test_data_file: Test data from target model
        """
        assert len(X) == len(Y), f"size of X and Y must match {len(X)} != {len(Y)}"
        assert batch_size > 0, "batch size cannot be negative"
        assert num_classes > 0, "num classes must be greater 0"
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.one_hot = one_hot
        self.one_hot_Y = keras.utils.to_categorical(self.Y, self.num_classes)
        self.train_indices = train_indices
        self.test_indices = test_indices
        self.last_layer_only = last_layer_only
        self.layers_used = []
        self.model = model
        loss = SparseCategoricalCrossentropy(reduction=tf.compat.v1.losses.Reduction.NONE)
        self.model.compile(optimizer=Adam(0.001), loss=loss, metrics=['accuracy'])
        # Init gradient extraction
        self.keras_get_gradients = self._init_gradient_extraction()

    def write_attack_info(self, folder_path, name):
        """
        Write a json file containing the used layer information (# neurons)
        And the file paths to the train/test-files
        """

        file_name_train = f'{folder_path}/{name}_target_train_attack_data.h5'
        file_name_test = f'{folder_path}/{name}_target_test_attack_data.h5'

        assert not os.path.exists(file_name_train), "training file already exists"
        assert not os.path.exists(file_name_test), "test file already exists"
        self._create_attack_data(file_name_train, self.train_indices)

        self._create_attack_data(file_name_test, self.test_indices)
        data_inf = {}
        data_inf['last_layer_only'] = self.last_layer_only
        data_inf['attack_train_file'] = file_name_train
        data_inf['attack_test_file'] = file_name_test
        data_inf['layers_used'] = self.layers_used

        data_inf_file = f'{folder_path}/{name}_data_inf.json'
        with open(data_inf_file, 'w') as outfile:
            json.dump(data_inf, outfile, indent=4, sort_keys=True)

    def _create_attack_data(self, output_file, indices):
        """
        extract and serialize the Features from the Target Model for the White Box Attacker Model
        """
        total_elements = len(indices)
        percent = 0
        print('Creating ' + output_file)
        print("get layer outputs")
        l_outs = self.get_layer_outputs()
        print("get losses")
        losses = self.get_losses()
        # loop datapoints in set
        print("get gradients")
        for i, ind in enumerate(indices):
            per = (i * 100) // total_elements
            if percent < per:
                percent = per
                print('Progress ' + str(per) + '%')

            # get gradients for datapoint
            gradients = self.get_gradients(self.X[ind:ind + 1], self.Y[ind:ind + 1])

            grads = None
            # at this point gradients are still in shape (n x m)
            # (where n is the previous layer size and m the current layer size)
            # if reshape_grads is set to True,
            # the gradient matrix will be transposed to get a (m x n) matrix before flattening
            reshape_grads = True
            for g in gradients:
                # append bias gradients to weight gradient matrix
                g_l = np.concatenate((g[0], [g[1]]))
                if reshape_grads:
                    g_l = g_l.T

                if grads is None:
                    grads = g_l.flatten()
                else:
                    grads = np.concatenate([grads, g_l.flatten()])

            # get outputs for datapoint
            l_out = None
            for l in l_outs:
                if l_out is None:
                    l_out = l[ind]
                else:
                    l_out = np.concatenate([l_out, l[ind]])

            # concatenate all components data
            if self.one_hot:
                output = np.concatenate(
                    [grads, l_out, np.array([losses[ind]]), self.one_hot_Y[ind]])
            else:
                output = np.concatenate([grads, l_out, np.array([losses[ind]]), self.Y[ind]])

            # write data to output file
            WBAttackGenerator.write_to_file(output_file, '%s' % WBAttackGenerator.DS_NAME, output)

    def get_layer_outputs(self):
        """
        get the layer outputs
        """
        layer_outputs = []
        percent = 0

        for i in range(0, self._get_iterations()):
            per = (i * 100) // self._get_iterations()
            if percent < per:
                percent = per
                print('Progress ' + str(per) + '%')
            start_i = i * self.batch_size
            end_i = (i + 1) * self.batch_size
            if i > self._get_batches():
                end_i = start_i + self._get_rest()

            layer_outputs.append(self._get_layer_outputs(self.X[start_i:end_i]))

        return np.concatenate(layer_outputs[:], axis=1)

    def get_losses(self):
        """
        Retrieve the losses by evaluating the model
        """
        all_losses = []
        percent = 0
        for i in range(0, self._get_iterations()):
            per = (i * 100) // self._get_iterations()
            if percent < per:
                percent = per
                print('Progress ' + str(per) + '%')

            start_i = i * self.batch_size
            end_i = (i + 1) * self.batch_size
            if i > self._get_batches():
                end_i = start_i + self._get_rest()

            loss = \
                self.model.evaluate(self.X[start_i:end_i], self.Y[start_i:end_i], batch_size=self.batch_size,
                                    verbose=0)[
                    0]
            all_losses.append(loss)
        return np.concatenate(all_losses[:])

    def get_gradients(self, x, y):
        """
        Return the gradient of every trainable weight in model
        """

        """
        gradients_tensor = K.gradients(self.model.output, weights_tensor)
        input_tensors = (
                    self.model._feed_inputs + self.model._feed_targets + self.model._feed_sample_weights)  # TODO: This was changed
        iterate = K.function(input_tensors, gradients_tensor)

        grads = np.array(iterate([x, y]))
        """
        grads = self.keras_get_gradients([x, y])
        grad_w_b = np.empty((len(grads) // 2, 2), dtype=object)
        # grad_biases = np.empty((grads/2, 1))

        # gradients always consist of two matrices
        # for each layer (one matrix for weights (n x m) and one for biases(1 x m))
        for i, l in enumerate(grads):
            if not i % 2:
                # l = [item for sublist in l for item in sublist]
                grad_w_b[i // 2, 0] = np.array(l)
            else:
                grad_w_b[i // 2, 1] = np.array(l)

        return grad_w_b

    def _init_gradient_extraction(self):
        """
        Set the trainable weights and the gradient function
        """
        weights_tensor = self.model.trainable_weights
        if self.last_layer_only:
            weights_tensor = weights_tensor[-2:]
        gradients_tensor = tf.gradients(self.model.total_loss, weights_tensor)
        input_tensors = self.model.inputs + self.model._targets
        return K.function(inputs=input_tensors, outputs=gradients_tensor)

    def _get_batches(self):
        return self.Y.shape[0] // self.batch_size

    def _get_rest(self):
        return self.Y.shape[0] % self.batch_size

    def _get_iterations(self):
        if self._get_rest() > 0:
            iterations = self._get_batches() + 1
        else:
            iterations = self._get_batches()
        return iterations

    def _get_layer_output(self, layer=1):
        return K.function([self.model.layers[0].input], [self.model.layers[layer].output])

    def _get_layer_outputs(self, X):
        layer_outputs = []
        if self.last_layer_only:
            layer_outputs.append(self._get_layer_output(-1)([X])[0])
            if not self.layers_used:
                self.layers_used.append(self.model.layers[-2].units)
                self.layers_used.append(self.model.layers[-1].units)
        else:
            for i in range(0, len(self.model.layers)):
                layer_outputs.append(self._get_layer_output(i)([X])[0])
                if not self.layers_used:
                    self.layers_used.append(self.model.layers[i].units)

        return np.asarray(layer_outputs)

    @staticmethod
    def write_to_file(file_path, ds_name, payload):
        """
        Write the features to a file
        """
        my_file = Path(file_path)

        if my_file.is_file():
            with h5py.File(file_path, 'a') as hf:
                hf[ds_name].resize((hf[ds_name].shape[0] + 1), axis=0)
                hf[ds_name][hf[ds_name].shape[0] - 1, :] = payload
        else:
            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset(ds_name, data=payload[np.newaxis], compression='gzip', chunks=True, maxshape=(
                    None, payload.shape[0]))
