import json
import os
import numpy as np
import h5py

from .wb_attack_data_generator import WBAttackGenerator


class WBFederatedAttackGenerator:
    """
    For each Epoch of a dataowner generate the attack data
    """

    def __init__(self, models, X, Y, batch_size, train_indices, test_indices,
                 num_classes, last_layer_only=True, one_hot=False):
        self.models = models
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.last_layer_only = last_layer_only
        self.one_hot = one_hot
        self.train_indices = train_indices
        self.test_indices = test_indices
        self.layers_used = None
        self.files = []

    def generate(self, path, name):
        """
        Generate the train and test features for the attack model. For each data owners model/epoch
        """
        i = 0
        for target_model in self.models:
            wb_generator = WBAttackGenerator(target_model, self.X, self.Y,
                                             self.train_indices, self.test_indices,
                                             self.batch_size, self.num_classes, last_layer_only=self.last_layer_only,
                                             one_hot=self.one_hot)
            self.files.append(f'{path}/{name}_{i}')
            wb_generator.write_attack_info(path, f'{name}_{i}')
            assert self.layers_used is None or \
                   self.layers_used == wb_generator.layers_used, \
                "all features should use the same layer layout"
            self.layers_used = wb_generator.layers_used
            i += 1
        assert self.layers_used is not None, "used layer should be defined"

    def merge(self, path, name, files=None, batch_size=100):
        """
        merge the features for each epoch e.g. T = 3
         [[feature, feature ,feature],....]
        """
        if files is not None:
            self.files = files

        assert batch_size < len(self.train_indices), "batch size should be smaller than train size"
        batch_amount = len(self.train_indices) // batch_size

        file_name_train = f'{path}/{name}_merged_train_attack_data.h5'
        file_name_test = f'{path}/{name}_merged_test_attack_data.h5'

        print("saving merged files")

        assert not os.path.exists(file_name_train), "training file already exists"
        assert not os.path.exists(file_name_test), "test file already exists"

        print("Creating train and test data for each target epoch model")
        for x in range(batch_amount):
            attack_data_train = []
            attack_data_test = []
            for file_path in self.files:
                file = open(f'{file_path}_data_inf.json', 'r')
                attack_data_inf = json.load(file)
                beginning = x * batch_size
                end = (x + 1) * batch_size if x < (batch_amount - 1) else None
                attack_data_train.append((h5py.File(attack_data_inf["attack_train_file"], 'r'))['attack_data'][
                                         beginning:end])
                attack_data_test.append((h5py.File(attack_data_inf["attack_test_file"], 'r'))['attack_data'][
                                        beginning:end])
                assert self.layers_used is None or \
                       self.layers_used == attack_data_inf["layers_used"], \
                    "all features should use the same layer layout"
                self.layers_used = attack_data_inf["layers_used"]
            attack_data_train = np.array(attack_data_train)
            attack_data_test = np.array(attack_data_test)
            attack_data_train = attack_data_train.reshape(
                (attack_data_train.shape[1], len(self.files), attack_data_train.shape[2]))
            attack_data_test = attack_data_test.reshape(
                (attack_data_test.shape[1], len(self.files), attack_data_test.shape[2]))

            self.write_to_file(file_name_train, WBAttackGenerator.DS_NAME, attack_data_train)
            self.write_to_file(file_name_test, WBAttackGenerator.DS_NAME, attack_data_test)

        data_inf = {}
        data_inf['last_layer_only'] = self.last_layer_only
        data_inf['attack_train_file'] = file_name_train
        data_inf['attack_test_file'] = file_name_test
        data_inf['layers_used'] = self.layers_used
        data_inf_file = f'{path}/{name}_merged_data_inf.json'
        with open(data_inf_file, 'w') as outfile:
            json.dump(data_inf, outfile, indent=4, sort_keys=True)

    def create_file(self, path, ds_name, payload):
        with h5py.File(path, "a") as f:
            f.create_dataset(ds_name, data=payload, compression='gzip',
                             maxshape=(None, payload.shape[1], payload.shape[2]))

    def write_to_file(self, path, ds_name, payload):
        if not os.path.exists(path):
            return self.create_file(path, ds_name, payload)
        with h5py.File(path, "a") as f:
            f[ds_name].resize((f[ds_name].shape[0] + payload.shape[0]), axis=0)
            f[ds_name][-payload.shape[0]:] = payload
