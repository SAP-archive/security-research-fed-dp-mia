import h5py
import numpy as np
from tensorflow.keras.utils import Sequence

from .wb_attack_data_generator import WBAttackGenerator


class WBAttackLoader(Sequence):
    """
    Generator class for tensorflow fit_Generator function.
    This is needed because the Training data is too large to fit into RAM
    The Generator class just ready a batch from the disk incrementally
    """

    def __init__(self, train_data_file, test_data_file, train_indices, test_indices, batch_size,
                 load_in_memory=False, shuffle=True):
        """
        Initialize Loader. Compared to Steffens code much has been removed,
        simply due to the fact that we know certainly that only the last layer with all its components are used
        """
        self.train_data_file = train_data_file
        self.test_data_file = test_data_file
        self.train_indices = train_indices
        self.test_indices = test_indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.load_in_memory = load_in_memory
        if self.load_in_memory:
            with h5py.File(self.train_data_file, 'r') as train_file, h5py.File(self.test_data_file, 'r') as test_file:
                self.train_data = self._get_data(None, file=train_file)
                self.test_data = self._get_data(None, file=test_file)
        assert self.get_train_amount() >= max(train_indices), f"train indices contain index out of range {max(train_indices)} > {self.get_train_amount()}"
        assert self.get_test_amount() >= max(test_indices), f"test indices contain index out of range {max(test_indices)} > {self.get_test_amount()}"

    def __len__(self):
        return int(np.floor(len(self.train_indices) * 2 / self.batch_size))

    def _get_data(self, ind, file):
        """
        Read the Data from the File.
        Subsequent index reads must be consecutive (hdfs restriction)
        """
        if ind is None:  # is no index is given, load all data
            return np.array(file[WBAttackGenerator.DS_NAME][:])
        return np.array(file[WBAttackGenerator.DS_NAME][ind])

    def __getitem__(self, idx):
        """
        Get a batch
        """
        start_i = idx * self.batch_size // 2
        end_i = (idx + 1) * self.batch_size // 2
        train_indices = self.train_indices[start_i: end_i]
        test_indices = self.test_indices[start_i: end_i]
        if self.load_in_memory:
            batch_x = []
            batch_y = []
            for train_i, test_i in zip(train_indices, test_indices):
                batch_x.append(self.train_data[train_i])
                batch_y.append(1)
                batch_x.append(self.test_data[test_i])
                batch_y.append(0)
        else:
            with h5py.File(self.train_data_file, 'r') as train_file, h5py.File(self.test_data_file, 'r') as test_file:
                batch_x = []
                batch_y = []
                for train_i, test_i in zip(train_indices, test_indices):
                    batch_x.append(self._get_data(train_i, train_file))
                    batch_y.append(1)
                    batch_x.append(self._get_data(test_i, test_file))
                    batch_y.append(0)
        return np.array(batch_x), np.array(batch_y)

    def on_epoch_end(self):
        """
        Shuffle when epoch ends
        """
        if self.shuffle:
            # Updates indexes after each epoch
            np.random.shuffle(self.train_indices)
            np.random.shuffle(self.test_indices)

    def get_input_size(self):
        """
        Input size of the training data.
        Steffen calculated that by: weights*layers+num_classes+1
        """
        train_file = h5py.File(self.train_data_file, 'r')
        print(train_file['attack_data'].shape)
        return train_file['attack_data'][0].shape[1]

    def get_train_amount(self):
        """
        get amount of training data
        """
        train_file = h5py.File(self.train_data_file, 'r')
        return train_file['attack_data'].shape[0]

    def get_test_amount(self):
        """
        get amount of test data
        """
        test_file = h5py.File(self.test_data_file, 'r')
        return test_file['attack_data'].shape[0]

    def get_indices(self):
        indices = []
        for train_i, test_i in list(zip(self.train_indices, self.test_indices)):
            indices.append([1, train_i])
            indices.append([0, test_i])

        return indices
