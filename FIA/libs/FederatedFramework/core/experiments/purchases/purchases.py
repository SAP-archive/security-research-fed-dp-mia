import numpy as np
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPAdamGaussianOptimizer
import tensorflow as tf

from ...entities.datasource import Datasource
from ...entities.experiment import Experiment
from ...entities.global_model import GlobalModel


class GlobalModel_Purchases(GlobalModel):
    """
    A Global model for Purchases classification task
    """

    def __init__(self, output_size=100):
        super(GlobalModel_Purchases, self).__init__(output_size)

    def build_model(self):
        """
        Example Architecture for Purchases
        :return:
        """
        model = Sequential()
        model.add(Dense(1024, activation="tanh", input_dim=600, kernel_regularizer=l2(0)))
        model.add(Dense(512, activation="tanh", kernel_regularizer=l2(0)))
        model.add(Dense(256, activation="tanh", kernel_regularizer=l2(0)))
        model.add(Dense(128, activation="tanh", kernel_regularizer=l2(0)))
        model.add(Dense(self.output_size, activation='softmax'))
        loss = SparseCategoricalCrossentropy(from_logits=False)

        model.compile(loss=loss,
                      optimizer=Adam(lr=0.001),
                      metrics=['accuracy'])
        return model


class Purchases(Datasource):
    """
    Some Helper Methods to prepare purchases100 Data
    """

    def __init__(self, alternative_dataset=None):
        """
        Retrieve purchases100 Data
        """

        super().__init__()
        print("load datasource purchases")
        DATA = "./data/purchases/100/shokri_purchases_100_classes.npz"
        if alternative_dataset is not None:
            DATA = alternative_dataset
        data = np.load(DATA, allow_pickle=True)
        self.x = data['x']
        self.y = data['y']
        print("datasource data loaded")

    def disjoint_dataset_indices(self, position, num_clients, seed):
        """
        Get disjoint data sets for each party.
        It is important to set the seed and the amount of clients for each party to the same value
        otherwise it is not guaranteed that the sets are disjoint
        """
        # Shokri uses 20k trainingsdata and 50k testdata for the central case. That means a proportion of 2/7
        proportion = 5 / 7
        np.random.seed(seed)
        indices = np.random.choice(self.x.shape[0], size=self.x.shape[0], replace=False)
        do_indices, ag_indices = train_test_split(indices, test_size=proportion)
        if position == 0:
            return ag_indices

        batch_size = len(do_indices) // (num_clients - 1)
        position = position - 1  # minus the aggregator
        return do_indices[position * batch_size:(position + 1) * batch_size]


class PurchasesExperiment(Experiment):
    """
    The purchases100 100 experiment
    """

    def __init__(self, args):
        self.args = args
        super().__init__(self.get_optimizer(), Purchases(args["alternative_dataset"]),
                         lambda: GlobalModel_Purchases(args["output_size"]))

    def get_optimizer(self):
        optimizer = None
        if self.args["noise_multiplier"] == 0:
            loss = SparseCategoricalCrossentropy(from_logits=False)
            optimizer = lambda: keras.optimizers.Adam(lr=self.args["learning_rate"])
        else:
            loss = SparseCategoricalCrossentropy(from_logits=False, reduction=tf.compat.v1.losses.Reduction.NONE)
            optimizer = lambda: DPAdamGaussianOptimizer(learning_rate=self.args["learning_rate"],
                                                        l2_norm_clip=self.args["norm_clip"],
                                                        noise_multiplier=self.args["noise_multiplier"],
                                                        num_microbatches=self.args["batch_size"],
                                                        unroll_microbatches=True)
        return {'loss': loss,
                'optimizer': optimizer,
                'metrics': ['accuracy']}
