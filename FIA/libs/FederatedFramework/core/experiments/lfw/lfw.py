from math import sqrt

import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import Model as KModel
from tensorflow.keras.regularizers import l2
from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPAdamGaussianOptimizer
import tensorflow as tf
from ...entities.datasource import Datasource
from ...entities.experiment import Experiment
from ...entities.global_model import GlobalModel
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam


class GlobalModel_Lfw(GlobalModel):
    """
    A Global model for Purchases classification task
    """

    def __init__(self, output_size=50):
        super(GlobalModel_Lfw, self).__init__(output_size)

    def build_model(self):
        """
        Example Architecture for Purchases
        :return:
        """
        input_shape = (250, 250, 3)
        hidden_dim = 4096
        activation = 'relu'

        vgg_model = self.VGGModel(input_shape)
        for layer in vgg_model.layers:
            layer.trainable = False
        vgg_model.trainabe = False
        last_layer = vgg_model.get_layer('pool5').output
        x = Flatten(name='flatten')(last_layer)
        x = Dense(hidden_dim, activation=activation, kernel_regularizer=l2(0), name='fc6')(x)
        x = Dense(hidden_dim, activation=activation, kernel_regularizer=l2(0), name='fc7')(x)
        out = Dense(self.output_size, activation='softmax', name='fc8')(x)
        m = KModel(vgg_model.input, out)
        loss = SparseCategoricalCrossentropy(from_logits=False)

        m.compile(loss=loss,
                  optimizer=Adam(lr=0.001),
                  metrics=['accuracy'])
        return m

    def VGGModel(self, input_shape):
        img_input = Input(shape=input_shape)

        # Block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1')(
            img_input)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')(
            x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(
            x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(
            x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(
            x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3')(
            x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)

        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(
            x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(
            x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3')(
            x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x)

        # Block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(
            x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(
            x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3')(
            x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(x)

        model = KModel(img_input, x, name='vggface_vgg16')
        model.load_weights('./models/rcmalli_vggface_tf_notop_vgg16.h5', by_name=True)

        return model


class Lfw(Datasource):
    """
    Some Helper Methods to prepare lfw Data
    """

    def __init__(self, alternative_dataset=None):
        """
        Retrieve lfw Data
        """

        super().__init__()
        print("load datasource lfw")
        DATA = "./data/lfw/lfw_50_classes.npz"
        if alternative_dataset is not None:
            DATA = alternative_dataset
        data = np.load(DATA, allow_pickle=True)
        x = data['x']
        y = data['y']
        x = x.astype(np.float32) / 255
        l = int(sqrt(len(x[0])))  # in some LDP cases we work with smaller images
        x = np.reshape(x, (-1, l, l, 1))
        x = np.tile(x, (1, 1, 1, 3))

        y = y.astype(np.int8)
        y = y.flatten()
        self.x = x
        self.y = y
        print("datasource data loaded")

    def disjoint_dataset_indices(self, position, num_clients, seed):
        """
        Get disjoint data sets for each party.
        It is important to set the seed and the amount of clients for each party to the same value
        otherwise it is not guaranteed that the sets are disjoint
        """
        proportion = 1 / 8
        np.random.seed(seed)
        indices = np.random.choice(self.x.shape[0], size=self.x.shape[0], replace=False)
        do_indices, ag_indices = train_test_split(indices, test_size=proportion)
        if position == 0:
            return ag_indices

        batch_size = len(do_indices) // (num_clients - 1)
        position = position - 1  # minus the aggregator
        return do_indices[position * batch_size:(position + 1) * batch_size]


class LfwExperiment(Experiment):
    """
    The purchases100 100 experiment
    """

    def __init__(self, args):
        self.args = args
        super().__init__(self.get_optimizer(), Lfw(args["alternative_dataset"]),
                         lambda: GlobalModel_Lfw(args["output_size"]))

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
