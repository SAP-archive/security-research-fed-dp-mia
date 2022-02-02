from tensorflow.keras import layers
from tensorflow import keras
import numpy as np


class WBAttackLayer(layers.Layer):
    """
    Custom Keras Layer
    WB Attacker Model architecture according to Nasr et al.
    """

    def __init__(self, target_layer_config, num_classes, disabled_components, dropout_rate, weight_initializer,
                 batch_norm=False, stacked_epochs=1, conv_height=5, **kwargs):
        """
            target_layer_config
            #classes
            disabled_components
            dropout_rate
            weight_initializer
            batch_norm
        """
        assert num_classes > 0, "num classes must be greater 0"
        assert not target_layer_config == [], "an empty target layer config is useless"
        super(WBAttackLayer, self).__init__(**kwargs)
        self.target_layer_config = target_layer_config
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.weight_initializer = weight_initializer
        self.batch_norm = batch_norm
        self.disabled_components_a = disabled_components
        self.stacked_epochs = stacked_epochs
        self.conv_height = conv_height
        self.disabled_components = {'gradient': False, 'output': False, 'loss': False, 'label': False}
        if disabled_components is not None and disabled_components != []:
            for dc in disabled_components:
                if dc in self.disabled_components.keys():
                    self.disabled_components[dc] = True
                else:
                    print('Warning: ' + dc + ' is not a valid component')

        self.gradient_components = {}
        self.output_components = {}

        self.gradient_offsets = [0]
        self.output_offsets = [0]

        self.total_gradients = 0
        self.total_outputs = 0

        # for each layer in target_layer_config
        for i, layer_conf in enumerate(target_layer_config):
            # skip input layer
            if i == 0:
                continue

            if not self.disabled_components['gradient']:
                self.create_gradient_component(i, layer_conf)

            if not self.disabled_components['output']:
                self.create_output_component(i, layer_conf)

            self.gradient_offsets.append(self.total_gradients)
            self.output_offsets.append(self.total_outputs)

        # input size
        self.input_size = self.total_gradients + self.total_outputs

        if not self.disabled_components['loss']:
            self.create_loss_component()

        if not self.disabled_components['label']:
            self.create_label_component()

    def create_label_component(self):
        """
          label component config
        """
        self.label_l1 = layers.Dense(128, activation=layers.LeakyReLU(alpha=0.1),
                                     kernel_initializer=self.weight_initializer, name="label_dense1")
        self.label_d1 = layers.Dropout(self.dropout_rate)
        self.label_l2 = layers.Dense(64, activation=layers.LeakyReLU(alpha=0.1),
                                     kernel_initializer=self.weight_initializer,
                                     name="label_dense2")
        self.input_size += self.num_classes

    def create_loss_component(self):
        """
        loss component config
        """
        self.loss_l1 = layers.Dense(128, activation=layers.LeakyReLU(alpha=0.1),
                                    kernel_initializer=self.weight_initializer, name="loss_dense1")
        self.loss_d1 = layers.Dropout(self.dropout_rate)
        self.loss_l2 = layers.Dense(64, activation=layers.LeakyReLU(alpha=0.1),
                                    kernel_initializer=self.weight_initializer,
                                    name="loss_dense2")
        self.input_size += 1

    def create_output_component(self, i, layer_conf):
        """
        output component
        """
        self.total_outputs += layer_conf
        # create FCN component
        output_component = []
        output_component.append(layers.Flatten(input_shape=(self.stacked_epochs, layer_conf)))
        output_component.append(layers.Dense(128, activation=layers.LeakyReLU(alpha=0.1),
                                             kernel_initializer=self.weight_initializer, name="output_dense1"))
        output_component.append(layers.Dropout(self.dropout_rate))
        output_component.append(
            layers.Dense(64, activation=layers.LeakyReLU(alpha=0.1), kernel_initializer=self.weight_initializer,
                         name="output_dense2"))
        # output_component.append(layers.Dropout(dropout_rate))
        self.output_components[str(i)] = output_component

    def create_gradient_component(self, i, layer_conf):
        """
        number of gradients:
        neurons_previous_layer * #neurons_current_layer + bias per neuron in current layer)
        """
        dim1 = self.target_layer_config[i - 1]
        dim2 = layer_conf
        num_gradients = (dim1 + 1) * dim2
        self.total_gradients += num_gradients
        # create CNN component
        gradient_component = []
        gradient_component.append(
            layers.Lambda(lambda x: keras.backend.permute_dimensions(x, (0, 2, 1)),
                          name="transpose_layer",
                          input_shape=(self.stacked_epochs, num_gradients)))
        gradient_component.append(
            layers.Reshape((dim1 + 1, dim2, self.stacked_epochs),
                           name="grad_reshape1"))
        gradient_component.append(
            layers.Conv2D(filters=1000, kernel_size=(dim1 + 1, self.conv_height),
                          kernel_initializer=self.weight_initializer,
                          name="grad_conv1"))
        gradient_component.append(layers.Flatten())
        if self.batch_norm:
            gradient_component.append(layers.BatchNormalization())
        gradient_component.append(layers.Dropout(self.dropout_rate))  # this one should always be here
        gradient_component.append(
            layers.Dense(128, activation=layers.LeakyReLU(alpha=0.1), kernel_initializer=self.weight_initializer,
                         name="grad_dense1"))
        gradient_component.append(layers.Dropout(self.dropout_rate))
        gradient_component.append(
            layers.Dense(64, activation=layers.LeakyReLU(alpha=0.1), kernel_initializer=self.weight_initializer,
                         name="grad_dense2"))
        gradient_component.append(layers.Dropout(self.dropout_rate))
        self.gradient_components[str(i)] = gradient_component

    def get_config(self):
        """
        necessary so that Keras can save the model
        """

        base_config = super(WBAttackLayer, self).get_config()
        config = {
            'target_layer_config': self.target_layer_config,
            'num_classes': self.num_classes,
            'dropout_rate': self.dropout_rate,
            'weight_initializer': self.weight_initializer,
            'batch_norm': self.batch_norm,
            'disabled_components': self.disabled_components_a,
            'stacked_epochs': self.stacked_epochs
        }
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs, training=False):
        """
        # INPUT
        # |------------------------|..................|---------------|..................|--------|-------|
        # | grads weights & biases | (for each layer) | layer output  | (for each layer) |  loss  | label |
        # |------------------------|..................|---------------|..................|--------|-------|
        #
        """
        gradient_component_outputs = []
        output_component_outputs = []

        # gradient and output components for each layer
        for i, layer_conf in enumerate(self.target_layer_config):
            # skip input layer
            if i == 0:
                continue

            # gradient component
            if not self.disabled_components['gradient']:
                gradient_offset = self._get_offset_gradient(i)
                l_in = layers.Lambda(lambda x: x[:, :, gradient_offset[0]:gradient_offset[1]])(inputs)
                gc = self.gradient_components[str(i)]
                for l in gc:
                    if isinstance(l, layers.Dropout) or isinstance(l, layers.BatchNormalization):
                        l_in = l(l_in, training=training)
                    else:
                        l_in = l(l_in)

                gradient_component_outputs.append(l_in)

            # output component
            if not self.disabled_components['output']:
                output_offset = self._get_offset_output(i)
                l_in = layers.Lambda(lambda x: x[:, :, output_offset[0]:output_offset[1]])(inputs)
                oc = self.output_components[str(i)]
                for l in oc:
                    if isinstance(l, layers.Dropout) or isinstance(l, layers.BatchNormalization):
                        l_in = l(l_in, training=training)
                    else:
                        l_in = l(l_in)

                output_component_outputs.append(l_in)

        merged_output = gradient_component_outputs + output_component_outputs

        # loss component
        if not self.disabled_components['loss']:
            num_layers = len(self.target_layer_config)
            loss_offset = self._get_offset_output(num_layers - 1)[1]
            l_in = layers.Lambda(lambda x: x[:, loss_offset:loss_offset + 1])(inputs)
            l_in = layers.Flatten(input_shape=(self.stacked_epochs, 1))(l_in)
            l_in = self.loss_l1(l_in)
            l_in = self.loss_d1(l_in, training=training)
            l_out = self.loss_l2(l_in)
            merged_output.append(l_out)

        # label component
        if not self.disabled_components['label']:
            num_layers = len(self.target_layer_config)
            label_offset = self._get_offset_output(num_layers - 1)[1] + 1
            l_in = layers.Lambda(lambda x: x[:, label_offset:self.input_size - 1])(inputs)
            l_in = layers.Flatten(input_shape=(self.stacked_epochs, self.num_classes))(l_in)
            l_in = self.label_l1(l_in)
            l_in = self.label_d1(l_in, training=training)
            l_out = self.label_l2(l_in)
            merged_output.append(l_out)

        if len(merged_output) > 1:
            return layers.Concatenate()(merged_output)

        else:
            return merged_output[0]

    def _get_offset_gradient(self, layer_id):
        if layer_id == 0:
            return (0, 0)
        return self.gradient_offsets[layer_id - 1], self.gradient_offsets[layer_id]

    def _get_offset_output(self, layer_id):
        if layer_id == 0:
            return (0, 0)
        start = self.gradient_offsets[-1] + self.output_offsets[layer_id - 1]
        end = self.gradient_offsets[-1] + self.output_offsets[layer_id]
        return start, end
