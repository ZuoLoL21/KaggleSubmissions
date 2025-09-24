import tensorflow as tf
import numpy as np


class LSTMStatefulModel:
    def __init__(self, original_model: tf.keras.Sequential):
        input_shape = (1,) + original_model.input_shape[1:]
        print(input_shape)

        # Create input with fixed batch size
        inputs = tf.keras.Input(batch_shape=input_shape)

        x = inputs

        # Process each layer
        for layer in original_model.layers:
            if isinstance(layer, tf.keras.layers.LSTM):
                # Get all config parameters and set stateful=True
                config = layer.get_config()
                config['stateful'] = True
                # Create new LSTM with all original parameters plus stateful=True
                new_layer = tf.keras.layers.LSTM(**config)
                x = new_layer(x)
            else:
                # Clone other layers using their config
                config = layer.get_config()
                new_layer = layer.__class__(**config)
                x = new_layer(x)

        # Build the new model
        self.stateful_model = tf.keras.Model(inputs=inputs, outputs=x)

        # Transfer all weights
        self.stateful_model.set_weights(original_model.get_weights())

    def warmup(self, warmup_inputs):
        # print(warmup_inputs.shape)
        to_return = self.stateful_model.predict(warmup_inputs, verbose=0)
        # print(to_return.shape)
        return tf.expand_dims(to_return, 0)

    def predict(self, latest_steps, num_to_predict):
        future = []

        for i in range(num_to_predict):
            latest_step = self.stateful_model.predict(latest_steps, verbose=False)
            latest_step = tf.expand_dims(latest_step, 0)
            future.append(latest_step)  # store the future steps

        return np.squeeze(np.array(future))

    def reset_lstm_states(self):
        for i, layer in enumerate(self.stateful_model.layers):
            if isinstance(layer, tf.keras.layers.LSTM):
                layer.reset_states()
