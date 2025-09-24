import tensorflow as tf
import numpy as np


class DataGeneratorSeq:
    """Generates timeseries data from a list of shape [data][C...]

    num_unroll: the number of timeframes to return
    """

    def __init__(self, prices, batch_size, num_unroll):
        self._prices = np.array(prices)
        self._prices_length = len(self._prices)
        self._batch_size = batch_size
        self._num_unroll = num_unroll
        self._reset_indices()

    def _next_batch(self):
        batch_data = self._prices[self._cursor]
        batch_labels = self._prices[self._cursor + 1]

        self._cursor = ((self._cursor + 2) % self._prices_length - 1) % self._prices_length

        return batch_data, batch_labels

    def unroll_batches(self):
        unroll_data, unroll_labels = [], []
        for ui in range(self._num_unroll):
            data, labels = self._next_batch()
            unroll_data.append(data)
            unroll_labels = labels

        unroll_data = tf.expand_dims(tf.convert_to_tensor(unroll_data), 2)
        unroll_data = tf.transpose(unroll_data, perm=[1, 0, 2])
        unroll_labels = tf.expand_dims(tf.convert_to_tensor(unroll_labels), axis=1)

        self._reset_indices()
        return unroll_data, unroll_labels

    def _reset_indices(self):
        self._cursor = np.random.randint(0, self._prices_length - self._num_unroll, self._batch_size)

    def as_generator(self, epochs=None):
        def generator():
            for _ in range(epochs):
                yield self.unroll_batches()

        def infinite_generator():
            while True:
                yield self.unroll_batches()

        return infinite_generator if epochs is None else generator


def main():
    data_generator = DataGeneratorSeq(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float), 5, 5)
    inputs, outputs = data_generator.unroll_batches()
    inputs = tf.squeeze(inputs)
    outputs = tf.squeeze(outputs)
    print(inputs, outputs)


if __name__ == '__main__':
    main()
