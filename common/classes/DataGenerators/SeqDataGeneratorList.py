import tensorflow as tf
import numpy as np


class SeqDataGeneratorList:
    """Generates timeseries data from a list of shape [data][C...]

    Transforms a List of shape (x) to (batch_size, numb_unrollings, 1)

    num_unroll: the number of timeframes to return
    """

    def __init__(self, xs, batch_size, num_unroll):
        self._xs = np.array(xs)
        self._xs_length = len(self._xs)
        self._batch_size = batch_size
        self._num_unroll = num_unroll
        self._reset_indices()

    def _next_batch(self):
        batch_data = self._xs[self._cursor]
        batch_labels = self._xs[self._cursor + 1]

        self._cursor = ((self._cursor + 2) % self._xs_length - 1) % self._xs_length

        return batch_data, batch_labels

    def unroll_batches(self):
        unroll_data, unroll_labels = [], []
        for ui in range(self._num_unroll):
            data, labels = self._next_batch()
            unroll_data.append(data)
            unroll_labels = labels

        unroll_data = tf.convert_to_tensor(unroll_data)
        if unroll_data.ndim == 2:
            unroll_data = tf.expand_dims(unroll_data, 2)
        unroll_data = tf.transpose(unroll_data, perm=[1, 0, *list(range(2, unroll_data.ndim))])
        unroll_labels = tf.expand_dims(tf.convert_to_tensor(unroll_labels), axis=1)

        self._reset_indices()
        return unroll_data, unroll_labels

    def _reset_indices(self):
        self._cursor = np.random.randint(0, self._xs_length - self._num_unroll, self._batch_size)

    def as_generator(self, epochs=None):
        def generator():
            for _ in range(epochs):
                yield self.unroll_batches()

        def infinite_generator():
            while True:
                yield self.unroll_batches()

        return infinite_generator if epochs is None else generator


def main():
    data_generator = SeqDataGeneratorList(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float), 10, 5)
    inputs, outputs = data_generator.unroll_batches()
    # inputs = tf.squeeze(inputs)
    # outputs = tf.squeeze(outputs)
    print(inputs, outputs)


if __name__ == '__main__':
    main()
