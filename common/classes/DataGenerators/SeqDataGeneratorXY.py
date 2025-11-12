import tensorflow as tf
import numpy as np


class SeqDataGeneratorXY:
    """Generates timeseries data from two list of shape [data][C...]
    Assuming that there is no offset
    Returns all the timeframes

    Transforms a List of shape (x, c, ...) and (x, c_1, ...) to (batch_size, numb_unrollings, c, ...), (batch_size, numb_unrollings, c_1, ...)

    num_unroll: the number of timeframes to return
    """

    def __init__(self, xs, ys, batch_size, num_unroll, seed=None):
        if len(xs) != len(ys):
            raise UserWarning('xs and ys must have same length')

        self._xs = np.array(xs)
        self._ys = np.array(ys)
        self._xs_length = len(self._xs)
        self._batch_size = batch_size
        self._num_unroll = num_unroll

        if seed is not None:
            np.random.seed(seed)

        self._reset_indices()

    def _next_batch(self):
        batch_data = self._xs[self._cursor]
        batch_labels = self._ys[self._cursor + 1]

        self._cursor = ((self._cursor + 2) % self._xs_length - 1) % self._xs_length

        return batch_data, batch_labels

    def unroll_batches(self):
        unroll_data, unroll_labels = [], []
        for ui in range(self._num_unroll):
            data, labels = self._next_batch()
            unroll_data.append(data)
            unroll_labels.append(labels)

        def transpose_data(to_transpose):
            to_transpose = tf.convert_to_tensor(to_transpose)
            if data.ndim == 1:
                to_transpose = tf.expand_dims(to_transpose, 2)
            return tf.transpose(to_transpose, perm=[1, 0, *list(range(2, to_transpose.ndim))])

        unroll_data = transpose_data(unroll_data)
        unroll_labels = transpose_data(unroll_labels)

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
    data_generator = SeqDataGeneratorXY(
            np.repeat(
                    np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                             dtype=float)
                    , 2).reshape(-1, 2),
            np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                     dtype=float),
            10,
            5,
            seed=42)
    inputs, outputs = data_generator.unroll_batches()
    # inputs = tf.squeeze(inputs)
    # outputs = tf.squeeze(outputs)
    print(inputs, outputs)

    data_generator = SeqDataGeneratorXY(
            np.repeat(
                    np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                             dtype=float)
                    , 2).reshape(-1, 2),
            np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                     dtype=float),
            10,
            5,
            seed=42)
    inputs2, outputs2 = data_generator.unroll_batches()
    print("Reproducable?: ", np.array_equal(inputs2, inputs) and np.array_equal(outputs2, outputs))


if __name__ == '__main__':
    main()
