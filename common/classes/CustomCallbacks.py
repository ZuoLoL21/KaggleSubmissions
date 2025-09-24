from tensorflow.keras.callbacks import Callback


class BatchMetricsCallback(Callback):
    def __init__(self):
        super().__init__()
        self.batch_losses = []
        self.batch_maes = []

    def on_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs.get('loss'))
        self.batch_maes.append(logs.get('mae'))
