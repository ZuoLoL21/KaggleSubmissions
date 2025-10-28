from tensorflow.keras.callbacks import Callback


class BatchMetricsCallback(Callback):
    def __init__(self):
        super().__init__()
        self.batch_losses = []
        self.batch_maes = []

    def on_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs.get('loss'))
        self.batch_maes.append(logs.get('mae'))


class BatchMetricsCallbackVal(Callback):
    def __init__(self):
        super().__init__()
        self.batch_losses = []
        self.batch_maes = []
        self.val_losses = []
        self.val_maes = []

    def on_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs.get('loss'))
        self.batch_maes.append(logs.get('mae'))

    def on_epoch_end(self, epoch, logs=None):
        self.val_losses.append(logs.get('val_loss'))
        self.val_maes.append(logs.get('val_mae'))
