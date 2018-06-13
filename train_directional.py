import keras
from keras.callbacks import ModelCheckpoint

from models import future_direction_conv
import numpy as np

from settings import *


def make_labels(x, y):
    price_diff = y[:, -1, 0] - x[:, -1, 0]
    return price_diff > 0


class SaveModel(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        acc = int(logs['val_acc'] * 100)
        if acc > 60:
            self.model.save('assets/direction_e{}_acc{}.h5'.format(epoch, acc))


if __name__ == '__main__':
    x_train = np.load('cache/x_train.npy')
    y_raw = np.load('cache/y_train.npy')
    y_train = make_labels(x_train, y_raw)
    # x_train = x_train[:1000]
    # y_train = y_train[:1000]

    x_valid = np.load('cache/x_test.npy')
    y_valid = np.load('cache/y_test.npy')
    y_valid = make_labels(x_valid, y_valid)
    # x_valid = x_valid[:1000]
    # y_valid = y_valid[:1000]

    ensure_dir_exists(os.path.join(ROOT_DIR, 'assets'))

    model = future_direction_conv(x_train.shape)

    checkpointer = SaveModel()
    train_history = model.fit(
        x_train, y_train, epochs=50, batch_size=128, shuffle=True,
        validation_data=(x_valid, y_valid),
        callbacks=[checkpointer]
    )
    print("\nTraining complete!\n")
    model.save('assets/direction_final.h5')
