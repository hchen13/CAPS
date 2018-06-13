import keras
import numpy as np

from models import future_price_conv
from settings import *


def make_labels(x, y):
    price_diff = y[:, -1, 0] - x[:, -1, 0]
    return price_diff


class SaveModel(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(logs)
        acc = int(logs['val_directional_accuracy'] * 100)
        if acc > 60:
            self.model.save('assets/price_e{}_acc{}.h5'.format(epoch, acc))


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

    model = future_price_conv(x_train.shape)

    cb_save = SaveModel()

    train_history = model.fit(
        x_train, y_train, epochs=50, batch_size=128, shuffle=True,
        validation_data=(x_valid, y_valid),
        callbacks=[cb_save]
    )
    print("\nTraining complete!\n")
