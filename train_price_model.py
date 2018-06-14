import keras
import numpy as np
from keras.models import load_model

from models import future_price_conv, continuous_price_model
from settings import *


def make_labels(x, y):
    price_diff = y[:, -1, 0] - x[:, -1, 0]
    return price_diff


class SaveModel(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        acc = int(logs['val_directional_accuracy'] * 100)
        if acc > 60:
            self.model.save('assets/price_e{}_acc{}.h5'.format(epoch, acc))


def train_from_scratch(x_train, y_train, x_valid, y_valid):
    model = future_price_conv(x_train.shape)
    cb_save = SaveModel()
    train_history = model.fit(
        x_train, y_train, epochs=50, batch_size=128, shuffle=True,
        validation_data=(x_valid, y_valid),
        callbacks=[cb_save]
    )
    print("\nTraining complete!\n")
    model.save('assets/price_final.h5')


def transfer_from_directional(x_train, y_train, x_valid, y_valid):
    directional_model_name = 'directional_61.h5'
    directional_model_path = 'assets/{}'.format(directional_model_name)
    pretrained = load_model(directional_model_path)
    model = continuous_price_model(pretrained, mode='transfer')
    cb_save = SaveModel()
    train_history = model.fit(
        x_train, y_train, epochs=50, batch_size=128, shuffle=True,
        validation_data=(x_valid, y_valid),
        callbacks=[cb_save]
    )
    print("\nTraining complete!\n")
    model.save('assets/price_final.h5')


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

    # train_from_scrach(x_train, y_train, x_valid, y_valid)
    transfer_from_directional(x_train, y_train, x_valid, y_valid)
