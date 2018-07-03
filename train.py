import keras
from keras.callbacks import ModelCheckpoint, History

from models import *
import numpy as np

from settings import *


class SaveModel(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        acc = int(logs['val_acc'] * 100)
        if acc > 60:
            self.model.save('assets/incep_e{}_acc{}.h5'.format(epoch, acc))


def make_labels(x, y):
    price_diff = y[:, -1, 1] - x[:, -1, 1]
    return price_diff > 0


def make_price_targets(x, y):
    price_diff = y[:, -1, 1] - x[:, -1, 1]
    return price_diff


def read_data():
    print("Preparing training set...")
    xs, ys = [], []
    for i in range(4):
        x = np.load('cache/x_train{}.npy'.format(i + 1))
        y = np.load('cache/y_train{}.npy'.format(i + 1))
        xs.append(x)
        ys.append(y)
    inputs = np.concatenate(xs)
    outputs = np.concatenate(ys)
    del xs, ys
    x_train = inputs
    y_train = make_labels(x_train, outputs)
    print("Training set ready.\nPreparing validation set...")
    print("Validation set ready.\n")


def train_directional():
    pass


if __name__ == '__main__':
    print("Preparing training set...")
    xs, ys = [], []
    for i in range(4):
        x = np.load('cache/x_train{}.npy'.format(i + 1))
        y = np.load('cache/y_train{}.npy'.format(i + 1))
        xs.append(x)
        ys.append(y)
    inputs = np.concatenate(xs)
    outputs = np.concatenate(ys)
    del xs, ys
    x_train = inputs
    y_train = make_labels(x_train, outputs)
    print("Training set ready. \nPreparing validation set...")

    x_valid = np.load('cache/x_valid.npy')
    y_valid = np.load('cache/y_valid.npy')
    y_valid = make_labels(x_valid, y_valid)
    print("Validation set ready.")

    ensure_dir_exists(os.path.join(ROOT_DIR, 'assets'))

    model = direction_inception_model(x_train.shape, .2)

    checkpointer = SaveModel()
    train_history = model.fit(
        x_train, y_train, epochs=100, batch_size=512, shuffle=True,
        validation_data=(x_valid, y_valid),
        callbacks=[checkpointer]
    )

    print("\nTraining complete!\n")
    model.save('assets/inception_final.h5')

    print(train_history.history)
