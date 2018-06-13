from models import future_direction_conv
import numpy as np


def make_labels(x, y):
    price_diff = y[:, -1, 0] - x[:, -1, 0]
    return price_diff > 0


if __name__ == '__main__':
    x_train = np.load('cache/x_train.npy')
    y_raw = np.load('cache/y_train.npy')
    y_train = make_labels(x_train, y_raw)

    x_valid = np.load('cache/x_test.npy')
    y_valid = np.load('cache/y_test.npy')
    y_valid = make_labels(x_valid, y_valid)

    model = future_direction_conv(x_train.shape)
    train_history = model.fit(
        x_train, y_train, epochs=1, batch_size=128, shuffle=True,
        validation_data=(x_valid, y_valid)
    )
    print("\nTraining complete!\n")
