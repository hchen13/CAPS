from models import *
from prep import *
from matplotlib import pyplot as plt

dataset = Dataset()

def generate_train_data(base=None, counter=None):
    coin_list = dataset.fetch(base=base, counter=counter)
    if not len(coin_list):
        print("There are no corresponding datasets.")
        return None, None
    inputs, targets = [], []
    print("{} coin pairs found. Generating datasets...".format(len(coin_list)))
    for i, pair in enumerate(coin_list):
        print("Accessing data: {}/{}...".format(pair.base, pair.counter))
        x, y = pair.stacked(future_length=12)

        if x.ndim != 3 or y.ndim != 3:
            continue

        if x.shape[0] < 20:
            continue

        inputs.append(x)
        targets.append(y)

    x = np.concatenate(inputs)
    y = np.concatenate(targets)
    return x, y


def split(x, y, ratio=.8, random=True):
    size = len(x)
    train_size = int(ratio * size)
    if random:
        x, y = shuffle(x, y)
    x_train, x_valid = x[:train_size], x[train_size:]
    y_train, y_valid = y[:train_size], y[train_size:]
    return x_train, y_train, x_valid, y_valid



if __name__ == '__main__':
    """Training data generation"""
    # x_btc, y_btc = generate_train_data(counter='btc')
    # x_usdt, y_usdt = generate_train_data(counter='usdt')
    # x_eth, y_eth = generate_train_data(counter='eth')
    #
    #
    # x = np.concatenate([x_btc, x_usdt, x_eth])
    # y = np.concatenate([y_btc, y_usdt, y_eth])
    #
    # x_train, y_train, x_valid, y_valid = split(x, y, ratio=.9)
    # print(x_train.shape, y_train.shape)
    # print(x_valid.shape, y_valid.shape)
    # cache(x_train, 'x_train')
    # cache(y_train, 'y_train')
    # cache(x_valid, 'x_valid')
    # cache(y_valid, 'y_valid')
    """ end """

    x_train = np.load('cache/x_train.npy')
    y_train = np.load('cache/y_train.npy')
    mini_x = x_train[:10_000]
    mini_y = y_train[:10_000]

    model = future_price_conv(mini_x.shape)
    prices_train = mini_y[:, -1, 0]
    price_diff = prices_train - mini_x[:, -1, 0]
    train_history = model.fit(mini_x, price_diff, epochs=1, batch_size=64, shuffle=True)


    # model = future_direction_conv(mini_x.shape)
    # prices_train = mini_y[:, -1, 0]
    # price_diff = prices_train - mini_x[:, -1, 0]
    # labels = price_diff > 0
    # train_history = model.fit(mini_x, labels, epochs=1, batch_size=128, shuffle=True)
    # print('\nTraining complete!\n')
    #
    # del x_train, y_train
    #
    # x_valid = np.load('cache/x_valid.npy')
    # y_valid = np.load('cache/y_valid.npy')
    #
    # last_prices = x_valid[:, -1, 0]
    # prices_valid = y_valid[:, -1, 0]
    # diff_valid = prices_valid - last_prices
    # dir_valid = diff_valid > 0
    #
    # dir_pred = model.predict(x_valid, verbose=True)
    #
    # dir_pred = dir_pred.squeeze()
    #
    # dir_pred = dir_pred > .5
    #
    # correct_dirs = np.sum(dir_valid == dir_pred)
    # acc = correct_dirs / len(dir_valid)
    #
    # print("Direction Accuracy: {:.3f}%".format(acc * 100))

