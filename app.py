from prep import *


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
        x, y = pair.stacked()

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
    x_btc, y_btc = generate_train_data(counter='btc')
    x_usdt, y_usdt = generate_train_data(counter='usdt')
    x_eth, y_eth = generate_train_data(counter='eth')

    x = np.concatenate([x_btc, x_usdt, x_eth])
    y = np.concatenate([y_btc, y_usdt, y_eth])

    x_train, y_train, x_valid, y_valid = split(x, y, ratio=.9)
    print(x_train.shape, y_train.shape)
    print(x_valid.shape, y_valid.shape)
    cache(x_train, 'x_train')
    cache(y_train, 'y_train')
    cache(x_valid, 'x_valid')
    cache(y_valid, 'y_valid')

