from prep import *


def generate_train_data():
    dataset = Dataset()
    coin_list = dataset.fetch(counter='usdt')
    inputs, targets = [], []
    for pair in coin_list:
        x, y = pair.stacked()
        inputs.append(x)
        targets.append(y)
    if not len(inputs):
        print("Dataset is empty")
        return
    x = np.concatenate(inputs)
    y = np.concatenate(targets)
    return x, y


if __name__ == '__main__':
    x, y = generate_train_data()

