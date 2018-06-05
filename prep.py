from datetime import datetime
from logging import warning

import numpy as np
import pandas as pd

from settings import *


def ma(array, n=5):
    ma = []
    for i, val in enumerate(array):
        batch = array[max(i - n + 1, 0): i + 1]
        ma.append(np.mean(batch))
    return np.array(ma)


def ema(array, n=5):
    beta = 1 - 1 / n
    ema = []
    for i, val in enumerate(array):
        if i == 0:
            ema.append(val)
            continue
        previous = ema[i - 1]
        current = beta * previous + (1 - beta) * val
        ema.append(current)
    return np.array(ema)


def macd(array, a=12, b=26, c=9):
    fast = ema(array, a)
    slow = ema(array, b)
    macd = fast - slow
    signal = ema(macd, c)
    diff = macd - signal
    return macd, signal, diff


class File(object):
    def __init__(self, *args):
        self.full_path = os.path.abspath(os.path.join(*args))
        self.location = os.path.dirname(self.full_path)
        self.file_name = os.path.basename(self.full_path)

    def __new__(cls, *args):
        full_path = os.path.abspath(os.path.join(*args))
        if cls.is_valid(full_path):
            return super(File, cls).__new__(cls)
        return None

    def __repr__(self):
        return self.full_path

    @classmethod
    def is_valid(cls, full_path):
        if not os.path.exists(full_path):
            print("Not exists:", full_path)
            return False
        if os.path.isfile(full_path):
            return True
        print("Not a file:", full_path)
        return False


class DataFile(File):

    @classmethod
    def is_valid(cls, full_path):
        if not super(DataFile, cls).is_valid(full_path):
            return False
        ext_list = ['.csv', '.xls', '.xlsx']
        _, ext = os.path.splitext(full_path)
        if not ext.lower() in ext_list:
            warning("Unaccepted data file format: {}".format(ext))
            return False

        if cls.identify_coin_pair(full_path) == False:
            return False
        return True

    @classmethod
    def identify_coin_pair(cls, path):
        name = os.path.basename(path)
        basename, ext = os.path.splitext(name)
        try:
            base, counter = basename.split('_')
        except ValueError:
            warning("Identify coin pair failed: {}".format(path))
            return False
        if base.isalpha() and counter.isalpha():
            return base, counter
        warning("Invalid coin pair: {}".format(path))
        return False

    def time_interval(self):
        df = pd.read_excel(self.full_path).sort_values('日期')
        time0 = df.iloc[0]['日期']
        time1 = df.iloc[-1]['日期']
        return time0, time1
    
    def __init__(self, *args):
        super(DataFile, self).__init__(*args)
        self.base, self.counter = DataFile.identify_coin_pair(self.file_name)

    def synthesize_indicators(self, df):
        """Synthesize various indicators along with the original OHLCV data"""
        prices = df['收盘价'].as_matrix()
        ma1 = ma(prices, 6)
        ma2 = ma(prices, 12)
        ma3 = ma(prices, 24)
        proper, signal, histogram = macd(prices)
        df = df.assign(
            ma1=ma1, ma2=ma2, ma3=ma3,
            macd=proper, macd_signal=signal, macd_diff=histogram
        )
        return df

    def flat(self):
        df = pd.read_excel(self.full_path).sort_values('日期')
        base = df[['收盘价', '成交量']]
        data = self.synthesize_indicators(base)
        return data.as_matrix()

    def stacked(self, past_length=72, future_length=1):

        def normalize(inputs, outputs):
            x, y = np.array(inputs), np.array(outputs)
            upper = x[:, 0].max()
            lower = x[:, 0].min()
            x[:, 0] = (x[:, 0] - lower) / (upper - lower + 1e-8)
            y[:, 0] = (y[:, 0] - lower) / (upper - lower + 1e-8)
            x[:, 2] = (x[:, 2] - lower) / (upper - lower + 1e-8)
            y[:, 2] = (y[:, 2] - lower) / (upper - lower + 1e-8)
            x[:, 3] = (x[:, 3] - lower) / (upper - lower + 1e-8)
            y[:, 3] = (y[:, 3] - lower) / (upper - lower + 1e-8)
            x[:, 4] = (x[:, 4] - lower) / (upper - lower + 1e-8)
            y[:, 4] = (y[:, 4] - lower) / (upper - lower + 1e-8)

            upper = x[:, 1].max()
            lower = x[:, 1].min()
            x[:, 1] = (x[:, 1] - lower) / (upper - lower + 1e-8)
            y[:, 1] = (y[:, 1] - lower) / (upper - lower + 1e-8)

            # macd = x[:, 4 : 6]
            # upper = macd.max(axis=0, keepdims=True)
            # lower = macd.min(axis=0, keepdims=True)
            # macd_norm = (macd - lower) / (upper - lower + 1e-8)
            # x[:, 4 : 6] = macd_norm
            # y[:, 4 : 6] = (y[:, 4 : 6] - lower) / (upper - lower + 1e-8)
            return x, y

        flat = self.flat()
        # normalize MACD
        macd = flat[:, 5 : 8]
        mean = macd.mean(axis=0, keepdims=True)
        std = macd.std(axis=0, keepdims=True)
        macd = (macd - mean) / (std + 1e-8)
        flat[:, 5 : 8] = macd

        data_size = len(flat)
        x, y = [], []
        window_size = past_length + future_length
        for i in range(data_size - window_size + 1):
            inputs = flat[i : i + past_length]
            outputs = flat[i + past_length : i + window_size]

            inputs, outputs = normalize(inputs, outputs)

            x.append(inputs)
            y.append(outputs)
        return np.array(x), np.array(y)

    def display(self, start=datetime(2018, 1, 1, 0, 0), end=datetime.now()):
        from matplotlib import pyplot as plt
        df = pd.read_excel(self.full_path).sort_values('日期')
        df['日期'] = pd.to_datetime(df['日期'])
        base = df[['日期', '收盘价', '成交量']]
        data = self.synthesize_indicators(base)
        mask = (data['日期'] >= start) & (data['日期'] <= end)
        data = data.loc[mask]

        figure, (top, down) = plt.subplots(2, 1, figsize=[10, 8], tight_layout=True)
        time = range(len(data))
        top.plot(time, data['收盘价'])
        top.plot(time, data['ma1'], 'y', linewidth=.5)
        top.plot(time, data['ma2'], 'r', linewidth=.5)
        top.plot(time, data['ma3'], 'r', linewidth=.5)
        down.plot(time, data['macd'], 'y')
        down.plot(time, data['macd_signal'], 'r')
        down.bar(time, data['macd_diff'])
        plt.show()


class Dataset(object):

    base_files = {}
    counter_files = {}

    def __init__(self, data_root=DATA_DIR):
        self.root = data_root
        self._get_data_files()

    def _get_data_files(self):
        print("Start scanning for data files...")
        all_files = os.listdir(self.root)
        count = 0
        for i, file_name in enumerate(all_files):
            datafile = DataFile(self.root, file_name)
            if datafile is None:
                continue
            if datafile.base not in self.base_files:
                self.base_files[datafile.base] = [datafile]
            else:
                self.base_files[datafile.base].append(datafile)
            if datafile.counter not in self.counter_files:
                self.counter_files[datafile.counter] = [datafile]
            else:
                self.counter_files[datafile.counter].append(datafile)
            count += 1
        print("Scan complete! {} data files detected.\n".format(count))

    def valid_pair(self, base, counter):
        if base not in self.base_files:
            return False
        for data_file in self.base_files[base]:
            if data_file.counter == counter:
                return True
        return False

    def _read_data(self, base, counter):
        pass

    def fetch(self, base=None, counter=None):
        """fetch data of the given base and counter, show all coin pairs based on base if counter is not given;
        similarly show all pairs countering with counter if base is not given. If neither base nor counter is
        given, show all available data

        """
        base = base.lower() if base else None
        counter = counter.lower() if counter else None

        if base is None and counter is None:
            fetch_list = sum([ self.base_files[key] for key in self.base_files.keys() ], [])
        elif base is None:
            try:
                fetch_list = self.counter_files[counter]
            except KeyError:
                print("There is no such counter symbol:", counter)
                return []
        elif counter is None:
            try:
                fetch_list = self.base_files[base]
            except KeyError:
                print("There is no such base symbol:", base)
                return []

        else:
            fetch_list = list(filter(lambda df: df.counter == counter, self.base_files[base]))

        return fetch_list


def cache(ndarray, filename):
    print("Caching array {} into file: {}".format(ndarray.shape, filename))
    path = os.path.join(CACHE_ROOT, filename)
    ensure_dir_exists(CACHE_ROOT)
    np.save(path, ndarray)
