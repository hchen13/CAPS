import os

ROOT_DIR = os.path.dirname(__file__)

CACHE_ROOT = os.path.join(ROOT_DIR, 'cache')

DATA_DIR = "/Users/ethan/datasets/cryptocurrency/huobi.pro"

def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


