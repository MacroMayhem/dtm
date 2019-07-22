__author__ = "Aditya Singh"
__version__ = "0.1"

from utils.utils import load_config
import argparse
from src.model import Model


def train(config):
    dtm = Model(config=config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training the PPE model')
    parser.add_argument('--config', required=True, help='Path to config.yaml')
    args = parser.parse_args()
    config = load_config(args.config)
    train(config)