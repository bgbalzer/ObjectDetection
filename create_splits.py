import argparse
import glob
import os
import random

import numpy as np

from utils import get_module_logger


def split(source, destination):
    """
    Create three splits from the processed records. The files should be moved to new folders in the
    same directory. This folder should be named train, val and test.

    args:
        - source [str]: source data directory, contains the processed tf records
        - destination [str]: destination data directory, contains 3 sub folders: train / val / test
    """

    train_portion = 0.75
    val_portion = 0.15
    test_portion = 0.10

    # create destination directories if needed
    train_destination = os.path.join(destination, 'train')
    val_destination = os.path.join(destination, 'val')
    test_destination = os.path.join(destination, 'test')

    if not os.path.isdir(train_destination):
        os.makedirs(train_destination)
    if not os.path.isdir(val_destination):
        os.makedirs(val_destination)
    if not os.path.isdir(test_destination):
        os.makedirs(test_destination)

    # get the source files and shuffle them
    file_list = glob.glob(os.path.join(source, '*.tfrecord'))
    np.random.shuffle(file_list)

    train_index = int(len(file_list) * train_portion)
    val_index = int(len(file_list) * (train_portion + val_portion))

    train_files, val_files, test_files = np.split(np.array(file_list), [train_index, val_index])

    # rename the files
    for file in train_files:
        filename = os.path.basename(file)
        os.rename(file, os.path.join(train_destination, filename))
    for file in val_files:
        filename = os.path.basename(file)
        os.rename(file, os.path.join(val_destination, filename))
    for file in test_files:
        filename = os.path.basename(file)
        os.rename(file, os.path.join(test_destination, filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--source', required=True,
                        help='source data directory')
    parser.add_argument('--destination', required=True,
                        help='destination data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.source, args.destination)