import shutil
import os
import pandas as pd 

import logging

logging.basicConfig(level=logging.INFO)

TRAIN_LABEL_TXT = 'metas/intra_test/train_label_100k.txt'
TEST_LABEL_TXT = 'metas/intra_test/test_label_100k.txt'


def create_train():
    logging.info('Copying 100k train dataset ...')
    train_label_df = pd.read_csv(TRAIN_LABEL_TXT, header=None, delimiter=' ')
    train_dir_list = train_label_df[0].tolist()
    train_label_list = train_label_df[1].tolist()
    i = 0
    for source in train_label_df[0].tolist():
        if train_label_df[1].tolist()[i] == 0:
            shutil.copy(source, 'dataset/train/real')
        else:
            shutil.copy(source, 'dataset/train/spoof')
        i += 1
    logging.info('Copying train dataset success')

def create_test():
    logging.info('Copying 100k test dataset ...')
    test_label_df = pd.read_csv(TEST_LABEL_TXT, header=None, delimiter=' ')
    test_dir_list = test_label_df[0].tolist()
    test_label_list = test_label_df[1].tolist()
    i = 0
    for source in test_label_df[0].tolist():
        if test_label_df[1].tolist()[i] == 0:
            shutil.copy(source, 'dataset/test/real')
        else:
            shutil.copy(source, 'dataset/test/spoof')
        i += 1
    logging.info('Copying test dataset success')

if __name__ == '__main__':
   logging.info('Creating dataset ...')
   create_train()
   create_test()