import shutil
import os
import pandas as pd 

import logging
from PIL import Image
from autocrop import Cropper


logging.basicConfig(level=logging.INFO)

TRAIN_LABEL_TXT = 'metas/intra_test/train_label.txt'
TEST_LABEL_TXT = 'metas/intra_test/test_label.txt'
cropper = Cropper()

def create_train():
    logging.info('Copying 100k cropped train dataset ...')
    train_label_df = pd.read_csv(TRAIN_LABEL_TXT, header=None, delimiter=' ')
    train_dir_list = train_label_df[0].tolist()
    train_label_list = train_label_df[1].tolist()
    i = 0

    for source in train_label_df[0].tolist():
        if train_label_df[1].tolist()[i] == 0:
            cropped_array = cropper.crop(source)
            # if cropped_array:
            if cropped_array is not None:
                cropped_image = Image.fromarray(cropped_array)
                cropped_image.save('dataset-crop/train/real/' + str(i) + '.png')
            else:
                pass
        else:
            cropped_array = cropper.crop(source)
            if cropped_array is not None:
                cropped_image = Image.fromarray(cropped_array)
                cropped_image.save('dataset-crop/train/spoof/' + str(i) + 'new.png')
            else:
                pass
            # shutil.copy(source, 'dataset/train/spoof')
        i += 1
    logging.info('Copying cropped train dataset success')

def create_test():
    logging.info('Copying 100k cropped test dataset ...')
    test_label_df = pd.read_csv(TEST_LABEL_TXT, header=None, delimiter=' ')
    test_dir_list = test_label_df[0].tolist()
    test_label_list = test_label_df[1].tolist()
    i = 0
    for source in test_label_df[0].tolist():
        if test_label_df[1].tolist()[i] == 0:
            cropped_array = cropper.crop(source)
            # if cropped_array:
            if cropped_array is not None:
                cropped_image = Image.fromarray(cropped_array)
                cropped_image.save('dataset-crop/test/real/' + str(i) + '.png')
            else:
                pass
        else:
            cropped_array = cropper.crop(source)
            # if cropped_array:
            if cropped_array is not None:
                cropped_image = Image.fromarray(cropped_array)
                cropped_image.save('dataset-crop/test/spoof/' + str(i) + '.png')
            else:
                pass
            # shutil.copy(source, 'dataset/test/spoof')
        i += 1
    logging.info('Copying cropped test dataset success')

if __name__ == '__main__':
   logging.info('Creating cropped dataset ...')
   # create_train()
   # create_test()