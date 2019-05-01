from tqdm import tqdm
from skimage.transform import resize
import numpy as np
from skimage.io import imread
import os

class NucleiTools:
    def load_images(self, data_type):
        print('Getting images')

        for n, id_ in tqdm(enumerate(self.ids[data_type]), total=len(self.ids[data_type])):
            path = self.dataset_path + id_
            if self.improved:
                img_data = imread(path + '/images/' + id_ + '_improved.png')
                filename = id_ + '_improved.png'
            else:
                img_data = imread(path + '/images/' + id_ + '.png')
                filename = id_ + '.png'
            if self.img_size:
                img_data = resize(img_data, (self.img_size[0], self.img_size[1]), mode='constant', preserve_range=True)
            width, height, _ = img_data.shape
            img = {
                #"data": img_data,
                "filename": filename,
                "width": width,
                "height": height,
                "masks": []
            }
            self.imgs[id_] = img

    def load_masks(self, data_type):
        print('Getting masks')
        for n, id_ in tqdm(enumerate(self.ids[data_type]), total=len(self.ids[data_type])):
            path = self.dataset_path + id_
            masks = []
            for mask_file in next(os.walk(path + '/masks/'))[2]:
                mask_ = imread(path + '/masks/' + mask_file)
                width, height = mask_.shape
                if self.img_size:
                    mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',
                                                preserve_range=True), axis=-1)
                masks.append(mask_)
            self.imgs[id_]['masks'] = masks

    # size argument(HEIGHT, WIDTH)
    def __init__(self, dataset_path, size=False, improved=False, val_size=0.2):
        self.dataset_path = dataset_path
        self.img_size = size
        self.improved = improved
        self.imgs = {}
        self.val_size = val_size

        # Get train IDs
        for id_tuples in os.walk(self.dataset_path):
            ids = id_tuples[1]
            break
        #Separate training and validation
        id_dict = {}
        # first (1 - val_size)% are training ids
        id_dict['train'] = ids[int(len(ids) * (1 - self.val_size)):]

        # last val_size % are validation ids
        id_dict['val'] = ids[:-int(len(ids) * self.val_size)]
        self.ids = id_dict
