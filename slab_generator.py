# -*- coding: utf-8 -*-
from tensorflow.keras import backend as K
import numpy as np
import math
from tensorflow.keras.utils import Sequence
from collections import namedtuple
import random
import mhd

SlabSeed = namedtuple('SlabSeed', ('x', 'y', 'index'))

class SlabGenerator(Sequence):
    '''Image Slab Generator
    '''

    def __init__(self, slab_seeds, slab_thickness, batch_size,
                 shuffle=False, transform=None, transpose=True):
        self.slab_seeds = slab_seeds
        self.slab_thickness = slab_thickness
        self.x_batch_shape = [slab_seeds[0].x.shape[1], slab_seeds[0].x.shape[2], slab_thickness]
        self.y_batch_shape = [slab_seeds[0].x.shape[1], slab_seeds[0].x.shape[2], 1]
        if isinstance(slab_seeds[0].y, str):
            self.y_dtype = mhd.read(slab_seeds[0].y)[0].dtype
        else:
            self.y_dtype = slab_seeds[0].y.dtype
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transform = transform
        self.transpose = transpose

    def __len__(self):
        return math.ceil(len(self.slab_seeds) / self.batch_size)

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.slab_seeds)

    def __getitem__(self, idx):
        cur_batch_size = min((idx+1)*self.batch_size, len(self.slab_seeds)) - idx*self.batch_size
        batch_x = np.zeros(tuple([cur_batch_size] + self.x_batch_shape), dtype=K.floatx())
        batch_y = np.zeros(tuple([cur_batch_size] + self.y_batch_shape), dtype=self.y_dtype)
        for batch_i in range(cur_batch_size):
            i = idx * self.batch_size + batch_i
            x = self.slab_seeds[i].x
            y = self.slab_seeds[i].y
            if isinstance(y, str):
                y = mhd.read(y)[0]
            index = self.slab_seeds[i].index
            xx = x[index-self.slab_thickness//2:index+math.ceil(self.slab_thickness/2)]
            if y.ndim==2:
                yy = y
            else:
                yy = y[index]
            if self.transform is not None:
                xx, yy = self.transform(xx, yy)

            xx = np.transpose(xx, (1,2,0))
            batch_x[batch_i] = xx
            batch_y[batch_i] = np.expand_dims(yy, -1)

        if self.transpose:
            return batch_x, batch_y
        else:
            return np.transpose(batch_x,(0,3,1,2)), batch_y
