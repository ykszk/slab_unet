import os
import json

def load_json(filename):
    with open(filename) as f:
        return json.load(f)

import copy
debug_mode = False
# from keras.optimizers import Adam
from adabound import AdaBound

from keras.models import Model
import os
import numpy as np
import tqdm
import glob
import gc
import inspect
import mhd
# from model_slab import build_model
# from model_res_slab import build_model
from model_slab import build_model
# from model_slab3d import build_model
from vat_slab_generator import VatSlabGenerator, VatSlabSeed
from slab_generator import SlabGenerator, SlabSeed
from vat_utils import VatLoss
# from sklearn.model_selection import StratifiedKFold, train_test_split
from kfold import StratifiedGroupKFold
import utils
import keras.callbacks
from keras import backend as K
from time import sleep
import gc
# setup output directories
from datetime import datetime
import shutil
base_outdir = os.path.join('result', datetime.today().strftime("%y%m%d_%H%M%S"))
image_outdir = os.path.join(base_outdir,'images')
os.makedirs(base_outdir, exist_ok=True)
os.makedirs(image_outdir, exist_ok=True)
shutil.copyfile(__file__, os.path.join(base_outdir,os.path.basename(__file__)))
shutil.copyfile(inspect.getfile(build_model), os.path.join(base_outdir,os.path.basename(inspect.getfile(build_model))))
slab_thickness = 16

def cv(k_index, train_index, test_index):
    outdir = os.path.join(base_outdir,'k{}'.format(k_index))
    os.makedirs(outdir, exist_ok=True)

    for ci, cid in enumerate(tqdm.tqdm(ids_test)):
        x, h = mhd.read_memmap(os.path.join(mhd_dir,cid+'.mha'))
        y,_ = mhd.read(os.path.join(label_dir,cid+'.mha'))
        valid_zs = [(slab_thickness//2 <= z < len(x)-slab_thickness//2) and np.any(lung[z]>0) for z in range(len(x))]
        zmin = np.min(np.where(valid_zs))
        zmax = np.max(np.where(valid_zs))
        seeds = [SlabSeed(x,x,i) for i in range(zmin,zmax+1)]
        p = model.predict_generator(SlabGenerator(seeds,slab_thickness,batch_size, False, transform=utils.transform, transpose=False),
                                    max_queue_size=workers+1,workers=workers,use_multiprocessing=workers>1)
        p = np.squeeze(p)
        label = np.argmax(p,axis=-1).astype(np.uint8)
        label = np.pad(label,((zmin,len(x)-zmax-1),(0,0),(0,0)),mode='constant',constant_values=0)
        h['CompressedData'] = True
        mhd.write(os.path.join(image_outdir,cid+'.mha'),label,h)

if __name__ == "__main__":
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True,random_state=20190618)
    groups = [v[0] for v in datalist_flattened]
    for k_index, (train_index, test_index) in enumerate(skf.split(datalist_flattened, np.array(label_list)-1, groups=groups)):
        cv(k_index, train_index, test_index)
