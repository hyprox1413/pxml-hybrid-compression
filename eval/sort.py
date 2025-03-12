import math
import os
import pickle
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import numpy.linalg as la

import torch

from datetime import datetime
from matplotlib import pyplot as plt
from pathlib import Path
from PIL import Image

from data_loaders.my_image_datasets import JPGDataset
from models.conv import Autoencoder

"""
BATCH_SIZE = 16
NUM_IMAGES = 1000
PURE_NUM_FILTERS = 40
HYBRID_NUM_FILTERS = 10
PURE_SCALE = 100
HYBRID_SCALE = 25
IMAGE_RANK = 1
LATENT_RANK = 5
PURE_MODEL_DIR = f"saved_models/x{PURE_SCALE}"
HYBRID_MODEL_DIR = f"saved_models/x{HYBRID_SCALE}"
"""

IMAGE_HEIGHT = 1000
IMAGE_WIDTH = 1000

def get_most_recent_file(str):
    path = Path(str)
    try:
        files = [f for f in path.iterdir() if f.is_file()]
        if not files:
            return None
        most_recent_file = max(files, key=os.path.getmtime)
        return most_recent_file
    except FileNotFoundError:
        return None

def channels_first(a: np.ndarray):
    return a.swapaxes(a.ndim - 3, a.ndim - 1).swapaxes(a.ndim - 2, a.ndim - 1)

def channels_last(a: np.ndarray):
    return a.swapaxes(a.ndim - 3, a.ndim - 1).swapaxes(a.ndim - 3, a.ndim - 2)

def svd_reduce(a: np.ndarray, rank):
    svd_tuple = la.svd(a)
    u = svd_tuple[0][..., :, :rank]
    s = svd_tuple[1][..., :rank]
    s_shape = s.shape[:-1]
    s = s.reshape(-1, rank)
    s = np.stack([np.diag(vector) for vector in s])
    s = s.reshape(*s_shape, rank, rank)
    vt = svd_tuple[2][..., :rank, :]
    return u @ s @ vt

def main():
    dump_file = open('test_images.pkl', 'rb')
    images_list = pickle.load(dump_file)

    images_list = sorted(images_list, key=lambda x: x[7])

    for i in range(len(images_list)):
        i_str = str(i).zfill(5)

        try:
            os.mkdir(f"test_images/{i_str}")
        except:
            pass

        image_tuple = images_list[i]

        image = Image.fromarray(image_tuple[0])
        image.save(f"test_images/{i_str}/original.png")

        recons_conv = Image.fromarray(image_tuple[1])
        recons_conv.save(f"test_images/{i_str}/conv.png")

        recons_svd = Image.fromarray(image_tuple[2])
        recons_svd.save(f"test_images/{i_str}/svd.png")

        hybrid = Image.fromarray(image_tuple[3])
        hybrid.save(f"test_images/{i_str}/hybrid.png")

if __name__ == '__main__':
    main()
