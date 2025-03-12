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

BATCH_SIZE = 16

NUM_IMAGES = 1000
PURE_NUM_FILTERS = 40
HYBRID_NUM_FILTERS = 10
PURE_SCALE = 20
HYBRID_SCALE = 5
IMAGE_RANK = 16
LATENT_RANK = 25

PURE_MODEL_DIR = f"saved_models/x{PURE_SCALE}"
HYBRID_MODEL_DIR = f"saved_models/x{HYBRID_SCALE}"

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

def frob_distance_sqr(a: np.ndarray, b: np.ndarray):
    assert a.shape == b.shape, "arrays should be same shape"
    rows, columns = a.shape[-2:]
    a = a.flatten()
    b = b.flatten()
    return np.sum(np.square(a - b)) ** (1 / 2)

def main():
    torch.manual_seed(0)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('cuda is available')
    else:
        device = torch.device('cpu')
        print('cuda not available, using cpu')

    dataset = JPGDataset(Path('./unsplash/processed'), device, num_images=1000)

    train_data, val_data, test_data = torch.utils.data.random_split(dataset, [0.7, 0.2, 0.1])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

    pure_model = Autoencoder(num_filters=PURE_NUM_FILTERS, kernel_size=PURE_SCALE, stride=PURE_SCALE).to(device)
    hybrid_model = Autoencoder(num_filters=HYBRID_NUM_FILTERS, kernel_size=HYBRID_SCALE, stride=HYBRID_SCALE).to(device)

    try:
        pure_model.load_state_dict(torch.load(get_most_recent_file(PURE_MODEL_DIR), weights_only=True))
        hybrid_model.load_state_dict(torch.load(get_most_recent_file(HYBRID_MODEL_DIR), weights_only=True))
    except:
        print("no saved models.")
        quit()
    
    hybrid_encoder, hybrid_decoder = hybrid_model.encoder, hybrid_model.decoder

    images_list = []

    for i in range(len(test_data)):
        image = channels_last(test_data[i].cpu().numpy()) * 256
        image = image.clip(0, 255).astype(np.uint8)

        recons_conv = channels_last(pure_model(test_data[i]).detach().cpu().numpy()) * 256
        recons_conv = recons_conv.clip(0, 255).astype(np.uint8)

        reduced = svd_reduce(test_data[i].cpu().numpy(), IMAGE_RANK)
        recons_svd = channels_last(reduced) * 256
        recons_svd = recons_svd.clip(0, 255).astype(np.uint8)
    
        hybrid = channels_last(hybrid_decoder(torch.from_numpy(svd_reduce(hybrid_encoder(test_data[i]).detach().cpu().numpy(), LATENT_RANK)).to(device)).detach().cpu().numpy()) * 256
        hybrid = hybrid.clip(0, 255).astype(np.uint8)

        images_list.append((image, recons_conv, recons_svd, hybrid,
                            frob_distance_sqr(image, image) ** (1 / 2), frob_distance_sqr(image, recons_conv) ** (1 / 2),
                            frob_distance_sqr(image, recons_svd) ** (1 / 2), frob_distance_sqr(image, hybrid) ** (1 / 2)))

    dump_file = open('test_images.pkl', 'wb')
    pickle.dump(images_list, dump_file)

if __name__ == '__main__':
    main()
