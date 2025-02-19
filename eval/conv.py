import math
import os
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
NUM_FILTERS = 25
KERNEL_SIZE = 10
STRIDE = 10

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

def frob_distance(a: np.ndarray, b: np.ndarray):
    assert a.shape == b.shape, "arrays should be same shape"
    rows, columns = a.shape[-2:]
    a = a.reshape(-1, rows, columns)
    b = b.reshape(-1, rows, columns)
    return np.sum(np.stack([la.norm(a[i] - b[i], ord='fro') for i in range(a.shape[0])]))

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

    model = Autoencoder(num_filters=NUM_FILTERS, kernel_size=KERNEL_SIZE, stride=STRIDE).to(device)

    try:
        model.load_state_dict(torch.load(get_most_recent_file("saved_models"), weights_only=True))
    except:
        print("no saved models.")
        quit()

    loss_fn = torch.nn.MSELoss()

    # testing loop
    total_loss = 0
    model.eval()
    for data in test_loader:
        with torch.no_grad():
            loss = loss_fn(model(data), data)
            total_loss += loss.item()

    print(f'average testing loss: {total_loss / len(test_data)}')

    """
    image = channels_last(test_data[0].cpu().numpy()) * 256
    image = image.clip(0, 255).astype(np.uint8)

    recons_conv = channels_last(model(test_data[0]).detach().cpu().numpy()) * 256
    recons_conv = recons_conv.clip(0, 255).astype(np.uint8)

    reduced = svd_reduce(test_data[0].cpu().numpy(), 42)
    recons_svd = channels_last(reduced * 256)
    recons_svd = recons_svd.clip(0, 255).astype(np.uint8)

    plt.imshow(image)
    plt.show()
    plt.imshow(recons_conv)
    plt.show()
    plt.imshow(recons_svd)
    plt.show()
    """

    frob_distance_conv = 0
    frob_distance_svd = 0

    for i in range(len(test_data)):
        image = channels_last(test_data[i].cpu().numpy())
        # image = image.clip(0, 255)

        recons_conv = channels_last(model(test_data[i]).detach().cpu().numpy())
        # recons_conv = recons_conv.clip(0, 255)

        reduced = svd_reduce(test_data[i].cpu().numpy(), 42)
        recons_svd = channels_last(reduced)
        # recons_svd = recons_svd.clip(0, 255)
    
        frob_distance_conv += frob_distance(image, recons_conv)
        frob_distance_svd += frob_distance(image, recons_svd)

    print(frob_distance_conv / len(test_data))
    print(frob_distance_svd / len(test_data))

if __name__ == '__main__':
    main()
