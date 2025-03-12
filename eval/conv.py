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

SAMPLE_IMAGE_IDX = int(sys.argv[1])

NUM_IMAGES = 1000
PURE_NUM_FILTERS = 12
HYBRID_NUM_FILTERS = 6
PURE_SCALE = 10
HYBRID_SCALE = 5
IMAGE_RANK = 20
LATENT_RANK = 50

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
    a = a.reshape(-1, rows, columns)
    b = b.reshape(-1, rows, columns)
    return np.sum(np.stack([la.norm(a[i] - b[i], ord='fro') ** 2 for i in range(a.shape[0])]))

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

    """
    loss_fn = torch.nn.MSELoss()

    # testing loop
    total_loss = 0
    pure_model.eval()
    for data in test_loader:
        with torch.no_grad():
            loss = loss_fn(pure_model(data), data)
            total_loss += loss.item()

    print(f'average pure model testing loss: {total_loss / len(test_data)}')

    # testing loop
    total_loss = 0
    hybrid_model.eval()
    for data in test_loader:
        with torch.no_grad():
            loss = loss_fn(hybrid_model(data), data)
            total_loss += loss.item()

    print(f'average hybrid model testing loss: {total_loss / len(test_data)}')
    """

    image = channels_last(test_data[SAMPLE_IMAGE_IDX].cpu().numpy()) * 256
    image = image.clip(0, 255).astype(np.uint8)

    recons_conv = channels_last(pure_model(test_data[SAMPLE_IMAGE_IDX]).detach().cpu().numpy()) * 256
    recons_conv = recons_conv.clip(0, 255).astype(np.uint8)

    reduced = svd_reduce(test_data[SAMPLE_IMAGE_IDX].cpu().numpy(), IMAGE_RANK)
    recons_svd = channels_last(reduced * 256)
    recons_svd = recons_svd.clip(0, 255).astype(np.uint8)

    hybrid = channels_last(hybrid_decoder(torch.from_numpy(svd_reduce(hybrid_encoder(test_data[SAMPLE_IMAGE_IDX]).detach().cpu().numpy(), LATENT_RANK)).to(device)).detach().cpu().numpy()) * 256
    hybrid = hybrid.clip(0, 255).astype(np.uint8)

    try:
        os.mkdir(f"test_images/{SAMPLE_IMAGE_IDX}")
    except:
        pass
    
    image = Image.fromarray(image)
    image.save(f"test_images/{SAMPLE_IMAGE_IDX}/original.png")

    recons_conv = Image.fromarray(recons_conv)
    recons_conv.save(f"test_images/{SAMPLE_IMAGE_IDX}/conv.png")

    recons_svd = Image.fromarray(recons_svd)
    recons_svd.save(f"test_images/{SAMPLE_IMAGE_IDX}/svd.png")

    hybrid = Image.fromarray(hybrid)
    hybrid.save(f"test_images/{SAMPLE_IMAGE_IDX}/hybrid.png")

    quit()

    plt.imshow(image)
    plt.show()
    plt.imshow(recons_conv)
    plt.show()
    plt.imshow(recons_svd)
    plt.show()
    plt.imshow(hybrid)
    plt.show()

    frob_distance_conv = 0
    frob_distance_svd = 0
    frob_distance_hybrid = 0

    for i in range(len(test_data)):
        image = channels_last(test_data[i].cpu().numpy())
        image = image.clip(0, 255)

        recons_conv = channels_last(pure_model(test_data[i]).detach().cpu().numpy())
        recons_conv = recons_conv.clip(0, 255)

        reduced = svd_reduce(test_data[i].cpu().numpy(), IMAGE_RANK)
        recons_svd = channels_last(reduced)
        recons_svd = recons_svd.clip(0, 255)
    
        hybrid = channels_last(hybrid_decoder(torch.from_numpy(svd_reduce(hybrid_encoder(test_data[0]).detach().cpu().numpy(), LATENT_RANK)).to(device)).detach().cpu().numpy()) * 256
        hybrid = hybrid.clip(0, 255).astype(np.uint8)

        frob_distance_conv += frob_distance(image, recons_conv)
        frob_distance_svd += frob_distance(image, recons_svd)
        frob_distance_hybrid += frob_distance(image, hybrid)

    print(frob_distance_conv / len(test_data))
    print(frob_distance_svd / len(test_data))
    print(frob_distance_hybrid / len(test_data))

if __name__ == '__main__':
    main()
