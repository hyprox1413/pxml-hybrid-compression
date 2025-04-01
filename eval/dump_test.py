import argparse
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
from utils.utils import *

parser = argparse.ArgumentParser(
        prog='TrainConv',
        description='Trains a convolutional model for images')

parser.add_argument('pure_rank')
parser.add_argument('pure_scale')
parser.add_argument('pure_filters')
parser.add_argument('hybrid_scale')
parser.add_argument('hybrid_filters')
parser.add_argument('hybrid_rank')

args = parser.parse_args()

BATCH_SIZE = 16

NUM_IMAGES = 1000
PURE_NUM_FILTERS = int(args.pure_filters)
HYBRID_NUM_FILTERS = int(args.hybrid_filters)
PURE_SCALE = int(args.pure_scale)
HYBRID_SCALE = int(args.hybrid_scale)
IMAGE_RANK = int(args.pure_rank)
LATENT_RANK = int(args.hybrid_rank)

PURE_MODEL_PATH = Path(f"saved_models/s{PURE_SCALE}-f{PURE_NUM_FILTERS}.pth")
HYBRID_MODEL_PATH = Path(f"saved_models/s{HYBRID_SCALE}-f{HYBRID_NUM_FILTERS}.pth")
DUMP_PATH = Path(f"test_dumps/pr{IMAGE_RANK}-ps{PURE_SCALE}-pf{PURE_NUM_FILTERS}-hs{HYBRID_SCALE}-hf{HYBRID_NUM_FILTERS}-hr{LATENT_RANK}.pkl")

IMAGE_HEIGHT = 1000
IMAGE_WIDTH = 1000

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
        pure_model.load_state_dict(torch.load(PURE_MODEL_PATH, weights_only=True))
        hybrid_model.load_state_dict(torch.load(HYBRID_MODEL_PATH, weights_only=True))
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

    dump_file = open(DUMP_PATH, 'wb')
    pickle.dump(images_list, dump_file)

if __name__ == '__main__':
    main()
