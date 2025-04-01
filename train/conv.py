import argparse
import sys
import os

from torch.optim import optimizer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import torch

from datetime import datetime
from matplotlib import pyplot as plt
from pathlib import Path

from data_loaders.my_image_datasets import JPGDataset
from models.conv import Autoencoder

parser = argparse.ArgumentParser(
        prog='TrainConv',
        description='Trains a convolutional model for images')

parser.add_argument('scale')
parser.add_argument('filters')

args = parser.parse_args()

BATCH_SIZE = 16
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3

NUM_IMAGES = 1000
SCALE = int(args.scale)
NUM_FILTERS = int(args.filters)

MODEL_PATH = Path(f"saved_models/s{SCALE}-f{NUM_FILTERS}.pth")

def main():
    print(f"checking {MODEL_PATH}")
    if MODEL_PATH.exists():
        print("preexisting model found, exiting")
        quit()

    print("no preexisting model found, training")
    torch.manual_seed(0)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('cuda is available')
    else:
        device = torch.device('cpu')
        print('cuda not available, using cpu')

    dataset = JPGDataset(Path('./unsplash/processed'), device, num_images=1000)

    """
    image = data[0].cpu().numpy()
    print(image.shape)
    plt.imshow(image)
    plt.show()
    """

    train_data, val_data, test_data = torch.utils.data.random_split(dataset, [0.7, 0.2, 0.1])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    model = Autoencoder(num_filters=NUM_FILTERS, kernel_size=SCALE, stride=SCALE).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.MSELoss()

    best_val_loss = float('inf')
    for _ in range(NUM_EPOCHS):

        # training loop
        total_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(data), data)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f'average training loss: {total_loss / len(train_data)}')

        # validation loop
        total_loss = 0
        model.eval()
        for data in val_loader:
            with torch.no_grad():
                loss = loss_fn(model(data), data)
                total_loss += loss.item()

        if (total_loss < best_val_loss):
            best_val_loss = total_loss
            torch.save(model.state_dict(), MODEL_PATH)

        print(f'average validation loss: {total_loss / len(val_data)}')


if __name__ == '__main__':
    main()
