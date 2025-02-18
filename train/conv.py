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

BATCH_SIZE = 16
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3

NUM_IMAGES = 1000
NUM_FILTERS = 24
KERNEL_SIZE = 4
STRIDE = 4

def main():
    torch.manual_seed(0)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('cuda is available')
    else:
        device = torch.device('cpu')
        print('cuda not available, using cpu')

    time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

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

    model = Autoencoder(num_filters=NUM_FILTERS, kernel_size=KERNEL_SIZE, stride=STRIDE).to(device)

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
            model_save_path = Path("./saved_models") / (time_str + ".pth")
            torch.save(model.state_dict(), model_save_path)

        print(f'average validation loss: {total_loss / len(val_data)}')


if __name__ == '__main__':
    main()
