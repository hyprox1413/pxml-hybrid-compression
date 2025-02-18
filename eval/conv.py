import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import torch

from datetime import datetime
from matplotlib import pyplot as plt
from pathlib import Path
from PIL import Image

from data_loaders.my_image_datasets import JPGDataset
from models.conv import Autoencoder

BATCH_SIZE = 16

NUM_IMAGES = 100
NUM_FILTERS = 24
KERNEL_SIZE = 4
STRIDE = 4

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

    image = test_data[0].cpu().numpy().swapaxes(0, 2).swapaxes(0, 1) * 256
    image = image.clip(0, 255).astype(np.uint8)
    recons = model(test_data[0]).detach().cpu().numpy().swapaxes(0, 2).swapaxes(0, 1) * 256
    recons = recons.clip(0, 255).astype(np.uint8)
    print(recons)

    plt.imshow(image)
    plt.show()
    plt.imshow(recons)
    plt.show()

if __name__ == '__main__':
    main()
