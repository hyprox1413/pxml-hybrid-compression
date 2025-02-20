import numpy as np
import pandas as pd

import glob
import random
import torch
import urllib.request

from multiprocessing.pool import ThreadPool
from pathlib import Path
from PIL import Image

random.seed(0)

class JPGDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, device, num_images=-1):
        self.device = device
        self.images = []
        image_paths = list(data_path.iterdir())
        random.shuffle(image_paths);

        # use first num_images images
        if (num_images >= 0):
            image_paths = image_paths[:num_images];

        # load images from path
        try:
            for image_file in image_paths:
                image = Image.open(image_file)
                self.images.append(np.asarray(image).swapaxes(0, 2).swapaxes(1, 2).astype(np.float32) / 256) 
        except Exception as e:
            print(f"error: {e}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return torch.from_numpy(self.images[idx]).to(self.device)

def main():
    path = './unsplash/'
    documents = ['photos', 'keywords', 'collections', 'conversions', 'colors']
    datasets = {}

    for doc in documents:
        files = glob.glob(path + doc + ".tsv*")

        subsets = []
        for filename in files:
          df = pd.read_csv(filename, sep='\t', header=0)
          subsets.append(df)

        datasets[doc] = pd.concat(subsets, axis=0, ignore_index=True)

    # Unsplash dataset downloading code taken from Vladimir Haltakov.
    # https://github.com/haltakov/natural-language-image-search/blob/main/02-download-unsplash-dataset.ipynb

    photos = datasets['photos'];
    photo_urls = photos[['photo_id', 'photo_image_url']].values.tolist()

    # Path where the photos will be downloaded
    photos_donwload_path = Path("./unsplash/images/");

    # Function that downloads a single photo
    def download_photo(photo):
        # Get the ID of the photo
        photo_id = photo[0]

        # Get the URL of the photo (setting the width to 640 pixels)
        photo_url = photo[1]

        # Path where the photo will be stored
        photo_path = photos_donwload_path / (photo_id + ".jpg")

        # Only download a photo if it doesn't exist
        if not photo_path.exists():
            try:
                urllib.request.urlretrieve(photo_url, photo_path)
            except:
                # Catch the exception if the download fails for some reason
                print(f"Cannot download {photo_url}")
                pass

    # Create the thread pool
    threads_count = 128
    pool = ThreadPool(threads_count)

    # Start the download
    pool.map(download_photo, photo_urls)

    # Display some statistics
    print(f'Photos downloaded: {len(photos)}')
        

if __name__ == '__main__':
    main()
