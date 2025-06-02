from pathlib import Path
from typing import Union

import polars as pl
import polars.selectors as cs
import torch.utils.data as data
import torchvision
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset


class GalaxyZooDecalsDataset(Dataset):
    def __init__(self, root: Union[str, Path], transform=None):
        if isinstance(root, str):
            root = Path(root)

        self.parquet_file = root / "gz_decals_volunteers_1_and_2_images.parquet"
        self.transform = transform
        self.df = pl.read_parquet(self.parquet_file)
        self.X = self.df.select("image_path").to_series()
        self.Y = self.df.select(cs.ends_with("debiased"))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.X[idx]
        image = torchvision.io.read_image(
            str(image_path), mode=torchvision.io.ImageReadMode.RGB
        )
        if self.transform:
            image = self.transform(image)

        features = self.Y[idx].to_numpy()

        return image, features
