from pathlib import Path
from typing import Union, Tuple, Optional

import polars as pl
import polars.selectors as cs
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset


class BaseGalaxyZooDataset(Dataset):
    def __init__(
        self,
        root: Union[str, Path] = "./dataset",
        transform: Optional[transforms.Transform] = None,
        n_rows: Optional[int] = None,
    ):
        if isinstance(root, str):
            root = Path(root)

        self.root = root.resolve()
        self.transform = transform

        self.parquet_file = (
            self.root / "gz_decals_volunteers_5_images.parquet"
        ).resolve()
        self.df = pl.read_parquet(self.parquet_file)

        if n_rows is not None:
            self.df = self.df.head(n_rows)

        self.X = self.df.select("image_path").to_series()
        self.Y = self.select_columns()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.root / self.X[idx]
        image = torchvision.io.read_image(
            str(image_path), mode=torchvision.io.ImageReadMode.RGB
        )
        if self.transform:
            image = self.transform(image)

        features = self.select_features(idx)

        return image, features

    def select_columns(self) -> pl.DataFrame:
        raise NotImplementedError("This method should be implemented in subclasses.")

    def select_features(self, idx) -> torch.Tensor:
        raise NotImplementedError("This method should be implemented in subclasses.")


class GalaxyZooDataset(Dataset):
    def select_columns(self) -> pl.DataFrame:
        return self.df.select(cs.ends_with("debiased"))

    def select_features(self, idx) -> pl.DataFrame:
        return self.Y[idx].to_torch(dtype=pl.Float32).squeeze()


class GalaxyZooClassDataset(BaseGalaxyZooDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.labels = ["merger", "elliptical", "spiral", "irregular"]

    def select_columns(self) -> pl.DataFrame:
        return self.df.select(pl.col("galaxy_class_int")).to_series()

    def select_features(self, idx) -> torch.Tensor:
        return torch.tensor(self.Y[idx], dtype=torch.int64)
