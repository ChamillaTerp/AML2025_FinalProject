import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class GalaxyBranchDataset(Dataset):
    def __init__(self, df, label_cols, transform = None, image_col = 'png_loc'):
        """
        General-purpose dataset for Galaxy Zoo hierarchical training.

        Parameters:
        - df: DataFrame containing image paths and labels
        - label_cols: List of label column names for this branch
        - transform: torchvision transforms to apply to the image
        - image_col: Column name for image path (default: 'png_loc')
        """
        self.df = df.reset_index(drop=True)
        self.label_cols = label_cols
        self.image_col = image_col

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.AutoAugment(policy = transforms.AutoAugmentPolicy.IMAGENET),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # Assuming RGB
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row[self.image_col]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Get the soft labels (debiased probabilities)
        label_values = row[self.label_cols].astype(float).values
        labels = torch.tensor(label_values, dtype=torch.float32)

        return image, labels