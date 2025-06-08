import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class GalaxyDataset(Dataset):
    def __init__(self, df, label_cols, transform = None):
        self.df = df                                            # DataFrame containing the image paths and labels
        self.label_cols = label_cols                            # List of label columns to be used

        # Ensuring the default transform is valid 
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
        image = Image.open(row['full_image_path']).convert('RGB')

        if self.transform:
            image = self.transform(image)  

        label = torch.tensor(row[self.label_cols].astype(float).values, dtype=torch.float32)

        return image, label