import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class GalaxyDataset(Dataset):
    def __init__(self, df, label_cols, transform = None):
        self.df = df                                                                # DataFrame containing the image paths and labels
        self.label_cols = label_cols                                                # List of label columns to be used
        self.transform = transform or transforms.Compose([                          # Default transformations if none are provided
            transforms.Resize((224, 224)),                                                  # Resize images to a fixed size
            transforms.ToTensor(),                                                          # Convert PIL images to PyTorch tensors
            transforms.Normalize([0.5], [0.5])                                              # Normalize with ImageNet stats assuming grayscale images or RGB images
        ])

    def __len__(self):
        return len(self.df)                                                          # Returns the number of samples in the dataset
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]                                                      # Get the row at the specified index
        image = Image.open(row['full_image_path']).convert('RGB')                    # Open the image using PIL and convert it to RGB format
        if self.transform:
            image = self.transform(image)                                            # Apply transformations to the image

        label = torch.tensor(row[self.label_cols].astype(float).values, dtype = torch.float32)     # Convert the labels to a PyTorch tensor

        return image, label                                                          # Return the image and its corresponding labels as a tuple