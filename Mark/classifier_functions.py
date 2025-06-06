import numpy as np
import os
from tensorflow.keras.utils import Sequence
from PIL import Image

class GalaxyDataGenerator(Sequence):
    def __init__(self, df, image_dir, batch_size=16, input_shape=(424, 424), shuffle=True):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        idxs = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_df = self.df.iloc[idxs]

        X = np.zeros((len(batch_df), *self.input_shape, 3), dtype=np.float32)
        y = np.zeros((len(batch_df), 5), dtype=np.float32)  # 5 vote fractions

        for i, row in enumerate(batch_df.itertuples()):
            name = row.iauname
            subdir = name[:4]
            img_path = os.path.join(self.image_dir, subdir, f"{name}.png")
            try:
                img = Image.open(img_path).convert('RGB').resize(self.input_shape)
                X[i] = np.array(img) / 255.0
                y[i] = np.array([row.elliptical, row.lenticular, row.spiral, row.irregular, row.artifact])
            except:
                continue  # skip missing or corrupt images

        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def encode_label(self, label):
        classes = ['elliptical', 'lenticular', 'spiral', 'irregular', 'artifact']
        one_hot = np.zeros(len(classes))
        if label in classes:
            one_hot[classes.index(label)] = 1.0
        return one_hot