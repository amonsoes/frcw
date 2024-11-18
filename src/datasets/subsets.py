import os
import pandas as pd

from torch.utils.data import Dataset
from PIL import Image

class Nips17Subset(Dataset):
    
    def __init__(self, img_path, label_path, transform, target_transform, adversarial, is_test_data=False):
        super().__init__()
        self.labels = pd.read_csv(label_path)
        self.img_dir = img_path
        if adversarial:
            self.getitem_func = self.getitem_adversarial
        elif is_test_data:
            self.getitem_func = self.getitem_withpath
        else:
            self.getitem_func = self.getitem
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        tup = self.getitem_func(idx)
        return tup

    def getitem_adversarial(self, idx):
        img_path = os.path.join(self.img_dir, self.labels.iloc[idx, 0])
        image = Image.open(img_path + '.png')
        label = self.labels.iloc[idx, 6] - 1
        image = self.transform(image)
        label = self.target_transform(label)
        return image, label, img_path

    def getitem_withpath(self, idx):
        img_path = os.path.join(self.img_dir, self.labels.iloc[idx, 0])
        image = Image.open(img_path + '.png')
        label = self.labels.iloc[idx, 6] - 1
        image = self.transform(image)
        label = self.target_transform(label)
        return image, label, img_path
    
    def getitem(self, idx):
        img_path = os.path.join(self.img_dir, self.labels.iloc[idx, 0])
        image = Image.open(img_path + '.png')
        label = self.labels.iloc[idx, 6] - 1
        image = self.transform(image, label)
        label = self.target_transform(label)
        return image, label


    
    
    


if __name__ == '__main__':
    pass