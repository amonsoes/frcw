import os
import numpy as np

from PIL import Image

class StatImg:

    def __init__(self, path):
        self.img = Image.open(path)
        self.img_tensor = np.array(self.img)
        self.r, self.g, self.b = self.img.split()
        self.r_tensor = np.array(self.r)
        self.b_tensor = np.array(self.g)
        self.g_tensor = np.array(self.b)
        self.len_col = len(self.img_tensor)
        self.len_row = len(self.img_tensor[0])
    
    def get_channels_tensor(self):
        return self.r_tensor, self.g_tensor, self.b_tensor

    def get_channels(self):
        return self.r, self.g, self.b


class StatTensor:

    def __init__(self, tensor):
        self.img_tensor = tensor
        self.r_tensor = tensor[0]
        self.b_tensor = tensor[1]
        self.g_tensor = tensor[2]
        self.len_col = len(self.img_tensor)
        self.len_row = len(self.img_tensor[0])
    
    def get_channels_tensor(self):
        return self.r_tensor, self.g_tensor, self.b_tensor

    def get_channels(self):
        return self.r, self.g, self.b


class ImgBase:

    def __init__(self, dir, batch_size=2):
        self.dir = dir
        self.batch_size = batch_size
        if self.batch_size > 1:
            self.batch_gener = self.get_batch()
        else:
            self.batch_gener = self.get_img()
    
    def get_batch(self):
        img_gener = self.get_img()
        batch_control = 0
        tensor = []
        for img in img_gener:
            tensor.append(img)
            batch_control +=1
            if batch_control == self.batch_size:
                batch_control = 0
                yield tensor
                tensor = []

    def get_img(self):
        for _, _, files in os.walk(self.dir):
            for name in files:
                yield StatImg(self.dir + '/' + name)
        

if __name__ == '__main__':
    pass