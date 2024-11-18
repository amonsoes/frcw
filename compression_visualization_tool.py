import os
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from torchvision.io import read_image, encode_jpeg, decode_jpeg

if __name__ == '__main__':
    data_dir_path = './data/nips17/images/'
    files = os.listdir(data_dir_path)
    print(files)
    
    to_grey = T.Grayscale()
    
    for filename in files:
        
        columns = 3
        rows = 2
        
        img_tensor = read_image(data_dir_path + '/' + filename)
        compressed = encode_jpeg(img_tensor, quality=90)
        compressed_tensor = decode_jpeg(compressed) 
        img_tensor = img_tensor / 255
        compressed_tensor = compressed_tensor / 255
        delta = (img_tensor - compressed_tensor).clip(max=1.0, min=0.0)
        delta = delta / delta.max()
        
        orig_y = to_grey(img_tensor)
        compressed_y = to_grey(compressed_tensor)
        
        orig_chroma = img_tensor - orig_y
        compressed_chroma = compressed_tensor - compressed_y
        
        delta_y = orig_y - compressed_y
        l2_y = delta_y.pow(2).sum().sqrt()
        delta_y = torch.abs(delta_y) / torch.abs(delta_y).max()
        
        delta_chroma = orig_chroma - compressed_chroma
        l2_chroma = delta_chroma.pow(2).sum().sqrt()  
        delta_chroma = torch.abs(delta_chroma) / torch.abs(delta_chroma).max()
        print('ig separated')  
        
        # show compressed, delta and uncompressed
        fig = plt.figure(figsize=(10, 12))        
        
        fig.add_subplot(1, columns, 1, title='compressed img')  
        plt.imshow(compressed_tensor.permute(1,2,0))
        fig.add_subplot(1, columns, 2, title='complete delta img')  
        plt.imshow(delta.permute(1,2,0))
        fig.add_subplot(1, columns, 3, title=f'delta y dist:{l2_y}')  
        plt.imshow(delta_y.permute(1,2,0))
        
        # show delta_y delta_r delta_g delta_b
        fig.add_subplot(rows, columns, 1, title=f'delta chroma dist:{l2_chroma}')  
        plt.imshow(delta_chroma[0])
        fig.add_subplot(rows, columns, 2, title='delta chroma g')  
        plt.imshow(delta_chroma[1])
        fig.add_subplot(rows, columns, 3, title='delta chroma b')  
        plt.imshow(delta_chroma[2])
             
        
        plt.show()