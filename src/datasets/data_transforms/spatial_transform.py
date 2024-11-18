import torch
import numpy as np

from torchvision.io import encode_jpeg, decode_image
from torchvision import transforms as T
from src.datasets.data_transforms.cooccurrence import BandCMatrix, CrossCMatrix
from src.datasets.data_transforms.data_helper import StatTensor


class CustomCompose(T.Compose):
    
    # This class extends the torch Compose class
    # by having the option of passing the label
    # to specific transforms in the list
    
    # accepts a boolean list "target_required" (ie. [0,1,0])
    # that stores for which transforms the label
    # should be passed
    
    def __init__(self, transforms, target_required=False ):
        self.transforms = transforms
        if not target_required:
            self.target_required = self.check_for_custom_composes([0 for i in range(len(transforms))])
        else:
            self.target_required = self.check_for_custom_composes(target_required)
        
    def check_for_custom_composes(self, lst):
        for e, t in enumerate(self.transforms):
            if type(t) == CustomCompose:
                lst[e] = 1
        return lst
                
    def __call__(self, img, tar=None):
        for t, req in zip(self.transforms, self.target_required):
            if req:
                img = t(img, tar)
            else:
                img = t(img)
        return img

class SpatialTransforms:


    def __init__(self, transform, dataset_type, cross_offset_type=(0,0)):

        self.cross_offset_type = cross_offset_type
        self.blur = T.GaussianBlur(kernel_size=5, sigma=(0.1, 3.0))
        if dataset_type == 'cifar10':
            self.normalize = T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        elif dataset_type == 'cifar100':
            self.normalize = T.Normalize([0.5070, 0.4865, 0.4409], [0.2673, 0.2564, 0.2761])
        elif dataset_type in ['nips17', '140k_flickr_faces']:
            self.normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            
        if transform == 'augmented':
            # std transforms with augmentations
            self.transform_train = T.Compose([T.Lambda(lambda x: self.augment(x)),
                                        T.ConvertImageDtype(torch.float32)])

            self.transform_val = T.Compose([T.Lambda(lambda x: self.augment(x)),
                                        T.ConvertImageDtype(torch.float32)])

        elif transform == 'pretrained':
            # IMGNet format without augmentations
            self.transform_train = T.Compose([T.ConvertImageDtype(torch.float32),
                                        self.normalize])

            self.transform_val = CustomCompose([T.ConvertImageDtype(torch.float32),
                                        self.normalize], [0,0])

        elif transform == 'augmented_pretrained_imgnet':
            # IMGNet format with augmentations
            self.transform_train = T.Compose([T.Lambda(lambda x: self.augment(x)),
                                        T.ConvertImageDtype(torch.float32),
                                        self.normalize])

            self.transform_val = T.Compose([T.Lambda(lambda x: self.augment(x)),
                                        T.ConvertImageDtype(torch.float32),
                                        self.normalize])

            
        elif transform == 'band_cooccurrence':
            # to band coocurrence matrix 12 x 256 x 256
            self.transform_train = T.Compose([BandCMatTransform(),
                                        T.ConvertImageDtype(torch.float32)])

            self.transform_val = T.Compose([BandCMatTransform(),
                                        T.ConvertImageDtype(torch.float32)])

        elif transform == 'basic_pix_attn_cnn':
            self.transform_train = T.Compose([T.ColorJitter(),
                                        T.ConvertImageDtype(torch.float32)])

            self.transform_val = T.Compose([T.ConvertImageDtype(torch.float32)])
        
        elif transform == 'calc_avg_attack_norm':
            self.transform_train = self.identity

            self.transform_val = self.identity
    
    def identity(self, x):
        return x


    def to_band_cmat(self, tensor):
        tensor = StatTensor(tensor)
        mat = BandCMatrix(tensor)
        return torch.tensor(mat.get_concat_matrix())

    def to_cross_cmat(self, tensor):
        tensor = StatTensor(tensor)
        mat = CrossCMatrix(tensor, self.cross_offset_type)
        return torch.tensor(mat.get_concat_matrix())

    def augment(self, img, aug_prob=1.0):
        if np.random.choice([0,1], p=[1-aug_prob, aug_prob]):
            img = self.blur(img)
            img = self.jpeg_compression(img)
        return img

    def jpeg_compression(self, img):
        if img.dtype != torch.unint8:
            invert_norm = T.Compose([ T.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                T.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])
            img = invert_norm(img)
            img = img * 255
        rand_compression_factor = 90
        compressed = encode_jpeg(img, rand_compression_factor)
        return decode_image(compressed)

class BandCMatTransform:

    def __init__(self):
        pass

    def __call__(self, tensor):
        tensor = StatTensor(tensor)
        mat = BandCMatrix(tensor)
        return torch.tensor(mat.get_concat_matrix())
        

if __name__ == '__main__':
    pass