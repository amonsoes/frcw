import torch

from torchvision import transforms as T
from src.datasets.data_transforms.spatial_transform import SpatialTransforms, CustomCompose
from src.adversarial.spatial import Augmenter, AttackLoader
from torchvision.transforms import InterpolationMode


class Clamp(torch.nn.Module):
    def __init__(self):
        super().__init__() 

    def forward(self, image):
        transformed_image = torch.clamp(image, min=0.0, max=1.0)
        return transformed_image

class PostTransforms:
    
    def __init__(self,
                transform, 
                input_size, 
                device, 
                adversarial_opt, 
                dataset_type,
                model,
                jpeg_compression,
                jpeg_compression_rate,
                val_base_trm_size,
                *args, 
                **kwargs):
        self.device = device
        self.input_size = input_size
        self.adversarial_opt = adversarial_opt
        self.dataset_type = dataset_type
        self.val_base_trm_size = val_base_trm_size

        self.transform_train, self.transform_val = self.get_main_trm(transform, *args, **kwargs)
        
        if jpeg_compression:
            self.transform_train, self.transform_val = self.set_compression(self.transform_train, self.transform_val, jpeg_compression_rate)
        
        if adversarial_opt.adversarial:
            self.transform_train, self.transform_val = self.get_adv_trm(self.transform_val, model, val_base_trm_size)


    def set_compression(self, train_transform, val_transform, jpeg_compression_rate):
        if self.adversarial_opt.attack_compression:
            print('Warning: JPEG compression already used after attack. Skipping JPEG compression')
        else:
            augmenter = Augmenter(compression_rate=jpeg_compression_rate)
            train_transform = T.Compose([augmenter.jpeg_compression ,train_transform])
            val_transform = CustomCompose([augmenter.jpeg_compression, val_transform])
        return train_transform, val_transform

    def get_main_trm(self, transform, *args, **kwargs):
        if transform in ['standard', 
                        'augmented', 
                        'pretrained',
                        'augmented_pretrained_imgnet', 
                        'band_cooccurrence', 
                        'cross_cooccurrence', 
                        'basic_pix_attn_cnn', 
                        'compute_sdn',
                        'ycbcr_transform',
                        'calc_avg_attack_norm']:
            
            transforms = SpatialTransforms(transform,
                                           dataset_type=self.dataset_type,
                                           *args, 
                                           **kwargs)

            transform_train = transforms.transform_train
            transform_val = transforms.transform_val

        else:
            raise ValueError('WRONG TRANSFORM INPUT. change in options')
        
        return transform_train, transform_val

    def get_adv_trm(self, transform_val, model, val_base_trm_size):

        # define spatial adv 
        
        self.transform_val_spatial = None
    
        
        # define input size and resize depending on surrogate model
        
        surrogate_model_name = self.adversarial_opt.surrogate_model_params.surrogate_model
        #surrogate_input_size = self.adversarial_opt.surrogate_model_params.surrogate_input_size
        surrogate_input_size = 224 if self.dataset_type in ['nips17', '140k_flickr_faces'] else 32
        surrogate_trm_name = self.adversarial_opt.surrogate_model_params.surrogate_transform
        surrogate_transforms = SpatialTransforms(surrogate_trm_name,
                                    self.dataset_type)
        surrogate_trm = surrogate_transforms.transform_val
        
        internal_jpeg_compr = None
        internal_compression_rate = None
        if self.adversarial_opt.attack_compression and self.adversarial_opt.spatial_adv_type in ['rcw', 'far']:
            adv_type = self.adversarial_opt.spatial_adv_type
            compression_rate = self.adversarial_opt.compression_rate[0] if adv_type == 'rcw' else self.adversarial_opt.spatial_attack_params.far_jpeg_quality 
            augmenter = Augmenter(compression_rate=compression_rate)
            internal_jpeg_compr = augmenter.jpeg_compression
            internal_compression_rate = compression_rate
        attack_loader = AttackLoader(attack_type=self.adversarial_opt.spatial_adv_type,
                                    device=self.device,
                                    dataset_type=self.dataset_type,
                                    model_trms=transform_val,
                                    model=model,
                                    surrogate_model=surrogate_model_name,
                                    surrogate_model_trms=surrogate_trm,
                                    input_size=self.input_size,
                                    jpeg_compr_obj=internal_jpeg_compr,
                                    internal_jpeg_quality=internal_compression_rate,
                                    **self.adversarial_opt.spatial_attack_params.__dict__)
        self.attack = attack_loader.attack
        self.surrogate_trm = surrogate_trm
        if surrogate_input_size != val_base_trm_size:
            surrogate_pre_resize = T.Resize(surrogate_input_size,  interpolation=InterpolationMode.BILINEAR)
            surrogate_post_resize = T.Resize(self.input_size,  interpolation=InterpolationMode.BILINEAR)
            attack = CustomCompose([surrogate_pre_resize, Clamp(), self.attack, surrogate_post_resize], [0, 0, 1, 0])
        else:
            attack = self.attack
        if self.adversarial_opt.attack_compression:
            if self.adversarial_opt.consecutive_attack_compr:
                compressions = []
                for compression in self.adversarial_opt.compression_rate:
                    augmenter = Augmenter(compression_rate=compression)
                    compressions.append(augmenter.jpeg_compression)
                compressions = CustomCompose(compressions, [0 for i in range(len(compressions))])
                self.transform_val_spatial = CustomCompose([attack, compressions, transform_val], [1, 0, 0])    
            else:
                compression_rate = self.adversarial_opt.compression_rate[0]
                augmenter = Augmenter(compression_rate=compression_rate)
                self.transform_val_spatial = CustomCompose([ attack, augmenter.jpeg_compression, transform_val], [1, 0, 0])
        else:
            self.transform_val_spatial = CustomCompose([attack, transform_val], [1, 0])
        
                
        transform_train_adv = self.identity
        transform_val_adv = self.transform_val_spatial
        
        return transform_train_adv, transform_val_adv

    def identity(self, x):
        return x

class PreTransforms:

    def __init__(self,
                input_size, 
                device,
                dataset_type,
                target_transform,
                adversarial_opt):
        
        self.device = device
        self.input_size = input_size
        self.dataset_type = dataset_type
        self.adversarial_opt = adversarial_opt
        self.transform_train, self.transform_val, self.val_base_trm_size = self.get_base_trm()
        if target_transform == None:
            self.target_transform = self.identity
        
    def identity(self, x):
        return x
    
    
    def get_base_trm(self):
        # method to get base trms that are shared across all models

        if self.input_size == 299:
            resize = T.Resize(299, interpolation=InterpolationMode.BILINEAR)
            crop_size = 299
        elif self.input_size == 32: # for CIFAR models
            resize = T.Resize(32,  interpolation=InterpolationMode.BILINEAR)
            crop_size = 32
        else:
            resize = T.Resize(256,  interpolation=InterpolationMode.BILINEAR)
            crop_size = 224
        
        #val_base_trm_size is needed for correct loading of adversarial attacks
        val_base_trm_size = crop_size
        
        if self.dataset_type == '140k_flickr_faces':
        
            train_base_transform = T.Compose([resize,
                                            T.RandomResizedCrop(crop_size, scale=(0.5,1)),
                                            T.RandomHorizontalFlip(),
                                            T.ConvertImageDtype(torch.float32),
                                            Clamp()])
            val_base_transform = T.Compose([resize,
                                             T.CenterCrop(crop_size),
                                             T.ConvertImageDtype(torch.float32),
                                             Clamp()])
    
        elif self.dataset_type == 'nips17':

            train_base_transform = T.Compose([resize,
                                            T.RandomResizedCrop(crop_size, scale=(0.5,1)),
                                            T.RandomHorizontalFlip(),
                                            T.ToTensor(),
                                            Clamp()])
            val_base_transform = T.Compose([resize,
                                             T.CenterCrop(crop_size),
                                             T.ToTensor(),
                                             Clamp()])
    
        
        elif self.dataset_type == 'cifar10':

            train_base_transform = T.Compose([resize,
                                            T.RandomHorizontalFlip(),
                                            T.ConvertImageDtype(torch.float32),
                                            Clamp()])
            val_base_transform = T.Compose([resize, 
                                            T.ConvertImageDtype(torch.float32), 
                                            Clamp()])
            
        else:
            raise ValueError('ERROR : dataset type currently not supported for image transforms.')
        
        #val_base_trm_size is needed for correct loading of adversarial attacks
        return train_base_transform, val_base_transform, val_base_trm_size
    
        
        

if __name__ == '__main__':
    pass