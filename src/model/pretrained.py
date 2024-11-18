from __future__ import print_function
from __future__ import division
import timm
import torch
import torch.nn as nn

from torchvision.models import resnet152, ResNet152_Weights, densenet201, DenseNet201_Weights, inception_v3, Inception_V3_Weights, ResNet50_Weights, resnet50
from src.model.xception import XceptionLoader, XceptionSettings

from src.model.preactresnet import PreActResNet18
from src.model.madry_resnet import madry_resnet50


class CNNLoader:

    def __init__(self, loading_dir='', adversarial_pretrained_opt=None):
        self.loading_dir = loading_dir
        self.adversarial_pretrained_opt = adversarial_pretrained_opt
        if adversarial_pretrained_opt != None:
            if isinstance(adversarial_pretrained_opt, str):
                self.load_adversarial_pretrained = True
            else:
                self.load_adversarial_pretrained = True if adversarial_pretrained_opt.adversarial_pretrained else False
        else:
            self.load_adversarial_pretrained = False
            
    def set_params_requires_grad(self, model, feature_extractor):
        if feature_extractor:
            for param in model.parameters():
                param.requires_grad = False
    
    def transfer(self, model_name, num_classes, feature_extract, device):

        device = torch.device(device)
        #print(f'\nloading model....\n model: {model_name}\n frozen weights: {feature_extract}\n on device: {device}')

        if self.load_adversarial_pretrained == True:
            if model_name.endswith('pgd')  :
                self.load_adversarial_pretrained = True
                self.adv_train_protocol = 'pgd'
                if num_classes == 1000:
                    self.loading_dir = './saves/models/Adversarial/pgd_models/imagenet_linf_4.pt'
                elif num_classes == 10:
                    self.loading_dir = './saves/models/Adversarial/pgd_models/cifar10_resnet50_linf_8255.pt'
            else:
                self.load_adversarial_pretrained = True
                self.adv_train_protocol = 'fbf'
                if num_classes == 1000:
                    self.loading_dir = './saves/models/Adversarial/fbf_models/imagenet_model_weights_4px.pth.tar'
                elif num_classes == 10:
                    self.loading_dir = './saves/models/Adversarial/fbf_models/cifar_model_weights_30_epochs.pth'
            if num_classes == 1000:
                model_ft, input_size = self.load_adv_pretrained_for_imgnet(device, feature_extract)
            elif num_classes == 10:
                model_ft, input_size = self.load_adv_pretrained_for_cifar10(device, feature_extract)
        else:
            if  self.loading_dir: # load from pretrained for inference
                model_ft, input_size = self.load_pretrained_for_inference(model_name, num_classes, device, feature_extract)
            elif num_classes == 1000: # no need for projection head as data is imgnet 
                model_ft, input_size = self.load_pretrained_for_imgnet(model_name, device, feature_extract)  
            elif num_classes == 10: # load CIFAR10 Resnet
                
                if model_name == 'resnet':
                
                    # loads a 20-layer CIFAR10 resnet
                    model_ft = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
                    model_ft.to(device)
                    model_ft.device = device
                    input_size = 32
                
                elif model_name == 'vgg':
                    model_ft = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg11_bn", pretrained=True)
                    model_ft.to(device)
                    model_ft.device = device
                    input_size = 32
                    
            else: # load from pretrained for transfer

                model_ft, input_size = self.load_pretrained_for_transfer(model_name, device=device, num_classes=num_classes, feature_extract=feature_extract)
        return model_ft, input_size
    
    @classmethod
    def init_from_dict(cls, params, pretrained_path):
        loader = CNNLoader(pretrained_path)
        cnn, input_size = loader.transfer(params['model_name'], 1, feature_extract=False, device='cpu')
        cnn.model_name = params['model_name']
        return cnn, input_size
    
    def load_pretrained_for_inference(self, model_name, num_classes, device, feature_extract):
        if model_name == "resnet":
            pretrained_weights = torch.load(self.loading_dir, map_location=device)
            model_ft = resnet152()
            self.set_params_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            model_ft.load_state_dict(pretrained_weights)
            input_size = 224

        elif model_name == "densenet":
            pretrained_weights = torch.load(self.loading_dir, map_location=device)
            model_ft = densenet201()
            self.set_params_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, num_classes)
            model_ft.load_state_dict(pretrained_weights)
            input_size = 224

        elif model_name == "inception":
            pretrained_weights = torch.load(self.loading_dir, map_location=device)
            model_ft = inception_v3()
            self.set_params_requires_grad(model_ft, feature_extract)
            # auxiliary net fc
            num_ftrs = model_ft.AuxLogits.fc.in_features
            model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
            # primary model fc
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            model_ft.load_state_dict(pretrained_weights)
            input_size = 299

        elif model_name == "xception":
            settings = XceptionSettings(device=device, is_truncated=False, url=self.loading_dir, num_classes=num_classes)
            xception_loader = XceptionLoader(settings=settings)
            model_ft = xception_loader.load()
            input_size = 299
        
        elif model_name == "coatnet":
            model_ft = timm.create_model('coatnet_1_rw_224', checkpoint_path=self.loading_dir, in_chans=3, num_classes=num_classes)
            input_size = 224

        elif model_name == "vit":
            model_ft = timm.create_model("vit_huge_patch14_224_in21k", checkpoint_path=self.loading_dir, num_classes=num_classes)
            input_size = 224

        else:
            raise ValueError('wrong model name')
        
        model_ft.device = device
        model_ft.to(device)

        #print(f'\n\nfinished loading.\n{model_ft}')
        
        return model_ft, input_size

    def load_pretrained_for_imgnet(self, model_name, device, feature_extract):
        if model_name == "resnet":
            model_ft = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
            self.set_params_requires_grad(model_ft, feature_extract)
            input_size = 224

        elif model_name == "densenet":
            model_ft = densenet201(weights=DenseNet201_Weights.IMAGENET1K_V1)
            self.set_params_requires_grad(model_ft, feature_extract)
            input_size = 224

        elif model_name == "inception":
            pretrained_weights = torch.load('./saves/models/ImgnetCNN/2023-1-26/inception_v3_google-1a9a5a14.pth')
            model_ft = inception_v3()
            #model_ft = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
            self.set_params_requires_grad(model_ft, feature_extract)
            model_ft.load_state_dict(pretrained_weights)
            model_ft.to(device)
            input_size = 299
            model_ft.eval()

        elif model_name == 'adv-inception':
            model_ft = timm.create_model('adv_inception_v3', pretrained=True)
            model_ft.eval()
            input_size = 299
            
        
        elif model_name == "xception":
            settings = XceptionSettings(device=device, is_truncated=False, num_classes=1000)
            xception_loader = XceptionLoader(settings=settings)
            model_ft = xception_loader.load()
            input_size = 299

        else:
            raise ValueError('wrong model name')
        
        model_ft.device = device
        model_ft.to(device)

        #print(f'\n\nfinished loading.\n{model_ft}')
        
        return model_ft, input_size

    def load_adv_pretrained_for_imgnet(self, device, feature_extract):
        
        print('\nWARNING: using the adversarial pretrained option essentially disables \
            the usage of the option "model_name. Using it results in the respective loading \
            of a particular model that was trained with the adv training protocol chosen in options.py"\n')
        
        if isinstance(self.adversarial_pretrained_opt, str):
            adv_training_protocol = self.adversarial_pretrained_opt
        else:
            adv_training_protocol = self.adversarial_pretrained_opt.adv_pretrained_protocol
            
        if adv_training_protocol == 'fgsm':
            model_ft = timm.create_model('adv_inception_v3', pretrained=True)
            model_ft.eval()
            input_size = 299

        elif adv_training_protocol == 'fbf':
            checkpoint = torch.load(self.loading_dir, map_location=device)
            model_ft = resnet50()
            model_ft = torch.nn.DataParallel(model_ft)
            self.set_params_requires_grad(model_ft, feature_extract)
            model_ft.load_state_dict(checkpoint['state_dict'])
            input_size = 224
            model_ft = model_ft.module # extract model from DataParallel Wrapper
        
        elif adv_training_protocol == 'pgd':
            # resnet50 was trained with PGDLinf / eps = 4/255
            # see details to model here: https://github.com/MadryLab/robustness
            checkpoint = torch.load(self.loading_dir, map_location=device)
            model_ft = resnet50()
            #model_ft = torch.nn.DataParallel(model_ft)
            self.set_params_requires_grad(model_ft, feature_extract)
            new_state_dict = self.remove_data_parallel(checkpoint['model'], 'module.model.')
            model_ft.load_state_dict(new_state_dict)
            input_size = 224
            
        else:
            raise ValueError('adv training protocol not recognized')
        
        model_ft.device = device
        model_ft.to(device)

        #print(f'\n\nfinished loading.\n{model_ft}')
        
        return model_ft, input_size

    def load_adv_pretrained_for_cifar10(self, device, feature_extract):
        
        print('\nWARNING: using the adversarial pretrained option essentially disables \
            the usage of the option "model_name. Using it results in the respective loading \
            of a particular model that was trained with the adv training protocol chosen in options.py"\n')

        if isinstance(self.adversarial_pretrained_opt, str):
            adv_training_protocol = self.adversarial_pretrained_opt
        elif self.adversarial_pretrained_opt == None:
            adv_training_protocol = self.adv_train_protocol
        else:
            adv_training_protocol = self.adversarial_pretrained_opt.adv_pretrained_protocol
        
        if adv_training_protocol == 'fbf':
            checkpoint = torch.load(self.loading_dir, map_location=device)
            model_ft = PreActResNet18()
            self.set_params_requires_grad(model_ft, feature_extract)
            model_ft.load_state_dict(checkpoint)
            input_size = 32
        
        elif adv_training_protocol == 'pgd':
            checkpoint = torch.load(self.loading_dir, map_location=device)
            model_ft = madry_resnet50()
            #model_ft = torch.nn.DataParallel(model_ft)
            self.set_params_requires_grad(model_ft, feature_extract)
            new_state_dict = self.remove_data_parallel(checkpoint['state_dict'], 'module.model.')
            model_ft.load_state_dict(new_state_dict)
            input_size = 32


        model_ft.device = device
        model_ft.to(device)

        #print(f'\n\nfinished loading.\n{model_ft}')
        
        return model_ft, input_size

    def remove_data_parallel(self, state_dict, prefix):
        # prefix is length of prefix in module names of state_dict
        # those will ned to be removed in order to load model properly without DataParalell
        prefix_len = len(prefix)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith(prefix):
                name = k[prefix_len:] # remove prefix eg.'.module.model'
                new_state_dict[name] = v
        return new_state_dict


    def load_pretrained_for_transfer(self, model_name, num_classes, device, feature_extract):
        if model_name == "resnet":
            model_ft = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
            self.set_params_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "densenet":
            model_ft = densenet201(weights=DenseNet201_Weights.IMAGENET1K_V1)
            self.set_params_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "inception":
            model_ft = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
            self.set_params_requires_grad(model_ft, feature_extract)
            # auxiliary net fc
            num_ftrs = model_ft.AuxLogits.fc.in_features
            model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
            # primary model fc
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 299
        
        elif model_name == "xception":
            settings = XceptionSettings(device=device, is_truncated=False, num_classes=num_classes)
            xception_loader = XceptionLoader(settings=settings)
            model_ft = xception_loader.load()
            input_size = 299

        elif model_name == "coatnet":
            model_ft = timm.create_model('coatnet_1_rw_224', pretrained=True, in_chans=3, num_classes=num_classes)
            input_size = 224
        
        elif model_name == "vit":
            model_ft = timm.create_model("vit_huge_patch14_224_in21k", pretrained=True, num_classes=num_classes)
            input_size = 224


        else:
            raise ValueError('wrong model name')
        
        model_ft.device = device
        model_ft.to(device)
        
        return model_ft, input_size

if __name__ == '__main__':
    pass