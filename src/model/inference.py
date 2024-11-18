import json
import torch
import torchvision.transforms as T

from torchvision.io import read_image

from src.model.pretrained import IMGNetCNNLoader
from src.datasets.data_transforms.img_transform import IMGTransforms
from src.datasets.data_transforms.spectral_transform import RealNDFourier

class Inferencer:
    
    def __init__(self, model_base, pretrained_path, transform, greyscale_opt, adversarial_opt, dual_model=False):
        self.model, self.params = self.load_model(model_base, pretrained_path)
        self.transform = self.load_transform(transform, greyscale_opt, adversarial_opt)
        self.greyscale_opt = greyscale_opt
        self.dual_model = dual_model
        if dual_model:
            self.inference = self.inference_dual
            normalize = T.Normalize([0.5], [0.5]) if greyscale_opt.greyscale_processing else T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            self.pix_transform = T.Compose([T.ConvertImageDtype(torch.float32),
                                            normalize])
            self.spec_transform = RealNDFourier(greyscale_opt.greyscale_fourier)
        else:
            self.inference = self.inference_single
    
    def load_model(self, model_base, pretrained_path):
        params = Inferencer.load_config(pretrained_path)
        if model_base == IMGNetCNNLoader:
            model, input_size = model_base.init_from_dict(params, pretrained_path)
            params['input_size'] = input_size
        else:
            model = model_base.init_from_dict(params, pretrained_path)
        return model, params
    
    def load_transform(self, transform, greyscale_opt, adversarial_opt):
        transforms = IMGTransforms(transform=transform,
                                   target_transform=None,
                                   input_size=self.params['input_size'], 
                                   device='cpu', 
                                   adversarial_opt=adversarial_opt, 
                                   greyscale_opt=greyscale_opt, 
                                   dataset_type='cross')
        return transforms.transform_val
    
    @staticmethod
    def load_config( pretrained_path):
        params = {}
        params_path = '/'.join(pretrained_path.split('/')[:-1])
        with open(params_path+'/'+'model_params.txt', 'r') as f:
            line = f.readline()
            while line.strip() !=  'RESULTS OBTAINED':
                line = f.readline().strip()
                if line and line != 'RESULTS OBTAINED':
                    param, value = line.split(':', maxsplit=1)
                    if value.startswith('{'):
                        params[param] = json.loads(value.replace("'", '"'))
                    elif value.startswith('['):
                        params[param] = [int(i.strip()) for i in value[1:-1].split(',')]
                    elif value.isdigit():    
                        params[param] = int(value)
                    else:
                        params[param] = value
        return params
    
    def __call__(self, img_path):
        result = self.inference(img_path)
        result_string = 'REAL' if result == 1 else 'GENERATED'
        print(f'Model classified {img_path} as: {result_string}')
    

    def inference_single(self, img_path):
        tensor = read_image(img_path)
        tensor = self.transform(tensor)
        out = self.model(tensor.unsqueeze(0))
        if out.squeeze().item() >= 0.5:
            return 1
        else:
            return 0
        
    def inference_dual(self, img_path):
        tensor = read_image(img_path)
        tensor = self.transform(tensor)
        x_p = self.pix_transform(tensor)
        x_f = self.spec_transform(tensor)
        out = self.model(x_p.unsqueeze(0), x_f.unsqueeze(0))
        if out.squeeze().item() >= 0.5:
            return 1
        else:
            return 0
        

if __name__ == '__main__':
    pass