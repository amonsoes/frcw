import torch

# Standard libraries
import itertools
import numpy as np
# PyTorch
import torch
import torch.nn as nn

from src.adversarial.attack_base import Attack
from diff_jpeg import diff_jpeg_coding



############ Attack #################

class JIFGSM(Attack):
    r"""
    After Richard Shin et al. 2017
    or
    After C. Reich et al. 2023

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)

    .. note:: If steps set to 0, steps will be automatically decided following the paper.

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.BIM(model, eps=8/255, alpha=2/255, steps=10)
        >>> adv_images = attack(images, labels)
    """
    def __init__(self, 
                model, 
                surrogate_loss,  
                model_trms, 
                eps=8/255, 
                alpha=2/255, 
                steps=7, 
                target_mode='default', 
                N=4,
                jifgsm_compr_type='shin'):
        super().__init__("BIM", model, model_trms)
        self.eps = eps
        self.alpha = eps / steps
        if steps == 0:
            self.steps = int(min(eps*255 + 4, 1.25*eps*255))
        else:
            self.steps = steps
        self.supported_mode = ['default', 'targeted']
        self.loss = surrogate_loss
        self.model_trms = model_trms
        self.compression_rates = []
        compression = 99
        self.N_comp_rates = N
        for i in range(N): # in 5-step decrements
            self.compression_rates.append(compression)
            compression -= 5
        if target_mode != 'default':
            self.set_target_mode(target_mode)
        
        if jifgsm_compr_type == 'shin':
            self.compress = self.compress_shin
        elif jifgsm_compr_type == 'reich':
            self.compress = self.compress_reich
    
    def compress(self, images, compression_rate):
        return self.compress_fn(self, images, compression_rate)
    
    def compress_shin(self, images, compression_rate):
        compress = DiffJPEG(height=224, width=224, differentiable=True, quality=compression_rate, device=self.device)
        return compress(images)
    
    def compress_reich(self, images, compression_rate):
        batch_size = images.shape[0]
        img = images * 255
        compressed = diff_jpeg_coding(image_rgb=img, jpeg_quality=torch.tensor([compression_rate]*batch_size).to(self.device))
        compressed_img = (compressed / 255).clip(min=0., max=1.)
        return compressed_img
    

    def set_target_mode(self, mode):
        if mode == 'least_likely':
            self.set_mode_targeted_least_likely()
        elif mode == 'most_likely':
            self.set_mode_targeted_most_likely()
        else:
            print('WARNING: set_target_mode was set to random. If unwanted, change "target_mode" arg to either "least_likely" or "most_likely".')
            self.set_mode_targeted_random()
            
            
    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        
        if self.targeted:
            target_labels = self.get_target_label(images, labels)
        
        adv_images = images.clone().detach()
        
        for i in range(self.steps):
            
            ensemble_grad = torch.zeros_like(adv_images).detach().to(self.device)    
            grad_list = []
            loss_tensor = torch.zeros(self.N_comp_rates)

            
            for e, compression_rate in enumerate(self.compression_rates):
                adv_images_i = adv_images.detach()
                adv_images_i.requires_grad = True
                comp_images = self.compress(adv_images_i, compression_rate)
                outputs = self.get_logits(self.model_trms(comp_images))
                

            # Calculate loss
                if self.targeted:
                    cost = -self.loss(outputs, target_labels)
                else:
                    cost = self.loss(outputs, labels)
                loss_tensor[e] = cost

                # Update adversarial images
                grad = torch.autograd.grad(cost, adv_images_i,
                                        retain_graph=False,
                                        create_graph=False)[0]
                
                grad_list.append(grad)
            
            total_cost_exp = loss_tensor.exp().sum()
            for cost, grad in zip(loss_tensor, grad_list):
                ensemble_grad +=  (1 - (torch.exp(cost)) / total_cost_exp) * grad

            adv_images = adv_images + self.alpha*ensemble_grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            #self.observer(delta.squeeze().detach())
            adv_images = torch.clamp(images + delta, min=0., max=1.).detach()
        
        if self.targeted:
            return adv_images, target_labels
        else:
            return adv_images
    
    
#########################################
#
# Original Author: Michael Lomnitz
# Original Repo : https://github.com/mlomnitz/DiffJPEG/
# Differentiable JPEG ported from tf to torch 
# After Richard Shin et al. 2017
#
######### DIFFJpeg Module ##############
    

class DiffJPEG(nn.Module):
    def __init__(self, height, width, differentiable=True, quality=80, device='cpu'):
        ''' Initialize the DiffJPEG layer
        Inputs:
            height(int): Original image hieght
            width(int): Original image width
            differentiable(bool): If true uses custom differentiable
                rounding function, if false uses standrard torch.round
            quality(float): Quality factor for jpeg compression scheme. 
        '''
        super(DiffJPEG, self).__init__()
        self.device = device
        if differentiable:
            rounding = diff_round
        else:
            rounding = torch.round
        factor = quality_to_factor(quality)
        self.compress = compress_jpeg(device, rounding=rounding, factor=factor)
        self.decompress = decompress_jpeg(device, height, width, rounding=rounding,
                                          factor=factor)

    def forward(self, x):
        '''

        '''
        y, cb, cr = self.compress(x)
        recovered = self.decompress(y, cb, cr)
        return recovered
    

#########################################


############ Compression ###############
    


class rgb_to_ycbcr_jpeg(nn.Module):
    """ Converts RGB image to YCbCr
    Input:
        image(tensor): batch x 3 x height x width
    Outpput:
        result(tensor): batch x height x width x 3
    """
    def __init__(self, device):
        super(rgb_to_ycbcr_jpeg, self).__init__()
        self.device = device
        matrix = np.array(
            [[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5],
             [0.5, -0.418688, -0.081312]], dtype=np.float32).T
        self.shift = nn.Parameter(torch.tensor([0., 128., 128.])).to(self.device)
        #
        self.matrix = nn.Parameter(torch.from_numpy(matrix)).to(self.device)

    def forward(self, image):
        image = image.permute(0, 2, 3, 1)
        result = torch.tensordot(image, self.matrix, dims=1) + self.shift
    #    result = torch.from_numpy(result)
        result.view(image.shape)
        return result



class chroma_subsampling(nn.Module):
    """ Chroma subsampling on CbCv channels
    Input:
        image(tensor): batch x height x width x 3
    Output:
        y(tensor): batch x height x width
        cb(tensor): batch x height/2 x width/2
        cr(tensor): batch x height/2 x width/2
    """
    def __init__(self, device):
        super(chroma_subsampling, self).__init__()
        self.device = device

    def forward(self, image):
        image_2 = image.permute(0, 3, 1, 2).clone()
        avg_pool = nn.AvgPool2d(kernel_size=2, stride=(2, 2),
                                count_include_pad=False).to(self.device)
        cb = avg_pool(image_2[:, 1, :, :].unsqueeze(1))
        cr = avg_pool(image_2[:, 2, :, :].unsqueeze(1))
        cb = cb.permute(0, 2, 3, 1)
        cr = cr.permute(0, 2, 3, 1)
        return image[:, :, :, 0], cb.squeeze(3), cr.squeeze(3)


class block_splitting(nn.Module):
    """ Splitting image into patches
    Input:
        image(tensor): batch x height x width
    Output: 
        patch(tensor):  batch x h*w/64 x h x w
    """
    def __init__(self, device):
        super(block_splitting, self).__init__()
        self.k = 8
        self.device = device

    def forward(self, image):
        height, width = image.shape[1:3]
        batch_size = image.shape[0]
        image_reshaped = image.view(batch_size, height // self.k, self.k, -1, self.k)
        image_transposed = image_reshaped.permute(0, 1, 3, 2, 4)
        return image_transposed.contiguous().view(batch_size, -1, self.k, self.k)
    

class dct_8x8(nn.Module):
    """ Discrete Cosine Transformation
    Input:
        image(tensor): batch x height x width
    Output:
        dcp(tensor): batch x height x width
    """
    def __init__(self, device):
        super(dct_8x8, self).__init__()
        self.device = device
        tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
        for x, y, u, v in itertools.product(range(8), repeat=4):
            tensor[x, y, u, v] = np.cos((2 * x + 1) * u * np.pi / 16) * np.cos(
                (2 * y + 1) * v * np.pi / 16)
        alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
        #
        self.tensor =  nn.Parameter(torch.from_numpy(tensor).float()).to(self.device)
        self.scale = nn.Parameter(torch.from_numpy(np.outer(alpha, alpha) * 0.25).float()).to(self.device)
        
    def forward(self, image):
        image = image - 128
        result = self.scale * torch.tensordot(image, self.tensor, dims=2)
        result.view(image.shape)
        return result


class y_quantize(nn.Module):
    """ JPEG Quantization for Y channel
    Input:
        image(tensor): batch x height x width
        rounding(function): rounding function to use
        factor(float): Degree of compression
    Output:
        image(tensor): batch x height x width
    """
    def __init__(self, device, rounding, factor=1):
        super(y_quantize, self).__init__()
        self.device = device
        self.rounding = rounding
        self.factor = factor
        self.y_table = y_table.to(self.device)

    def forward(self, image):
        image = image.float() / (self.y_table * self.factor)
        image = self.rounding(image)
        return image


class c_quantize(nn.Module):
    """ JPEG Quantization for CrCb channels
    Input:
        image(tensor): batch x height x width
        rounding(function): rounding function to use
        factor(float): Degree of compression
    Output:
        image(tensor): batch x height x width
    """
    def __init__(self, device, rounding, factor=1):
        super(c_quantize, self).__init__()
        self.device = device
        self.rounding = rounding
        self.factor = factor
        self.c_table = c_table.to(self.device)

    def forward(self, image):
        image = image.float() / (self.c_table * self.factor)
        image = self.rounding(image)
        return image


class compress_jpeg(nn.Module):
    """ Full JPEG compression algortihm
    Input:
        imgs(tensor): batch x 3 x height x width
        rounding(function): rounding function to use
        factor(float): Compression factor
    Ouput:
        compressed(dict(tensor)): batch x h*w/64 x 8 x 8
    """
    def __init__(self, device, rounding=torch.round, factor=1):
        super(compress_jpeg, self).__init__()
        self.device = device
        self.jpeg_quality = factor
        self.l1 = nn.Sequential(
            rgb_to_ycbcr_jpeg(device=self.device),
            chroma_subsampling(device=self.device)
        )
        self.l2 = nn.Sequential(
            block_splitting(device=self.device),
            dct_8x8(device=self.device)
        )
        self.c_quantize = c_quantize(device, rounding=rounding, factor=factor)
        self.y_quantize = y_quantize(device, rounding=rounding, factor=factor)

    def forward(self, image):
        y, cb, cr = self.l1(image*255)
        components = {'y': y, 'cb': cb, 'cr': cr}
        for k in components.keys():
            comp = self.l2(components[k])
            if k in ('cb', 'cr'):
                comp = self.c_quantize(comp)
            else:
                comp = self.y_quantize(comp)

            components[k] = comp

        return components['y'], components['cb'], components['cr']


#########################################


############ Decompression ###############

class y_dequantize(nn.Module):
    """ Dequantize Y channel
    Inputs:
        image(tensor): batch x height x width
        factor(float): compression factor
    Outputs:
        image(tensor): batch x height x width

    """
    def __init__(self, device, factor=1):
        super(y_dequantize, self).__init__()
        self.device = device
        self.y_table = y_table.to(self.device)
        self.factor = factor

    def forward(self, image):
        return image * (self.y_table * self.factor)


class c_dequantize(nn.Module):
    """ Dequantize CbCr channel
    Inputs:
        image(tensor): batch x height x width
        factor(float): compression factor
    Outputs:
        image(tensor): batch x height x width

    """
    def __init__(self, device, factor=1):
        super(c_dequantize, self).__init__()
        self.device = device
        self.factor = factor
        self.c_table = c_table.to(self.device)

    def forward(self, image):
        return image * (self.c_table * self.factor)
    
class y_quantize_no_rounding(nn.Module):
    """ JPEG Quantization for Y channel
    Input:
        image(tensor): batch x height x width
        rounding(function): rounding function to use
        factor(float): Degree of compression
    Output:
        image(tensor): batch x height x width
    """
    def __init__(self, device, factor=1):
        super(y_quantize_no_rounding, self).__init__()
        self.device = device
        self.factor = factor
        self.y_table = y_table.to(self.device)

    def forward(self, image):
        image = image.float() / (self.y_table * self.factor)
        return image


class c_quantize_no_rounding(nn.Module):
    """ JPEG Quantization for CrCb channels
    Input:
        image(tensor): batch x height x width
        rounding(function): rounding function to use
        factor(float): Degree of compression
    Output:
        image(tensor): batch x height x width
    """
    def __init__(self, device, factor=1):
        super(c_quantize_no_rounding, self).__init__()
        self.device = device
        self.factor = factor
        self.c_table = c_table.to(self.device)

    def forward(self, image):
        image = image.float() / (self.c_table * self.factor)
        return image


class idct_8x8(nn.Module):
    """ Inverse discrete Cosine Transformation
    Input:
        dcp(tensor): batch x height x width
    Output:
        image(tensor): batch x height x width
    """
    def __init__(self, device):
        super(idct_8x8, self).__init__()
        self.device = device
        alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
        self.alpha = nn.Parameter(torch.from_numpy(np.outer(alpha, alpha)).float()).to(device)
        tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
        for x, y, u, v in itertools.product(range(8), repeat=4):
            tensor[x, y, u, v] = np.cos((2 * u + 1) * x * np.pi / 16) * np.cos(
                (2 * v + 1) * y * np.pi / 16)
        self.tensor = nn.Parameter(torch.from_numpy(tensor).float()).to(device)

    def forward(self, image):
        
        image = image * self.alpha
        result = 0.25 * torch.tensordot(image, self.tensor, dims=2) + 128
        result.view(image.shape)
        return result


class block_merging(nn.Module):
    """ Merge pathces into image
    Inputs:
        patches(tensor) batch x height*width/64, height x width
        height(int)
        width(int)
    Output:
        image(tensor): batch x height x width
    """
    def __init__(self, device):
        super(block_merging, self).__init__()
        self.device = device
        
    def forward(self, patches, height, width):
        k = 8
        batch_size = patches.shape[0]
        image_reshaped = patches.view(batch_size, height//k, width//k, k, k)
        image_transposed = image_reshaped.permute(0, 1, 3, 2, 4)
        return image_transposed.contiguous().view(batch_size, height, width)


class chroma_upsampling(nn.Module):
    """ Upsample chroma layers
    Input: 
        y(tensor): y channel image
        cb(tensor): cb channel
        cr(tensor): cr channel
    Ouput:
        image(tensor): batch x height x width x 3
    """
    def __init__(self, device):
        super(chroma_upsampling, self).__init__()
        self.device = device

    def forward(self, y, cb, cr):
        def repeat(x, k=2):
            height, width = x.shape[1:3]
            x = x.unsqueeze(-1)
            x = x.repeat(1, 1, k, k)
            x = x.view(-1, height * k, width * k)
            return x

        cb = repeat(cb)
        cr = repeat(cr)
        
        return torch.cat([y.unsqueeze(3), cb.unsqueeze(3), cr.unsqueeze(3)], dim=3)


class ycbcr_to_rgb_jpeg(nn.Module):
    """ Converts YCbCr image to RGB JPEG
    Input:
        image(tensor): batch x height x width x 3
    Outpput:
        result(tensor): batch x 3 x height x width
    """
    def __init__(self, device):
        super(ycbcr_to_rgb_jpeg, self).__init__()
        self.device = device
        matrix = np.array(
            [[1., 0., 1.402], [1, -0.344136, -0.714136], [1, 1.772, 0]],
            dtype=np.float32).T
        self.shift = nn.Parameter(torch.tensor([0, -128., -128.])).to(device)
        self.matrix = nn.Parameter(torch.from_numpy(matrix)).to(device)

    def forward(self, image):
        result = torch.tensordot(image + self.shift, self.matrix, dims=1)
        #result = torch.from_numpy(result)
        result.view(image.shape)
        return result.permute(0, 3, 1, 2)


class decompress_jpeg(nn.Module):
    """ Full JPEG decompression algortihm
    Input:
        compressed(dict(tensor)): batch x h*w/64 x 8 x 8
        rounding(function): rounding function to use
        factor(float): Compression factor
    Ouput:
        image(tensor): batch x 3 x height x width
    """
    def __init__(self, device, height, width, rounding=torch.round, factor=1):
        super(decompress_jpeg, self).__init__()
        self.device=device
        self.c_dequantize = c_dequantize(device, factor=factor)
        self.y_dequantize = y_dequantize(device, factor=factor)
        self.idct = idct_8x8(device)
        self.merging = block_merging(device)
        self.chroma = chroma_upsampling(device)
        self.colors = ycbcr_to_rgb_jpeg(device)
        
        self.height, self.width = height, width
        
    def forward(self, y, cb, cr):
        components = {'y': y, 'cb': cb, 'cr': cr}
        for k in components.keys():
            if k in ('cb', 'cr'):
                comp = self.c_dequantize(components[k])
                height, width = int(self.height/2), int(self.width/2)                
            else:
                comp = self.y_dequantize(components[k])
                height, width = self.height, self.width                
            comp = self.idct(comp)
            components[k] = self.merging(comp, height, width)
            #
        image = self.chroma(components['y'], components['cb'], components['cr'])
        image = self.colors(image)

        image = torch.min(255*torch.ones_like(image),
                          torch.max(torch.zeros_like(image), image))
        return image/255

y_table = np.array(
    [[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60,
                                        55], [14, 13, 16, 24, 40, 57, 69, 56],
     [14, 17, 22, 29, 51, 87, 80, 62], [18, 22, 37, 56, 68, 109, 103,
                                        77], [24, 35, 55, 64, 81, 104, 113, 92],
     [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]],
    dtype=np.float32).T

y_table = nn.Parameter(torch.from_numpy(y_table))
#
c_table = np.empty((8, 8), dtype=np.float32)
c_table.fill(99)
c_table[:4, :4] = np.array([[17, 18, 24, 47], [18, 21, 26, 66],
                            [24, 26, 56, 99], [47, 66, 99, 99]]).T
c_table = nn.Parameter(torch.from_numpy(c_table))


def diff_round(x):
    """ Differentiable rounding function
    Input:
        x(tensor)
    Output:
        x(tensor)
    """
    return torch.round(x) + (x - torch.round(x))**3


def quality_to_factor(quality):
    """ Calculate factor corresponding to quality
    Input:
        quality(float): Quality for jpeg compression
    Output:
        factor(float): Compression factor
    """
    if quality < 50:
        quality = 5000. / quality
    else:
        quality = 200. - quality*2
    return quality / 100.
