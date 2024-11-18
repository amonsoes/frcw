import torch
from src.adversarial.jpeg_ifgm import compress_jpeg, decompress_jpeg
from src.adversarial.jpeg_ifgm import c_quantize_no_rounding, y_quantize_no_rounding

############################
#
# ToJPEGModel
#
############################


class UnquantizedJPEGConversion(compress_jpeg):
    
    def __init__(self, device, rounding=torch.round, factor=1):
        super().__init__( device, rounding, factor)
        self.y_quantize = y_quantize_no_rounding(self.device, factor=self.jpeg_quality)
        self.c_quantize = c_quantize_no_rounding(self.device, factor=self.jpeg_quality)

    def forward(self, image):
        """
        Input: BxCxHxW RGB Image
        Quantize DCT's but do not round (See Paragraph 3. Proposed Method Paper)
        Output: Y,Cb,Cr channels of shape batch x h*w/64 x 8 x 8
        
        """

        y, cb, cr = self.l1(image)
        components = {'y': y, 'cb': cb, 'cr': cr}
        for k in components.keys():
            comp = self.l2(components[k])
            if k in ('cb', 'cr'):
                comp = self.c_quantize(comp)
            else:
                comp = self.y_quantize(comp)
            components[k] = comp

        return components['y'], components['cb'], components['cr']

############################
#
## ToRGBModel
#
############################

class RGBConversion(decompress_jpeg):

    def __init__(self, device, height, width, rounding=torch.round, factor=1):
        """ Full JPEG decompression algortihm
        Input:
            compressed(dict(tensor)): batch x h*w/64 x 8 x 8
            rounding(function): rounding function to use
            factor(float): Compression factor
        Ouput:
            image(tensor): batch x 3 x height x width
        """
        super().__init__(device, height, width, rounding, factor)
        
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
    

############################
#
## COLOR CONVERSION MODELS
#
############################

def round_or_approx(x, round):
    """
    :param x:
    :param round:
    :return:
    """
    if round == 'round':
        return torch.round(x)
    elif round == 'shin':
        return shin_rounding_approximation(x)
    elif round is None:
        return x

    else:
        raise ValueError(f'Rounding scheme {round} unknown')


def shin_rounding_approximation(x):
    """
    See Shin & Song, 2017:
    :param x: input tensor
    :return: tensor with Shin rounding approximation applied
    """
    return torch.round(x) + torch.pow(x - torch.round(x), 3)


