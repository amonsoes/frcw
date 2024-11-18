import numpy as np
import torch
import scipy.ndimage as nd


from torchvision import transforms as T
from src.datasets.data_transforms.spatial_transform import BandCMatTransform
from src.adversarial.spectral import SpectrumNorm

class SpectralTransforms:

    def __init__(self, transform, greyscale_opt, adversarial_opt, input_size, dataset_type):

        if transform == 'real_nd_fourier':
            self.transform_train = RealNDFourier(greyscale_opt.greyscale_fourier)

            self.transform_val = RealNDFourier(greyscale_opt.greyscale_fourier)
        
        elif transform == 'augmented_nd_fourier':
            
            self.transform_train = T.Compose([RealNDFourier(greyscale_opt.greyscale_fourier),
                                            SpectralAugmenter(aug_prob=0.9,
                                                                num_features=40,
                                                                greyscale_processing=adversarial_opt.greyscale_processing,
                                                                img_size=input_size,
                                                                path_power_dict=adversarial_opt.power_dict_path, 
                                                                path_delta=adversarial_opt.spectral_delta_path,
                                                                dataset_type=dataset_type)])

            self.transform_val = T.Compose([RealNDFourier(greyscale_opt.greyscale_fourier),
                                            SpectralAugmenter(aug_prob=0.9,
                                                                num_features=40,
                                                                greyscale_processing=adversarial_opt.greyscale_processing,
                                                                img_size=input_size,
                                                                path_power_dict=adversarial_opt.power_dict_path, 
                                                                path_delta=adversarial_opt.spectral_delta_path,
                                                                dataset_type=dataset_type)])

            

        elif transform == 'spectral_band_cooccurrence':
            # to band coocurrence matrix 12 x 256 x 256
            self.transform_train = T.Compose([RealNDFourier(fourier_from_greyscale=False),
                                        BandCMatTransform(),
                                        T.ConvertImageDtype(torch.float32)])

            self.transform_val = T.Compose([RealNDFourier(fourier_from_greyscale=False),
                                        BandCMatTransform(),
                                        T.ConvertImageDtype(torch.float32)])

        elif transform == 'basic_fr_attn_cnn':
            self.transform_train = RealNDFourier(greyscale_opt.greyscale_fourier)

            self.transform_val = RealNDFourier(greyscale_opt.greyscale_fourier)

                                        
class RealNDFourier:

    def __init__(self, fourier_from_greyscale):
        self.to_greyscale = T.Grayscale() if fourier_from_greyscale else self.identity
    
    
    def __call__(self, x):
        return self.dft_magn_normalized(x)

    def dft_magn_normalized(self, img):
        img = self.to_greyscale(img)
        img = img.div(255)
        f = torch.fft.fftn(img)
        fshift = torch.fft.fftshift(f)
        magn = torch.log(fshift.abs()+1e-3)
        fft_min = torch.quantile(magn,0.05)
        fft_max = torch.quantile(magn,0.95)
        magn = (magn-fft_min)/((magn-fft_max)+1e-3)
        magn[magn<-1] = -1
        magn[magn>1] = 1
        return magn


    @staticmethod
    def ff_func(im):
        im = im/255.0
        for i in range(3):
            img = im[:,:,i]
            fft_img = np.fft.fft2(img)
            fft_img = np.log(np.abs(fft_img)+1e-3)
            fft_img = RealNDFourier.fourier_normalize(fft_img)
            #set mid and high freq to 0
            fft_img = np.fft.fftshift(fft_img)
            RealNDFourier.set_mid_high_zero(fft_img)
            fft_img = np.fft.fftshift(fft_img)
        im[:,:,i] = fft_img
        return im
    
    @staticmethod
    def fourier_normalize(fft_img):
        fft_min = np.percentile(fft_img,5)
        fft_max = np.percentile(fft_img,95)
        fft_img = (fft_img - fft_min)/(fft_max - fft_min)
        fft_img = (fft_img-0.5)*2
        fft_img[fft_img<-1] = -1
        fft_img[fft_img>1] = 1
        return fft_img

    @staticmethod
    def set_mid_high_zero(fft_img):
            fft_img[:21, :] = 0
            fft_img[:, :21] = 0
            fft_img[203:, :] = 0
            fft_img[:, 203:] = 0
            fft_img[57:177, 57:177] = 0
            return fft_img
    
    def identity(self, x):
        return x

class SpectralAugmenter:
    
    def __init__(self, aug_prob, *args, **kwargs):
        self.spectrum_norm = SpectrumNorm(is_adv=False, *args, **kwargs)
        self.aug_prob = aug_prob
    
    def __call__(self, x_f):
        if np.random.choice([0,1], p=[1-self.aug_prob, self.aug_prob]):
            x_f = self.spectrum_norm(x_f)
        return x_f
        
class LaplacianOfGaussian:

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, img):
        return torch.tensor(nd.gaussian_laplace(img, self.sigma))


class InverseLaplacianOfGaussian:

    def __init__(self, variance):
        self.variance = variance


    def inv_log(self, x):
        pass
    """
    def inv_log(self, x):
        return (-((self.variance*x**2)/(2*math.pi)))*math.e**(0.5*-((self.variance*x)**2))"""


if __name__ == '__main__':
    pass