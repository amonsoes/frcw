import torch
import numpy as np
import torchvision.transforms as T

from src.adversarial.jpeg_quantization import RGBConversion, UnquantizedJPEGConversion
from src.adversarial.jpeg_ifgm import c_dequantize, y_dequantize, quality_to_factor
from src.adversarial.attack_base import Attack
from src.adversarial.jpeg_ifgm import DiffJPEG


#######################################
#
##  Fast Adversarial Rounding
#
######################################


def convert_eta(eta_in):
    return int((1. - eta_in) * 100)


class FastAdversarialRounding(Attack):
    """
    This class implements the fast adversarial rounding from
    Shi et al.: On generating JPEG adversarial images, 2021.
    """

    def __init__(self, 
                model, 
                model_trms, 
                eta, 
                far_jpeg_quality,
                surrogate_loss,
                chroma_subsampling=True,
                eps=0.04,
                target_mode='default',
                img_size=224,
                *args, 
                **kwargs):
        super().__init__("FastAdvRounding", model, model_trms, *args, **kwargs)
        self.supported_mode = ['default', 'targeted']
        if target_mode != 'default':
            self.set_target_mode(target_mode)
        self.eta = eta
        self.jpeg_quality = far_jpeg_quality
        factor = quality_to_factor(far_jpeg_quality)
        self.loss = surrogate_loss
        self.chroma_subsampling = chroma_subsampling
        self.dequantize_y = y_dequantize(self.device, factor=factor)
        self.dequantize_c = c_dequantize(self.device, factor=factor)
        self.img_size = img_size
        
        # for fgsm
        self.eps = eps
        self.normalize_to_float = T.ConvertImageDtype(torch.float32)
    

        # load a model that converts JPEG to RGB data
        self.rgb_to_jpeg_model = UnquantizedJPEGConversion(self.device, rounding=torch.round, factor=factor)
        # load a model that converts RGB to unquantized (intermediate) JPEG data
        self.jpeg_to_rgb_model = RGBConversion(self.device, img_size, img_size, rounding=torch.round, factor=factor)

    def __call__(self, images, labels):
        """
        1) adv_images_jpeg -> get quantized DCT patch repr.
        Eq. 3
        2) adv rounding
        3) Convert back to RGB
        """
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        # Make Adv with FGSM
        adv_images_rgb = self.fgsm(images, labels)
        adv_images_jpeg = self.rgb_to_jpeg_model(adv_images_rgb)
        
        # get target laben if targeted
        if self.targeted:
            labels = self.get_target_label(images, labels)
        
        # perform FAR
        y_rounded, cb_rounded, cr_rounded = self._fast_adversarial_rounding(adv_images_jpeg, labels)
        adv_images = self.jpeg_to_rgb_model(y_rounded,cb_rounded, cr_rounded)
        adv_images = torch.clip(adv_images, min=0., max=1.).detach()
        if self.targeted:
            return adv_images, labels
        else:
            return adv_images

    def _first_rounding_step(self, jpeg_adversarial_unrounded, Y_grad, Cb_grad, Cr_grad):
        """
        Only changes DCT coeffs to X_nearest coeffs where either:
            1.) the coeff was uprounded and the grad is positive or
            2.) the coeff was downr and the grad is negative.
            -> "the rounding is consistent with the sign of the gradient"
        See Section 3 in Paper

        """
        Y, Cb, Cr = jpeg_adversarial_unrounded
        Y_nearest, Cb_nearest, Cr_nearest = torch.round(Y), torch.round(Cb), torch.round(Cr)

        upr_and_grad_positive = torch.logical_and(Y_nearest >= Y, Y_grad >= 0.)
        downr_and_grad_negative = torch.logical_and(Y_nearest <= Y, Y_grad <= 0.)
        
        Y_first_step_rounding = torch.where(torch.logical_or(upr_and_grad_positive, downr_and_grad_negative),Y_nearest, Y)

        # do same for Cb but less verbose
        Cb_first_step_rounding = torch.where(
            torch.logical_or(
                torch.logical_and(Cb_nearest >= Cb, Cb_grad >= 0.),
                torch.logical_and(Cb_nearest <= Cb, Cb_grad <= 0.)
            ),
            Cb_nearest, Cb
        )

        # do same for Cr but less verbose
        Cr_first_step_rounding = torch.where(
            torch.logical_or(
                torch.logical_and(Cr_nearest >= Cr, Cr_grad >= 0.),
                torch.logical_and(Cr_nearest <= Cr, Cr_grad <= 0.)
            ),
            Cr_nearest, Cr
        )

        return Y_first_step_rounding, Cb_first_step_rounding, Cr_first_step_rounding

    def _fast_adversarial_rounding_gradients(self, jpeg_adversarial, labels):
        """
        Computes the gradient of the loss function wrt. the quantized DCT repr D^u.
        Eq. 5
        """
        
        jpeg_adversarial = [x.clone().detach().requires_grad_(True) for x in jpeg_adversarial]
        #jpeg_adversarial = jpeg_adversarial.clone().detach()
        #jpeg_adversarial.requires_grad = True
        loss = self.loss_from_image(jpeg_adversarial, labels) # in der ursprünglichen loss fn ist sicher mehr passiert - im repo nachschauen
        loss.backward()
        

        Y_grad = jpeg_adversarial[0].grad
        Cb_grad = jpeg_adversarial[1].grad
        Cr_grad = jpeg_adversarial[2].grad

        return Y_grad, Cb_grad, Cr_grad

    def _second_rounding_step(self, Y_first_step_rounding, Cb_first_step_rounding, Cr_first_step_rounding, Y_grad, Cb_grad, Cr_grad):
        """
        D_u -> Y_first_step_rounding, Cb_first_step_rounding, Cr_first_step_rounding
        D_u is now the updated coefficients, where from the first rounding step some coeffs have been rounded. 
        The others were left unrounded. Now we round the intermediate coeffs based on Eq. 6.
        The ceil and floor operations are applied based on the gradient.
        
        First, the cutoff-percentile tau is calculated based on Eq. 7.
        
        """
        Y_d_diff = (torch.ceil(Y_first_step_rounding) - Y_first_step_rounding) - (Y_first_step_rounding - torch.floor(Y_first_step_rounding))
        Cb_d_diff = (torch.ceil(Cb_first_step_rounding) - Cb_first_step_rounding) - (Cb_first_step_rounding - torch.floor(Cb_first_step_rounding))
        Cr_d_diff = (torch.ceil(Cr_first_step_rounding) - Cr_first_step_rounding) - (Cr_first_step_rounding - torch.floor(Cr_first_step_rounding))

        dequantized_differences_y = self.dequantize_y(Y_d_diff)
        dequantized_differences_cb = self.dequantize_c(Cb_d_diff)
        dequantized_differences_cr = self.dequantize_c(Cr_d_diff)

        Y_tau = torch.abs(Y_grad) / (dequantized_differences_y ** 2)
        Cb_tau = torch.abs(Cb_grad) / (dequantized_differences_cb ** 2)
        Cr_tau = torch.abs(Cr_grad) / (dequantized_differences_cr ** 2)

        eta_percentile = convert_eta(self.eta)
        Y_percentile = np.percentile(Y_tau.cpu().detach().numpy(), eta_percentile, axis=(1, 2, 3), keepdims=True)
        Cb_percentile = np.percentile(Cb_tau.cpu().detach().numpy(), eta_percentile, axis=(1, 2, 3), keepdims=True)
        Cr_percentile = np.percentile(Cr_tau.cpu().detach().numpy(), eta_percentile, axis=(1, 2, 3), keepdims=True)

        Y_percentile = torch.tensor(Y_percentile, device=Y_tau.device)
        Cb_percentile = torch.tensor(Cb_percentile, device=Cb_tau.device)
        Cr_percentile = torch.tensor(Cr_percentile, device=Cr_tau.device)

        Y_second_rounding_step = torch.where(
            Y_tau > Y_percentile,
            torch.where(Y_grad >= 0, torch.ceil(Y_first_step_rounding), torch.floor(Y_first_step_rounding)),
            torch.round(Y_first_step_rounding)
        )

        Cb_second_rounding_step = torch.where(
            Cb_tau > Cb_percentile,
            torch.where(Cb_grad >= 0, torch.ceil(Cb_first_step_rounding), torch.floor(Cb_first_step_rounding)),
            torch.round(Cb_first_step_rounding)
        )

        Cr_second_rounding_step = torch.where(
            Cr_tau > Cr_percentile,
            torch.where(Cr_grad >= 0, torch.ceil(Cr_first_step_rounding), torch.floor(Cr_first_step_rounding)),
            torch.round(Cr_first_step_rounding)
        )

        return Y_second_rounding_step, Cb_second_rounding_step, Cr_second_rounding_step

    def _fast_adversarial_rounding(self, jpeg_adversarial_unrounded, labels):
        Y_grad, Cb_grad, Cr_grad = self._fast_adversarial_rounding_gradients(jpeg_adversarial_unrounded, labels)

        Y_first_step_rounding, Cb_first_step_rounding, Cr_first_step_rounding = self._first_rounding_step(
            jpeg_adversarial_unrounded, Y_grad, Cb_grad, Cr_grad
        )


        first_step_rounding_images = (Y_first_step_rounding, Cb_first_step_rounding, Cr_first_step_rounding)
        Y_grad, Cb_grad, Cr_grad = self._fast_adversarial_rounding_gradients(first_step_rounding_images, labels)
        return self._second_rounding_step(
            Y_first_step_rounding, Cb_first_step_rounding, Cr_first_step_rounding, Y_grad, Cb_grad, Cr_grad
        )
    
    def loss_from_image(self, images, labels):
        rgb_images = self.jpeg_to_rgb_model(*images)
        outputs = self.get_logits(self.model_trms(rgb_images))
        if self.targeted:
            cost = -self.loss(outputs, labels)
        else:
            cost = self.loss(outputs, labels)
        return cost
        
        
    def fgsm(self, images, labels):
        
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        
        if self.targeted:
            target_labels = self.get_target_label(images, labels)
        images = self.normalize_to_float(images)
        images.requires_grad = True
        

        outputs = self.get_logits(self.model_trms(images))

        # Calculate loss
        if self.targeted:
            cost = -self.loss(outputs, target_labels)
        else:
            cost = self.loss(outputs, labels)

        # Update adversarial images
        grad = torch.autograd.grad(cost, images,
                                   retain_graph=False, create_graph=False)[0]

        adv_images = images + self.eps*grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        
        return adv_images * 255

    def to_jpeg(self, image, quality):
        with torch.no_grad():
            #factor = quality_to_factor(quality)
            
            compress_decompress = DiffJPEG(height=self.img_size, width=self.img_size, differentiable=False, quality=quality, device='cpu')
            #jpeg_to_rgb_model = RGBConversion(self.device, self.img_size, self.img_size, rounding=torch.round, factor=factor)
            #quantized_r2j = QuantizedJPEGConversion(self.device, rounding=torch.round, factor=factor)
            
            #jpeg_image = quantized_r2j(image)
            #rgb_image = jpeg_to_rgb_model(jpeg_image)
            
            adjusted_image = compress_decompress(image)
            return adjusted_image

if __name__ == '__main__':
    pass