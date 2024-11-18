import torch
import scipy.ndimage as nd
import torch_dct as dct
import torchvision.transforms as T

class HPFMasker:
    
    def __init__(self,
                device,
                input_size,
                use_sal_mask=True, 
                hpf_mask_tau=0.7, 
                sal_mask_only=False, 
                diagonal=-4, 
                log_sigma=3.0, 
                dct_gauss_ksize=13, 
                log_mu=0.6,  
                lf_boosting=0.0,
                targeted=False,
                is_black_box=False,
                dct_patch_size=8):
        
        if not use_sal_mask and sal_mask_only:
            raise ValueError('You cannot cannot disable the saliency mask and use the saliency mask only at the same time.')
        self.device = device
        self.dct = DCT(img_size=input_size, patch_size=dct_patch_size, n_channels=3, diagonal=diagonal)
        self.log = LaplacianOfGaussian(log_sigma)
        self.gaussian = T.GaussianBlur(kernel_size=dct_gauss_ksize)
        self.to_grey = T.Grayscale()
        self.lf_boosting = lf_boosting
        self.use_sal_mask = use_sal_mask
        self.sal_mask_only = sal_mask_only
        self.targeted = targeted
        self.hpf_mask = None
        self.is_black_box = is_black_box
        self.convert_to_float = T.ConvertImageDtype(torch.float32)
        if input_size == 299:
            self.resize = T.Resize(input_size) # for 299 x 299 img sizes
            self.get_log_and_dct_mask_fn = self.get_log_and_dct_mask_for_299
        else:
            self.get_log_and_dct_mask_fn = self.get_log_and_dct_mask
        if not is_black_box:
            self.saliency_mask = None
    
    def __call__(self, images, labels=None, model=None, model_trms=None, loss=None):

        self.hpf_mask = None
        self.saliency_mask = None
        images = images.squeeze(0)
        ori_images = images.clone().detach()
        norm_images = images.clone().detach()
        norm_images = self.normalize_for_dct(norm_images)
        
        if not self.sal_mask_only:
            hpf_mask = self.get_log_and_dct_mask_fn(ori_images, norm_images)
            self.hpf_mask = hpf_mask
        
        if (self.use_sal_mask or self.sal_mask_only) and not self.is_black_box:
        
            inputs = self.normalize(images.clone().detach())
            saliency_mask = self.get_saliency_mask(inputs, labels, model, model_trms, loss)
            self.saliency_mask = saliency_mask
            
            if self.sal_mask_only:
                attack_mask = saliency_mask
            else:
                attack_mask = torch.clamp(hpf_mask+saliency_mask, min=0., max=1.)
        else:
            attack_mask = hpf_mask
        
        return attack_mask

    def compute_for_hpf_mask(self, images):
    
        self.hpf_mask = None
        self.saliency_mask = None
        
        ori_images = images.clone().detach()
        norm_images = images.clone().detach()
        norm_images = self.normalize_for_dct(norm_images)
        
        hpf_mask = self.get_log_and_dct_mask_fn(ori_images, norm_images)
        self.hpf_mask = hpf_mask
        
        attack_mask = hpf_mask
        return attack_mask


    def compute_with_grad(self, images, grad):
        """
        compute mask with grad supplied
        """
        
    
        self.hpf_mask = None
        self.saliency_mask = None
        
        ori_images = images.clone().detach()
        norm_images = images.clone().detach()
        norm_images = self.normalize_for_dct(norm_images)
        
        if not self.sal_mask_only:
            hpf_mask = self.get_log_and_dct_mask_fn(ori_images, norm_images)
            self.hpf_mask = hpf_mask
        
        if (self.use_sal_mask or self.sal_mask_only) and not self.is_black_box:
        
            inputs = self.normalize(images.clone().detach())
            saliency_mask = self.get_saliency_mask_from_grad(inputs, grad)
            self.saliency_mask = saliency_mask
            
            if self.sal_mask_only:
                attack_mask = saliency_mask
            else:
                attack_mask = torch.clamp(hpf_mask+saliency_mask, min=0., max=1.)
        else:
            attack_mask = hpf_mask
        
        return attack_mask
    
    def get_log_and_dct_mask(self, ori_images, norm_images):
        """Builds DCT and LoG mask and defines a tradeoff by the
        log param. Additionally, does LF boosting if wanted.

        Args:
            ori_images: input in range [0, 255]
            norm_images: input in range [-1, 1] for DCT computation
        
        Returns:
            hpf_mask : mask containing LoG and DCT tradeoff coefficients [0, 1]
        """
        img_shape = ori_images.shape
        log_masks = torch.ones(img_shape)
        dct_coeff_masks = torch.ones(img_shape)
        
        for e, (img, norm_img) in enumerate(zip(ori_images, norm_images)):
            log_mask = self.log(img.cpu())
            if log_mask.min().item() < 0:
                log_mask = log_mask + (-1*log_mask.min().item())
            log_mask = log_mask / log_mask.max().item()
            log_masks[e] = log_mask
            dct_coeffs, fgsm_coeffs = self.dct(norm_img)
            dct_coeff_masks[e] = fgsm_coeffs
        
        log_masks = log_masks.to(self.device)
        dct_coeff_masks = dct_coeff_masks.to(self.device)
        #hpf_mask = (log_masks*self.log_mu)+(dct_coeff_masks*self.dct_mu)
        hpf_mask = torch.clamp(log_masks+dct_coeff_masks, min=0., max=1.)

        if self.lf_boosting > 0.0:
            inv_mask = 1 - hpf_mask
            hpf_mask = (self.lf_boosting*inv_mask) + hpf_mask
        
        return hpf_mask

    def get_log_and_dct_mask_for_299(self, ori_images, norm_images):
        """Builds DCT and LoG mask and defines a tradeoff by the
        log param. Additionally, does LF boosting if wanted.

        Args:
            ori_images: input in range [0, 255]
            norm_images: input in range [-1, 1] for DCT computation
        
        Returns:
            hpf_mask : mask containing LoG and DCT tradeoff coefficients [0, 1]
        """
        img_shape = ori_images.shape
        log_masks = torch.ones(img_shape)
        dct_coeff_masks = torch.ones(img_shape)
        
        for e, (img, norm_img) in enumerate(zip(ori_images, norm_images)):
            log_mask = self.log(img.cpu())
            if log_mask.min().item() < 0:
                log_mask = log_mask + (-1*log_mask.min().item())
            log_mask = log_mask / log_mask.max().item()
            log_masks[e] = log_mask
            dct_coeffs, fgsm_coeffs = self.dct(norm_img)
            dct_coeff_masks[e] = self.resize(fgsm_coeffs)
        
        log_masks = log_masks.to(self.device)
        dct_coeff_masks = dct_coeff_masks.to(self.device)
        #hpf_mask = (log_masks*self.log_mu)+(dct_coeff_masks*self.dct_mu)
        hpf_mask = torch.clamp(log_masks+dct_coeff_masks, min=0., max=1.)

        if self.lf_boosting > 0.0:
            inv_mask = 1 - hpf_mask
            hpf_mask = (self.lf_boosting*inv_mask) + hpf_mask
        
        return hpf_mask

    def get_saliency_mask(self, adv_images, labels, model, model_trms, loss, target_labels=None):
        """
        
        This generates a coefficient mask based on the gradient intensities
        of the grad of the surrogate model. This can be thought of as a 
        coefficient representation of a saliency map. The coefficients should be high
        in areas of gradient importance and less so in other areas.

        Args:
            img : input of attack [0, 1]
        
        Returns:
            saliency_mask : coefficients of salient regions [0, 1]
        """
        
        inputs = adv_images.clone().detach()
        inputs.requires_grad = True

        outputs = model(model_trms(inputs))
        
        if self.targeted:
            cost = -loss(outputs, target_labels)
        else:
            cost = loss(outputs, labels)

        grad = torch.autograd.grad(cost, inputs,
                                retain_graph=False, create_graph=False)[0]
            
        abs_grad_vals = grad.abs()
        saliency_mask = abs_grad_vals / abs_grad_vals.max()
        model.zero_grad()
        return saliency_mask

    def get_saliency_from_grad(self, grad):
        abs_grad_vals = grad.abs()
        saliency_mask = abs_grad_vals / abs_grad_vals.max()
        return saliency_mask
    
    def update_saliency_mask(self, grad):
        abs_grad_vals = grad.abs()
        self.saliency_mask = abs_grad_vals / abs_grad_vals.max()
        if self.sal_mask_only:
            attack_mask = self.saliency_mask
        else:
            attack_mask = torch.clamp(self.hpf_mask+self.saliency_mask, min=0., max=1.)
        return attack_mask
        
    def normalize_for_dct(self, imgs):
        imgs = imgs.to(torch.float32) - 128
        imgs = imgs / 128
        return imgs
    
    def normalize(self, imgs):
        imgs = self.convert_to_float(imgs)
        return imgs


class Patchify:
    
    def __init__(self, img_size, patch_size, n_channels):
        self.img_size = img_size
        self.n_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size

        #assert (img_size // patch_size) * patch_size == img_size
        
    def __call__(self, x):
        p = x.unfold(1, self.patch_size, self.patch_size).unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size) # x.size() -> (batch, model_dim, n_patches, n_patches)
        self.unfold_shape = p.size()
        p = p.contiguous().view(-1,self.patch_size,self.patch_size)
        return p
    
    def inverse(self, p):
        if not hasattr(self, 'unfold_shape'):
            raise AttributeError('Patchify needs to be applied to a tensor in ordfer to revert the process.')
        x = p.view(self.unfold_shape)
        output_h = self.unfold_shape[1] * self.unfold_shape[4]
        output_w = self.unfold_shape[2] * self.unfold_shape[5]
        x = x.permute(0,1,4,2,5,3).contiguous()
        x = x.view(3, output_h, output_w)
        return x 

class DCT:
    
    def __init__(self, img_size=224, patch_size=8, n_channels=3, diagonal=0):
        """
        diagonal parameter will decide how much cosine components will be taken into consideration
        while calculating fgsm patches.
        """
        print('DCT class transforms on 3d tensors')
        self.patchify = Patchify(img_size=img_size, patch_size=patch_size, n_channels=n_channels)
        self.normalize = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        self.mask = torch.flip(torch.triu(torch.ones((patch_size,patch_size)), diagonal=diagonal), dims=[0])
        self.patch_size = patch_size
        self.n = 0
        self.tile_mean = 0
        
    def __call__(self, tensor):
        p, fgsm_coeffs = self.patched_dct(tensor)
        dct_coeffs = self.patchify.inverse(p)
        fgsm_coeffs = self.patchify.inverse(fgsm_coeffs)
        fgsm_coeffs = fgsm_coeffs / fgsm_coeffs.max()
        return dct_coeffs, fgsm_coeffs
    
    def patched_dct(self, tensor):
        p = self.patchify(tensor)
        fgsm_coeff_tensor = torch.zeros(p.shape, dtype=torch.float32)
        for e, patch in enumerate(p):
            
            dct_coeffs = dct.dct_2d(patch, norm='ortho')
            dct_coeffs[0][0] = 0.0
            fgsm_coeffs = self.calculate_fgsm_coeffs(dct_coeffs)
            fgsm_coeff_tensor[e] = fgsm_coeffs
            p[e] = dct_coeffs
        
        self.tile_mean += fgsm_coeff_tensor.mean()
        self.n += 1
        return p, fgsm_coeff_tensor
    
    def calculate_fgsm_coeffs(self, patch):
        masked_patch = patch[self.mask == 1].abs()
        sum_patch = sum(masked_patch)
        return torch.full((self.patch_size,self.patch_size), fill_value=sum_patch.item())

class LaplacianOfGaussian:
    
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, img):
        return torch.tensor(nd.gaussian_laplace(img, self.sigma))
    
if __name__ == '__main__':
    pass