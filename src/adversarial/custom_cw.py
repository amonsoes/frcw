####################################################
# original CW class is transferred from the torchattacks library
# https://adversarial-attacks-pytorch.readthedocs.io/en/latest/
#
# Adjusted by A. S. (anonymized for review)
# All other classes by A. S. (anonymized for review)
####################################################

import torch
import torch.nn as nn
import torch.optim as optim
import csv
import random

from kornia.color import ycbcr_to_rgb, rgb_to_ycbcr
from IQA_pytorch import SSIM, MAD
from DISTS_pytorch import DISTS
from torchmetrics.image import VisualInformationFidelity as VIF
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure as MSSSIM
from diff_jpeg import diff_jpeg_coding
from torchvision import transforms as T

from src.adversarial.attack_base import Attack
from src.adversarial.iqm import PSNR, HPFL2



class CW(Attack):
    r"""
    CW in the paper 'Towards Evaluating the Robustness of Neural Networks'
    [https://arxiv.org/abs/1608.04644]

    Distance Measure : L2

    Arguments:
        model (nn.Module): model to attack.
        c (float): c in the paper. parameter for box-constraint. (Default: 1)    
            :math:`minimize \Vert\frac{1}{2}(tanh(w)+1)-x\Vert^2_2+c\cdot f(\frac{1}{2}(tanh(w)+1))`
        kappa (float): kappa (also written as 'confidence') in the paper. (Default: 0)
            :math:`f(x')=max(max\{Z(x')_i:i\neq t\} -Z(x')_t, - \kappa)`
        steps (int): number of steps. (Default: 50)
        lr (float): learning rate of the Adam optimizer. (Default: 0.01)

    .. warning:: With default c, you can't easily get adversarial images. Set higher c like 1.

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.CW(model, c=1, kappa=0, steps=50, lr=0.01)
        >>> adv_images = attack(images, labels)


    """
    def __init__(self, 
                model, 
                model_trms, 
                batch_size,
                c=1, 
                kappa=0, 
                steps=10000, 
                attack_lr=0.01, 
                write_protocol=False,
                protocol_file=False,
                n_starts=1,
                verbose_cw=False,
                target_mode='random',
                eps=0.04):
        super().__init__("CW", model, model_trms)
        self.c = c
        self.c_value = c
        self.original_c = c
        self.kappa = kappa
        self.steps = steps
        self.lr = attack_lr
        self.supported_mode = ['default', 'targeted']
        self.loss = nn.MSELoss(reduction='none') # will be used to get L2 dist in loss fn
        self.flatten = nn.Flatten()
        self.write_protocol = write_protocol
        self.protocol_file = protocol_file
        if self.protocol_file:
            self.protocol_dir = "/".join(self.protocol_file.split('/')[:-2]) + '/'
        self.eps = 0.001
        self.n_samples = n_starts
        self.use_attack_mask = False
        self.n = 0 # used for calculation cumulative avg of adjustment runtimes
        self.ca_runtime = 0 # cumulative average at n
        self.verbose_cw = verbose_cw
        self.forward_fn = self.forward_verbose if verbose_cw else self.forward_regular
        self.set_target_mode(target_mode)
        self.eps = eps # this is needed for a maximum perturbation estimation
    
    def set_target_mode(self, mode):
        if mode == 'least_likely':
            self.set_mode_targeted_least_likely()
        elif mode == 'most_likely':
            self.set_mode_targeted_most_likely()
        else:
            print('WARNING: set_target_mode was set to random. If unwanted, change "target_mode" arg to either "least_likely" or "most_likely".')
            self.set_mode_targeted_random()

    def get_l2_loss(self, adv_images, images, attack_mask):
        #current_iq_loss = self.loss(self.flatten(adv_images,), self.flatten(images)).sum(dim=1)
        current_iq_loss = (adv_images - images).pow(2).sum(dim=(1,2,3)).sqrt()
        iq_loss = current_iq_loss.sum()
        return iq_loss, current_iq_loss
        
    def forward(self, images, labels):
        best_adv_images_from_starts = self.forward_fn(images, labels)
        return best_adv_images_from_starts
        
    def forward_regular(self, images, labels):
        r"""
        Overridden.
        """
        raise ValueError('Regular forward for CW currently not functional.')

    def forward_verbose(self, images, labels):
        r"""
        Overridden.
        """
        print(f'\nCalculating adversarial example for images...\n\n')
        #images = images / 255
        
        self.c = self.original_c
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        
        if self.use_attack_mask:
            attack_mask = self.hpf_masker(images, 
                                        labels, 
                                        model=self.model, 
                                        model_trms=self.model_trms, 
                                        loss=self.surr_model_loss,
                                        invert_mask=True)
        else:
            attack_mask = 1 # multiplication results in identity

        # implement outer step like in perccw

        # set the lower and upper bounds accordingly
        batch_size = images.shape[0]
        lower_bound = torch.zeros(batch_size, device=self.device)
        CONST = torch.full((batch_size,), self.c_value, device=self.device)
        upper_bound = torch.full((batch_size,), 1e10, device=self.device)
        self.batch_size = batch_size
        
        # set markers for inter-c-run comparison
        best_adv_images_from_starts = images.clone().detach()
        best_iq_from_starts = torch.full_like(CONST, 1e10)
        adv_found = torch.zeros_like(CONST)
        
        for outer_step in range(self.n_samples):    
        # search to get optimal c
            print(f'step in binary search for constant c NR:{outer_step}, c:{self.c}\n')
        
            # w = torch.zeros_like(images).detach() # Requires 2x times    
            w = self.inverse_tanh_space(images).detach()
            w.requires_grad = True

            best_adv_images = images.clone().detach()
            best_iq = 1e10*torch.ones((len(images))).to(self.device)
            best_cost = 1e10*torch.ones((len(images))).to(self.device)
            dim = len(images.shape)

            self.optimizer = optim.Adam([w], lr=self.lr)
            self.optimizer.zero_grad()

            for step in range(self.steps):
                    
                # Get adversarial images
                adv_images = self.tanh_space(w)

                # Calculate image quality loss
                iq_loss, current_iq_loss = self.get_l2_loss(adv_images, images, attack_mask)

                # Calculate adversarial loss including robustness loss
                f_loss, comp_outputs = self.get_adversarial_loss(adv_images, labels, target_labels)

                cost = (current_iq_loss + f_loss).sum() # f_loss holds c tradeoff computation
                
                # Update adversarial images
                self.optimizer.zero_grad()
                cost.backward()
                self.optimizer.step()

                # filter out images that get either correct predictions or non-decreasing loss, 
                # i.e., only images that are not target and loss-decreasing are left 

                # check if adv_images were classified incorrectly
                _, pre = torch.max(comp_outputs.detach(), 1)
                is_adversarial = (pre == target_labels).float()  
                
                # check if the overall los          
                has_lower_iq_loss  = best_iq > current_iq_loss.detach()
                
                # filter to only change images where both if True
                mask =  is_adversarial * has_lower_iq_loss

                best_cost = mask*cost.detach() + (1-mask)*best_cost
                best_iq = mask*current_iq_loss.detach() + (1-mask)*best_iq
                
                mask = mask.view([-1]+[1]*(dim-1))
                best_adv_images = mask*adv_images.detach() + (1-mask)*best_adv_images
                # either the current adv_images is the new best_adv_images or the old one depending on mask
                
                print(f'\n{step} - iq_loss: {iq_loss.item()}, r_loss: {f_loss.sum().item()} cost: {cost.item()}')  

            # set the best output of run as best adv if (1) an adv was found (2) the cost is lower than the one from the last starts
            # compute any per batch image. igs could only be different if one output was adv during optim
            r = torch.any((best_adv_images != images), dim=1)
            r = torch.any(r, dim=1)
            adv_found_in_run = torch.any(r, dim=1)
            
            is_lower_than_all_starts = (best_iq_from_starts > best_iq).float()
            mask = adv_found_in_run * is_lower_than_all_starts
            best_iq_from_starts = mask*best_iq + (1-mask)*best_iq_from_starts
            mask = mask.view([-1]+[1]*(dim-1))
            best_adv_images_from_starts = mask*best_adv_images + (1-mask)*best_adv_images_from_starts

            # adjust the constant as needed
            adv_found = adv_found_in_run
            upper_bound[adv_found] = torch.min(upper_bound[adv_found], CONST[adv_found])
            adv_not_found = ~adv_found
            lower_bound[adv_not_found] = torch.max(lower_bound[adv_not_found], CONST[adv_not_found])
            is_smaller = upper_bound < 1e9
            CONST[is_smaller] = (lower_bound[is_smaller] + upper_bound[is_smaller]) / 2
            CONST[(~is_smaller) * adv_not_found] *= 10
            self.c = CONST
        

        if self.protocol_file:
            self.write_to_protocol_dir(iq_loss, f_loss, cost)
            self.write_runtime_to_protocol_dir()
        return (best_adv_images, target_labels)
    
    def get_adversarial_loss(self, adv_images, labels, target_labels):
        outputs = self.get_outputs(adv_images)
        if self.targeted:
            f_loss = self.f(outputs, target_labels)
            f_loss *= self.c
        else:
            f_loss = self.f(outputs, labels).sum()
            f_loss *= self.c
        return f_loss, outputs

    def get_outputs(self, adv_images):
        return self.get_logits(self.model_trms(adv_images))

    def get_rand_points_around_sample(self, sample_list):
        sample = sample_list[0]
        for _ in range(self.n_samples-1):
            neighbor_image = torch.clamp((sample.detach() + torch.randn_like(sample).uniform_(-self.eps, self.eps)), min=0.0, max=1.0).to(self.device)
            neighbor_image.requires_grad = True
            sample_list.append(neighbor_image)
        return sample_list
    
    def write_to_protocol_dir(self, iq_loss, f_loss, cost):
            with open(self.protocol_file, 'a') as report_file:
                report_obj = csv.writer(report_file)
                report_obj.writerow(['1', str(iq_loss.item()), str(f_loss.item()), str(cost.item())])
    
    def write_runtime_to_protocol_dir(self):
        with open(self.protocol_dir + 'runtimes.csv', 'a') as runtimes_file:
            runtimes_obj = csv.writer(runtimes_file)
            runtimes_obj.writerow([self.ca_runtime])
    
    # this should be overwritten by subclass custom CW
    def get_iq_loss(self, adv_images, images, attack_mask):
        current_iq_loss = self.loss(self.flatten(adv_images), self.flatten(images)).sum(dim=1)
        iq_loss = current_iq_loss.sum()
        return iq_loss, current_iq_loss

    def tanh_space(self, x):
        return 1/2*(torch.tanh(x) + 1)

    def inverse_tanh_space(self, x):
        # torch.atanh is only for torch >= 1.7.0
        return self.atanh(x*2-1)

    def atanh(self, x):
        return 0.5*torch.log((1+x)/(1-x))

    # f-function in the paper
    def f(self, outputs, labels):
        one_hot_labels = torch.eye(len(outputs[0]), device=self.device)[labels].to(self.device)

        i, _ = torch.max((1-one_hot_labels)*outputs, dim=1) # get the largest logit
        j = torch.masked_select(outputs, one_hot_labels.bool()) # get the target logit

        if self.targeted:
            return torch.clamp(((i - j) + self.kappa), min=0) # to mimic perccw's inclusion of kappa
            #return torch.clamp((i-j), min=-self.kappa)
        else:
            return torch.clamp((j-i), min=-self.kappa)
        
    def bin_search_c(self, images, labels, low=1e-01, high=5, steps=30, iters=20):
        print(f'\ninitialize binary search for c in range {low} to {high} in steps {steps}.\n')
        # get random starting point c
        c_values = torch.linspace(low, high, steps)
        best_c = c_values[0]
        best_loss = 9e+04
        for i in range(iters):
            print(f'\n...initializing iteration {i+1}...\n')
            self.c = c_values[i]
            best_adv_img = self(images, labels)
            if self.loss < best_loss:
                # self.loss will change in the forward fn to the one obtained in current iter
                best_c = c_values[i]
                best_loss = self.loss
        return best_c, best_loss
    
    def update_ca_runtime(self, new_value):
        new_n = self.n + 1
        ca_runtime_new = (new_value + (self.n * self.ca_runtime)) / new_n
        self.n = new_n
        self.ca_runtime = ca_runtime_new


class HpfCW(CW):
    
    def __init__(self, model, model_trms, surr_model_loss, hpf_masker, *args, **kwargs):
        super().__init__(model, model_trms, *args, **kwargs)
        self.loss = HPFL2()  # internally (N, 3, H, W)
        self.hpf_masker = hpf_masker
        self.surr_model_loss = surr_model_loss

    def get_iq_loss(self, adv_images, images, attack_mask):
        current_iq_loss = self.loss(adv_images, images, attack_mask)
        iq_loss = current_iq_loss.sum()
        return iq_loss, current_iq_loss

class WHpfCW(CW):
    
    """Introduces a constraint based on wavelet frequency processing
    that computes the L2 with HF details excluded.
    """
    
    def __init__(self, model, model_trms, surr_model_loss, hpf_masker, *args, **kwargs):
        super().__init__(model, model_trms, *args, **kwargs)
        self.loss = HPFL2()  # internally (N, 3, H, W)
        self.hpf_masker = hpf_masker
        self.surr_model_loss = surr_model_loss

    def get_iq_loss(self, adv_images, images, attack_mask):
        current_iq_loss = self.loss(adv_images, images, attack_mask)
        iq_loss = current_iq_loss.sum()
        return iq_loss, current_iq_loss
        
        
class SsimCW(CW):
    
    def __init__(self, model, model_trms, *args, **kwargs):
        super().__init__(model, model_trms, *args, **kwargs)
        self.loss = SSIM(channels=3) # internally (N, 3, H, W)

    def get_iq_loss(self, adv_images, images, attack_mask):
        current_iq_loss = self.loss(adv_images, images)
        iq_loss = current_iq_loss.sum()
        return iq_loss, current_iq_loss

class MSSsimCW(CW):
    
    def __init__(self, model, model_trms, *args, **kwargs):
        super().__init__(model, model_trms, *args, **kwargs)
        self.loss = MSSSIM(data_range=1.0).to(self.device)

    def get_iq_loss(self, adv_images, images, attack_mask):
        current_iq_loss = self.loss(adv_images, images)
        current_iq_loss = 1 - current_iq_loss
        iq_loss = current_iq_loss.sum()
        return iq_loss, current_iq_loss


class MadCW(CW):
    
    """currently not usable
    (1) no normalization possible
    (2) computationally prohibitive
    (3) for very little dist. returns 0 
    """
    
    def __init__(self, model, model_trms, *args, **kwargs):
        super().__init__(model, model_trms,*args, **kwargs)
        self.loss = MAD(channels=3) # internally (N, 3, H, W)

    def get_iq_loss(self, adv_images, images, attack_mask):
        current_iq_loss = self.loss(images, adv_images)
        iq_loss = current_iq_loss.sum()
        return iq_loss, current_iq_loss


class PsnrCW(CW):
    
    def __init__(self, model, model_trms, *args, **kwargs):
        super().__init__(model, model_trms, *args, **kwargs)
        self.loss = PSNR(channels=3) # internally (N, 3, H, W)

    def get_iq_loss(self, adv_images, images, attack_mask):
        current_iq_loss = self.loss(adv_images, images)
        iq_loss = current_iq_loss.sum()
        return iq_loss, current_iq_loss


class DistsCW(CW):
    
    def __init__(self, model, model_trms, *args, **kwargs):
        """From:
        https://arxiv.org/abs/2004.07728
        """
        super().__init__(model, model_trms, *args, **kwargs)
        self.loss = DISTS().to(self.device)

    def get_iq_loss(self, adv_images, images, attack_mask):
        current_iq_loss = self.loss(adv_images, images, require_grad=True, batch_average=True)
        iq_loss = current_iq_loss.sum()
        return iq_loss, current_iq_loss

class VifCW(CW):
    
    def __init__(self, model, model_trms, *args, **kwargs):
        """Based on VIF metric
        
        weird value range makes this hard to use for optimization
        """
        
        super().__init__(model, model_trms, *args, **kwargs)
        self.loss = VIF()

    def get_iq_loss(self, adv_images, images, attack_mask):
        current_iq_loss = self.loss(adv_images, images)
        current_iq_loss = 1 - current_iq_loss
        iq_loss = current_iq_loss.sum() # 1 is maximum fidelitys
        return iq_loss, current_iq_loss
    
    
class WRCW(CW):
    
    def __init__(self, model, model_trms, rcw_comp_lower_bound, rcw_beta, *args, **kwargs):
        """Robust CW employs a differentiable approximation
        in the constrained optimization in the form of T
        to improve the robustness in compression scenarios.
        
        DWT(x) = x_ll, x_lh, x_hl, x_hh

        optim(x_ll, x_lh, x_hl)

        min(x_ll, x_lh, x_hl) = || IDWT(x_ll, x_lh, x_hl, x_hh) - x || + c * f(JPEG(IDWT(x_ll, x_lh, x_hl, x_hh)) ; theta)
        
        jpeg_quality argument should be in range [0,1]
        they will be internally projected to 255
        and after compression back to 1.0
        
        self.compress(x) should be applied directly on image 
        
        """
        super().__init__(model, model_trms, *args, **kwargs)
        self.hpf_dwt = HpfDWT(device=self.device)
        self.compression_lower_bound = rcw_comp_lower_bound
        self.mu = 40
        
    def compress(self, img, jpeg_quality):
        img = img * 255
        compressed =  diff_jpeg_coding(image_rgb=img, jpeg_quality=jpeg_quality)
        return (compressed / 255).clip(min=0., max=1.)

    def get_robustness_loss(self, adv_images, labels, target_labels):
        
        
        #compressed_img = adv_images.clone().detach()
        #compressed_img.requires_grad = True
        random_quality = random.randint(self.compression_lower_bound, 99)
        compressed_img = self.compress(adv_images, jpeg_quality=torch.tensor([random_quality]).to(self.device))
        
        #outputs = self.get_outputs(adv_images)
        compressed_outputs = self.get_outputs(compressed_img)
        
        if self.targeted:
            f_compressed = self.f(compressed_outputs, target_labels).sum()
        else:
            f_compressed = self.f(compressed_outputs, labels).sum()
            
        return f_compressed, compressed_outputs

    def forward_regular(self, images, labels):
        r"""
        Overridden.
        """
        print(f'\nCalculating adversarial example for images...\n\n')
        #images = images / 255
        
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        
        if self.use_attack_mask:
            attack_mask = self.hpf_masker(images, 
                                        labels, 
                                        model=self.model, 
                                        model_trms=self.model_trms, 
                                        loss=self.surr_model_loss,
                                        invert_mask=True)
        else:
            attack_mask = 1 # multiplication results in identity
        
        # w = torch.zeros_like(images).detach() # Requires 2x times
        
        x_ll, x_lh_hl_hh = self.hpf_dwt(images)
        #w = self.inverse_tanh_space(x_ll).detach()
        #w.requires_grad = True
        
        reference_x_ll = x_ll.clone().detach()
        reference_x_ll.requires_grad = True
        
        x_ll.requires_grad = True

        best_adv_images = images.clone().detach()
        best_iq = 1e10*torch.ones((len(images))).to(self.device)
        prev_cost = 1e10
        dim = len(images.shape)

        optimizer = optim.Adam([x_ll], lr=self.lr)
        optimizer.zero_grad()
        

        for step in range(self.steps):
                
            # Get adversarial images
            #x_ll = self.tanh_space(w)
            
            adv_images = self.hpf_dwt.invert(x_ll, x_lh_hl_hh)

            # Calculate image quality loss
            iq_loss, current_iq_loss = self.get_iq_loss(x_ll, reference_x_ll, attack_mask)

            # Calculate adversarial loss including robustness loss
            ro_loss, comp_outputs = self.get_robustness_loss(adv_images, labels, target_labels)

            regularization_penalty = torch.sum(adv_images[adv_images > 1] - 1) + torch.sum(-adv_images[adv_images < 0])

            # cost = iq_loss + self.c*f_loss + self.beta*ro_loss
            cost = iq_loss + self.c*ro_loss + self.mu*regularization_penalty
            
            # Update adversarial images
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # check if adv_images were classified incorrectly
            _, pre = torch.max(comp_outputs.detach(), 1)
            correct = (pre == labels).float()

            # filter out images that get either correct predictions or non-decreasing loss, 
            # i.e., only images that are both misclassified and loss-decreasing are left 
            mask = (1-correct)*(best_iq > current_iq_loss.detach())
            best_iq = mask*current_iq_loss.detach() + (1-mask)*best_iq

            mask = mask.view([-1]+[1]*(dim-1))
            best_adv_images = mask*adv_images.detach() + (1-mask)*best_adv_images

            # Early stop when loss does not converge.
            # max(.,1) To prevent MODULO BY ZERO error in the next step.
            if step % max(self.steps//10,1) == 0:
                if cost.item() >= prev_cost:
                    break
                prev_cost = cost.item()
                
        if self.protocol_file:
            self.write_to_protocol_dir(iq_loss, ro_loss, cost)
            self.write_runtime_to_protocol_dir()
        
        best_adv_images = torch.clip(best_adv_images, max=1.0, min=0.0)
        return (best_adv_images, target_labels)

    def forward_verbose(self, images, labels):
        r"""
        Overridden.
        """
        print(f'\nCalculating adversarial example for images...\n\n')
        #images = images / 255
        
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        
        if self.use_attack_mask:
            attack_mask = self.hpf_masker(images, 
                                        labels, 
                                        model=self.model, 
                                        model_trms=self.model_trms, 
                                        loss=self.surr_model_loss,
                                        invert_mask=True)
        else:
            attack_mask = 1 # multiplication results in identity
        
        # w = torch.zeros_like(images).detach() # Requires 2x times
        
        x_ll, x_lh_hl_hh = self.hpf_dwt(images)
        #w = self.inverse_tanh_space(x_ll).detach()
        #w.requires_grad = True
        
        reference_x_ll = x_ll.clone().detach()
        reference_x_ll.requires_grad = True
        
        x_ll.requires_grad = True

        best_adv_images = images.clone().detach()
        best_iq = 1e10*torch.ones((len(images))).to(self.device)
        prev_cost = 1e10
        dim = len(images.shape)

        optimizer = optim.Adam([x_ll], lr=self.lr)
        optimizer.zero_grad()
        

        for step in range(self.steps):
                
            # Get adversarial images
            #x_ll = self.tanh_space(w)
            
            adv_images = self.hpf_dwt.invert(x_ll, x_lh_hl_hh)
            
            if adv_images.max() > 1.0 or adv_images.min() < 0.0:
                print('WARNING: image out of bounds!')

            # Calculate image quality loss
            iq_loss, current_iq_loss = self.get_iq_loss(x_ll, reference_x_ll, attack_mask)

            # Calculate adversarial loss including robustness loss
            ro_loss, comp_outputs = self.get_robustness_loss(adv_images, labels, target_labels)

            regularization_penalty = torch.sum(adv_images[adv_images > 1] - 1) + torch.sum(-adv_images[adv_images < 0])

            # cost = iq_loss + self.c*f_loss + self.beta*ro_loss
            cost = iq_loss + self.c*ro_loss + self.mu*regularization_penalty
            
            # Update adversarial images
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # check if adv_images were classified incorrectly
            _, pre = torch.max(comp_outputs.detach(), 1)
            correct = (pre == labels).float()

            # filter out images that get either correct predictions or non-decreasing loss, 
            # i.e., only images that are both misclassified and loss-decreasing are left 
            mask = (1-correct)*(best_iq > current_iq_loss.detach())
            best_iq = mask*current_iq_loss.detach() + (1-mask)*best_iq

            mask = mask.view([-1]+[1]*(dim-1))
            best_adv_images = mask*adv_images.detach() + (1-mask)*best_adv_images

            # Early stop when loss does not converge.
            # max(.,1) To prevent MODULO BY ZERO error in the next step.
            if step % max(self.steps//10,1) == 0:
                if cost.item() >= prev_cost:
                    break
                prev_cost = cost.item()
            
            print(f'\n{step} - iq_loss: {current_iq_loss.item()}, ro_loss: {ro_loss.item()} cost: {cost.item()}')

        if self.protocol_file:
            self.write_to_protocol_dir(iq_loss, ro_loss, cost)
            self.write_runtime_to_protocol_dir()
        
        best_adv_images = torch.clip(best_adv_images, max=1.0, min=0.0)        
        return (best_adv_images, target_labels)

'''class YCW(CW):
    
    def __init__(self, model, model_trms, rcw_comp_lower_bound, rcw_beta, *args, **kwargs):
        """
        calculates perturbations in the Y space
        for robustness against color downsampling in JPEG
        """
        super().__init__(model, model_trms, *args, **kwargs)
        self.compression_lower_bound = rcw_comp_lower_bound
        self.hpf_dwt = HpfDWT(device=self.device)
        self.beta = rcw_beta
        self.to_grey = T.Grayscale()
        self.mu = 40
        
    def compress(self, img, jpeg_quality):
        img = img * 255
        compressed =  diff_jpeg_coding(image_rgb=img, jpeg_quality=jpeg_quality)
        return (compressed / 255).clip(min=0., max=1.)

    def get_robustness_loss(self, adv_images, labels, target_labels):
        
        random_quality = random.randint(self.compression_lower_bound, 99)
        compressed_img = self.compress(adv_images, jpeg_quality=torch.tensor([random_quality]).to(self.device))
        
        #outputs = self.get_outputs(adv_images)
        compressed_outputs = self.get_outputs(compressed_img)
        
        if self.targeted:
            f_compressed = self.f(compressed_outputs, target_labels).sum()
        else:
            f_compressed = self.f(compressed_outputs, labels).sum()
        return f_compressed, compressed_outputs
    
    """def separate_color_lum(self, img):
        img_grey = self.to_grey(img)
        img_color = img - img_grey
        return img_grey, img_color"""

    def separate_color_lum(self, img):
        ycbcr_img = rgb_to_ycbcr(img)
        y = ycbcr_img[0][0]
        return y, ycbcr_img
    
    """def invert_color_lum(self, img_grey, img_color):
        img = img_grey + img_color
        return img"""
    
    def invert_color_lum(self, updated_y, ycbcr_img):
        ycbcr_img[0][0] = updated_y
        rgb_img = ycbcr_to_rgb(ycbcr_img)
        return rgb_img

    def forward_regular(self, images, labels):
        r"""
        Overridden.
        """
        print(f'\nCalculating adversarial example for images...\n\n')
        #images = images / 255
        
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        
        if self.use_attack_mask:
            attack_mask = self.hpf_masker(images, 
                                        labels, 
                                        model=self.model, 
                                        model_trms=self.model_trms, 
                                        loss=self.surr_model_loss,
                                        invert_mask=True)
        else:
            attack_mask = 1 # multiplication results in identity
        
        # w = torch.zeros_like(images).detach() # Requires 2x times
        
        x_ll, _ = self.hpf_dwt(images)
        reference_x_ll = x_ll.clone().detach()
        reference_x_ll.requires_grad = True
        x_lum, x_color = self.separate_color_lum(images)

        x_lum.requires_grad = True

        best_adv_images = images.clone().detach()
        best_iq = 1e10*torch.ones((len(images))).to(self.device)
        prev_cost = 1e10
        dim = len(images.shape)

        optimizer = optim.Adam([x_lum], lr=self.lr)
        optimizer.zero_grad()

        for step in range(self.steps):
                
            # Get adversarial images
            
            adv_images = self.invert_color_lum(x_lum, x_color)
            #adv_images = torch.clip(adv_z, min=0.0, max=1.0)
            
            if adv_images.max() > 1.0 or adv_images.min() < 0.0:
                print('WARNING: image out of bounds!')

            # Calculate image quality loss
            x_ll, _ = self.hpf_dwt(adv_images)
            iq_loss, current_iq_loss = self.get_iq_loss(x_ll, reference_x_ll, attack_mask)

            # Calculate adversarial loss including robustness loss
            #f_loss, ro_loss, outputs, compressed_outputs = self.get_robustness_loss(adv_images, labels, target_labels)
            ro_loss, compressed_outputs = self.get_robustness_loss(adv_images, labels, target_labels)
            
            #regularization_penalty = torch.max(torch.zeros_like(adv_images), adv_images - 1, )# + torch.max(0, 0 - adv_images)
            regularization_penalty = torch.sum(adv_images[adv_images > 1] - 1) + torch.sum(-adv_images[adv_images < 0])

            # cost = iq_loss + self.c*f_loss + self.beta*ro_loss
            cost = iq_loss + self.c*ro_loss + self.mu*regularization_penalty
            # Update adversarial images
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            _, pre_comp = torch.max(compressed_outputs.detach(), 1)
            correct = (pre_comp == labels).float()
            # union bc if either one predicts true we want to keep optimizing

            # filter out images that get either correct predictions or non-decreasing loss, 
            # i.e., only images that are both misclassified and loss-decreasing are left 
            mask = (1-correct)*(best_iq > current_iq_loss.detach())
            best_iq = mask*current_iq_loss.detach() + (1-mask)*best_iq

            mask = mask.view([-1]+[1]*(dim-1))
            best_adv_images = mask*adv_images.detach() + (1-mask)*best_adv_images

            # Early stop when loss does not converge.
            # max(.,1) To prevent MODULO BY ZERO error in the next step.
            if step % max(self.steps//10,1) == 0:
                if cost.item() >= prev_cost:
                    break
                prev_cost = cost.item()
            
        if self.protocol_file:
            self.write_to_protocol_dir(iq_loss, ro_loss, cost)
            self.write_runtime_to_protocol_dir()
        best_adv_images = torch.clip(best_adv_images, max=1.0, min=0.0)
        return (best_adv_images, target_labels)
    


    def forward_verbose(self, images, labels):
        r"""
        Overridden.
        """
        print(f'\nCalculating adversarial example for images...\n\n')
        #images = images / 255
        
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        
        if self.use_attack_mask:
            attack_mask = self.hpf_masker(images, 
                                        labels, 
                                        model=self.model, 
                                        model_trms=self.model_trms, 
                                        loss=self.surr_model_loss,
                                        invert_mask=True)
        else:
            attack_mask = 1 # multiplication results in identity
        
        # w = torch.zeros_like(images).detach() # Requires 2x times
        
        x_ll, _ = self.hpf_dwt(images)
        reference_x_ll = x_ll.clone().detach()
        reference_x_ll.requires_grad = True
        x_lum, x_color = self.separate_color_lum(images)

        x_lum.requires_grad = True

        best_adv_images = images.clone().detach()
        best_iq = 1e10*torch.ones((len(images))).to(self.device)
        prev_cost = 1e10
        dim = len(images.shape)

        optimizer = optim.Adam([x_lum], lr=self.lr)
        optimizer.zero_grad()

        for step in range(self.steps):
                
            # Get adversarial images
            
            adv_images = self.invert_color_lum(x_lum, x_color)
            #adv_images = torch.clip(adv_z, min=0.0, max=1.0)
            
            if adv_images.max() > 1.0 or adv_images.min() < 0.0:
                print('WARNING: image out of bounds!')

            # Calculate image quality loss
            x_ll, _ = self.hpf_dwt(adv_images)
            iq_loss, current_iq_loss = self.get_iq_loss(x_ll, reference_x_ll, attack_mask)

            # Calculate adversarial loss including robustness loss
            #f_loss, ro_loss, outputs, compressed_outputs = self.get_robustness_loss(adv_images, labels, target_labels)
            ro_loss, compressed_outputs = self.get_robustness_loss(adv_images, labels, target_labels)
            
            #regularization_penalty = torch.max(torch.zeros_like(adv_images), adv_images - 1, )# + torch.max(0, 0 - adv_images)
            regularization_penalty = torch.sum(adv_images[adv_images > 1] - 1) + torch.sum(-adv_images[adv_images < 0])

            # cost = iq_loss + self.c*f_loss + self.beta*ro_loss
            cost = iq_loss + self.c*ro_loss + self.mu*regularization_penalty
            # Update adversarial images
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            _, pre_comp = torch.max(compressed_outputs.detach(), 1)
            correct = (pre_comp == labels).float()
            # union bc if either one predicts true we want to keep optimizing

            # filter out images that get either correct predictions or non-decreasing loss, 
            # i.e., only images that are both misclassified and loss-decreasing are left 
            mask = (1-correct)*(best_iq > current_iq_loss.detach())
            best_iq = mask*current_iq_loss.detach() + (1-mask)*best_iq

            mask = mask.view([-1]+[1]*(dim-1))
            best_adv_images = mask*adv_images.detach() + (1-mask)*best_adv_images

            # Early stop when loss does not converge.
            # max(.,1) To prevent MODULO BY ZERO error in the next step.
            if step % max(self.steps//10,1) == 0:
                if cost.item() >= prev_cost:
                    break
                prev_cost = cost.item()
            

            print(f'\n{step} - iq_loss: {current_iq_loss.item()}, r_loss: {ro_loss.item()}, cost: {cost.item()}')
            
        if self.protocol_file:
            self.write_to_protocol_dir(iq_loss, ro_loss, cost)
            self.write_runtime_to_protocol_dir()
        best_adv_images = torch.clip(best_adv_images, max=1.0, min=0.0)
        return (best_adv_images, target_labels)'''


class YCW(CW):
    
    def __init__(self, model, model_trms, rcw_comp_lower_bound, rcw_beta, *args, **kwargs):
        """
        calculates perturbations in the Y space
        for robustness against color downsampling in JPEG
        """
        super().__init__(model, model_trms, *args, **kwargs)
        self.compression_lower_bound = rcw_comp_lower_bound
        self.hpf_dwt = HpfDWT(device=self.device)
        self.beta = rcw_beta
        self.to_grey = T.Grayscale()
        self.mu = 40
        
    def compress(self, img, jpeg_quality):
        img = img * 255
        compressed =  diff_jpeg_coding(image_rgb=img, jpeg_quality=jpeg_quality)
        return (compressed / 255).clip(min=0., max=1.)

    def get_robustness_loss(self, adv_images, labels, target_labels):
        
        random_quality = random.randint(self.compression_lower_bound, 99)
        compressed_img = self.compress(adv_images, jpeg_quality=torch.tensor([random_quality]).to(self.device))
        
        #outputs = self.get_outputs(adv_images)
        compressed_outputs = self.get_outputs(compressed_img)
        
        if self.targeted:
            f_compressed = self.f(compressed_outputs, target_labels).sum()
        else:
            f_compressed = self.f(compressed_outputs, labels).sum()
        return f_compressed, compressed_outputs
    
    """def separate_color_lum(self, img):
        img_grey = self.to_grey(img)
        img_color = img - img_grey
        return img_grey, img_color"""

    def separate_color_lum(self, img):
        ycbcr_img = rgb_to_ycbcr(img)
        y = ycbcr_img[0][0]
        return y, ycbcr_img
    
    """def invert_color_lum(self, img_grey, img_color):
        img = img_grey + img_color
        return img"""
    
    def invert_color_lum(self, updated_y, ycbcr_img):
        ycbcr_img[0][0] = updated_y
        rgb_img = ycbcr_to_rgb(ycbcr_img)
        return rgb_img

    def forward_regular(self, images, labels):
        r"""
        Overridden.
        """
        
        print(f'\nCalculating adversarial example for images...\n\n')
        #images = images / 255

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        
        if self.use_attack_mask:
            attack_mask = self.hpf_masker(images, 
                                        labels, 
                                        model=self.model, 
                                        model_trms=self.model_trms, 
                                        loss=self.surr_model_loss,
                                        invert_mask=True)
        else:
            attack_mask = 1 # multiplication results in identity
        
        # w = torch.zeros_like(images).detach() # Requires 2x times
        
        #x_ll, _ = self.hpf_dwt(images)
        #reference_x_ll = x_ll.clone().detach()
        #reference_x_ll.requires_grad = True
        
        # optimize w(y) only
        ycbcr_img = rgb_to_ycbcr(images).detach()
        w = self.inverse_tanh_space(ycbcr_img).detach()
        w_y = w[0][0].clone().detach()
        w_cb = w[0][1].clone().detach()
        w_cr = w[0][2].clone().detach()
        w_y.requires_grad = True

        best_adv_images = images.clone().detach()
        best_iq = 1e10*torch.ones((len(images))).to(self.device)
        prev_cost = 1e10
        dim = len(images.shape)

        optimizer = optim.Adam([w_y], lr=self.lr)
        optimizer.zero_grad()
    

        for step in range(self.steps):
    
            w_cat = torch.stack((w_y, w_cb, w_cr))
            w_cat = w_cat.unsqueeze(0)            
            # Get adversarial images
            
            #w[0][0] = w_y
            ycbcr_adv_images = self.tanh_space(w_cat)
            adv_images = ycbcr_to_rgb(ycbcr_adv_images)

            # Calculate image quality loss
            #x_ll, _ = self.hpf_dwt(adv_images)
            iq_loss, current_iq_loss = self.get_iq_loss(adv_images, images, attack_mask)

            # Calculate adversarial loss including robustness loss
            ro_loss, compressed_outputs = self.get_robustness_loss(adv_images, labels, target_labels)
            
            regularization_penalty = torch.sum(adv_images[adv_images > 1] - 1) + torch.sum(-adv_images[adv_images < 0])

            cost = iq_loss + self.c*ro_loss + self.mu*regularization_penalty
            # Update adversarial images
            optimizer.zero_grad()
            cost.backward() #retain_graph=True
            optimizer.step()

            _, pre_comp = torch.max(compressed_outputs.detach(), 1)
            correct = (pre_comp == labels).float()
            # union bc if either one predicts true we want to keep optimizing

            # filter out images that get either correct predictions or non-decreasing loss, 
            # i.e., only images that are both misclassified and loss-decreasing are left 
            mask = (1-correct)*(best_iq > current_iq_loss.detach())
            best_iq = mask*current_iq_loss.detach() + (1-mask)*best_iq

            mask = mask.view([-1]+[1]*(dim-1))
           # best_adv_images = mask*adv_images.detach() + (1-mask)*best_adv_images

            # Early stop when loss does not converge.
            # max(.,1) To prevent MODULO BY ZERO error in the next step.
            if step % max(self.steps//10,1) == 0:
                if cost.item() >= prev_cost:
                    break
                prev_cost = cost.item()
            
        if self.protocol_file:
            self.write_to_protocol_dir(iq_loss, ro_loss, cost)
            self.write_runtime_to_protocol_dir()
        best_adv_images = torch.clip(best_adv_images, max=1.0, min=0.0)
        return (best_adv_images, target_labels)
    


    def forward_verbose(self, images, labels):
        r"""
        Overridden.
        """
        
        print(f'\nCalculating adversarial example for images...\n\n')
        #images = images / 255

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        
        if self.use_attack_mask:
            attack_mask = self.hpf_masker(images, 
                                        labels, 
                                        model=self.model, 
                                        model_trms=self.model_trms, 
                                        loss=self.surr_model_loss,
                                        invert_mask=True)
        else:
            attack_mask = 1 # multiplication results in identity
        
        # w = torch.zeros_like(images).detach() # Requires 2x times
        
        #x_ll, _ = self.hpf_dwt(images)
        #reference_x_ll = x_ll.clone().detach()
        #reference_x_ll.requires_grad = True
        
        # optimize w(y) only
        ycbcr_img = rgb_to_ycbcr(images).detach()
        w = self.inverse_tanh_space(ycbcr_img).detach()
        w_y = w[0][0].clone().detach()
        w_cb = w[0][1].clone().detach()
        w_cr = w[0][2].clone().detach()
        w_y.requires_grad = True

        best_adv_images = images.clone().detach()
        best_iq = 1e10*torch.ones((len(images))).to(self.device)
        prev_cost = 1e10
        dim = len(images.shape)

        optimizer = optim.Adam([w_y], lr=self.lr)
        optimizer.zero_grad()
    

        for step in range(self.steps):
    
            w_cat = torch.stack((w_y, w_cb, w_cr))
            w_cat = w_cat.unsqueeze(0)            
            # Get adversarial images
            
            #w[0][0] = w_y
            ycbcr_adv_images = self.tanh_space(w_cat)
            adv_images = ycbcr_to_rgb(ycbcr_adv_images)
            
            if adv_images.max() > 1.0 or adv_images.min() < 0.0:
                print(f'WARNING: image out of bounds! max:{adv_images.max()}  min:{adv_images.min()}')

            # Calculate image quality loss
            #x_ll, _ = self.hpf_dwt(adv_images)
            iq_loss, current_iq_loss = self.get_iq_loss(adv_images, images, attack_mask)

            # Calculate adversarial loss including robustness loss
            ro_loss, compressed_outputs = self.get_robustness_loss(adv_images, labels, target_labels)
            
            regularization_penalty = torch.sum(adv_images[adv_images > 1] - 1) + torch.sum(-adv_images[adv_images < 0])

            cost = iq_loss + self.c*ro_loss + self.mu*regularization_penalty
            # Update adversarial images
            optimizer.zero_grad()
            cost.backward() #retain_graph=True
            optimizer.step()

            _, pre_comp = torch.max(compressed_outputs.detach(), 1)
            correct = (pre_comp == labels).float()
            # union bc if either one predicts true we want to keep optimizing

            # filter out images that get either correct predictions or non-decreasing loss, 
            # i.e., only images that are both misclassified and loss-decreasing are left 
            mask = (1-correct)*(best_iq > current_iq_loss.detach())
            best_iq = mask*current_iq_loss.detach() + (1-mask)*best_iq

            mask = mask.view([-1]+[1]*(dim-1))
           # best_adv_images = mask*adv_images.detach() + (1-mask)*best_adv_images

            # Early stop when loss does not converge.
            # max(.,1) To prevent MODULO BY ZERO error in the next step.
            if step % max(self.steps//10,1) == 0:
                if cost.item() >= prev_cost:
                    break
                prev_cost = cost.item()

            print(f'\n{step} - iq_loss: {current_iq_loss.item()}, r_loss: {ro_loss.item()}, cost: {cost.item()}')
            
        if self.protocol_file:
            self.write_to_protocol_dir(iq_loss, ro_loss, cost)
            self.write_runtime_to_protocol_dir()
        best_adv_images = torch.clip(best_adv_images, max=1.0, min=0.0)
        return (best_adv_images, target_labels)
    