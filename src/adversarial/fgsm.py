import torch
import torchvision.transforms as T
import pytorch_colors as colors

from src.adversarial.attack_base import Attack


class FGSM(Attack):
    r"""
    FGSM in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.FGSM(model, eps=8/255)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, model, surrogate_loss, model_trms, eps=8/255, regularization=False, l2_lambda=0.001):
        super().__init__("FGSM", model, model_trms)
        self.eps = eps
        self.supported_mode = ['default', 'targeted']
        self.regularization = regularization
        self.l2_lambda = l2_lambda
        self.get_loss_fn = self.get_loss_l2 if regularization == True else self.get_loss_regular
        self.loss = surrogate_loss
        self.model_trms = model_trms
        self.avg_salient_mask = torch.zeros((3,224,224))
        self.n = 0

    def activations_to_avg_mask(self, grads):
        batch_size = grads.shape[0]
        abs_grads = grads.abs()
        sum_abs_grads = grads.sum(dim=0)
        self.avg_salient_mask += sum_abs_grads
        self.n += batch_size
    
    def get_avg_salient_mask(self):
        return ((self.avg_salient_mask / self.n) / self.avg_salient_mask.max())


    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)
        
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
        
        self.activations_to_avg_mask(grad.clone().detach())

        adv_images = images + self.eps*grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        return adv_images
    
    def get_loss(self, outputs):
        self.get_loss_fn(outputs)
    
    def get_loss_l2(self, outputs):
        loss = self.loss(outputs) + self.get_abs_param_sum() * self.l2_lambda
        return loss
    
    def get_loss_regular(self, outputs):
        loss = self.loss(outputs)
        return loss

    def get_abs_param_sum(self):
        abs_param_sum = 0
        for p in self.model.parameters():
            abs_param_sum += p.pow(2.0).sum()
        return abs_param_sum
        
        
class YCBCRTransform:
    
    def __init__(self):
        print('Warning: input tensors must be in the format RGB')
    
    def __call__(self, tensor):
        ycbcr = self.get_channels(tensor)
        return ycbcr
    
    @staticmethod
    def normalize(tensor):
        tensor / 255
        return tensor

    @staticmethod
    def to_int(tensor):
        tensor += 0.5
        tensor *= 255
        return tensor
        
    def get_channels(self, tensor):
        return colors.rgb_to_ycbcr(tensor)
    
    def inverse(self, tensor):
        return colors.ycbcr_to_rgb(tensor)


class HpfFGSM(FGSM):
    r"""
    HPF-Version of standard FGSM
    """
    def __init__(self, model, hpf_masker, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.hpf_masker = hpf_masker

    def get_eps_delta(self, mask):
        # mask values are between 0-1
        mean_inv_mask = (1 - mask).mean()
        eps_delta = 1 + mean_inv_mask
        return eps_delta.item()
        
    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        # =====inserted===
        
        attack_mask = self.hpf_masker(images, labels, model=self.model, model_trms=self.model_trms, loss=self.loss)
        
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

        eps_delta = self.get_eps_delta(attack_mask)
        adv_images = images + (self.eps*eps_delta)*attack_mask*grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        return adv_images
    

class YcbcrHpfFGSM(FGSM):
    r"""
    HPF-Version of standard FGSM
    """
    def __init__(self, model, hpf_masker, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.hpf_masker = hpf_masker
        self.to_grey = T.Grayscale()
        
    def get_eps_delta(self, mask):
        # mask values are between 0-1
        ones_tensor = torch.ones_like(mask)
        greyscale_mask = self.to_grey(mask)
        ones_tensor[0, 0] = greyscale_mask
        mean_inv_mask = (1 - ones_tensor).mean()
        eps_delta = 1 + mean_inv_mask
        return eps_delta.item()

    def normalize_ycbcr(self, imgs):
        imgs = imgs.to(torch.float32)
        max_item_half = imgs[0].max().item() / 2
        imgs = imgs - max_item_half
        imgs = imgs / max_item_half
        return imgs

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # =====inserted===
        
        attack_mask = self.hpf_masker(images, labels, model=self.model, model_trms=self.model_trms, loss=self.loss)
        
        images = images.to(torch.float32) / 255
        images.requires_grad = True
        
        eps_delta = self.get_eps_delta(attack_mask)
        adjusted_eps = self.eps * eps_delta

        # Get Loss
        outputs = self.get_logits(self.model_trms(images))
        if self.targeted:
            cost = -self.loss(outputs, target_labels)
        else:
            cost = self.loss(outputs, labels)

        # Update adversarial images
        grad = torch.autograd.grad(cost, images,
                                   retain_graph=False, create_graph=False)[0]

        # Make HPF and regular adv_images
        adv_images = images + adjusted_eps*grad.sign().detach()
        #adv_images = torch.clamp(inter_adv_images, min=0, max=1).detach()
        
        hpf_adv_images = images + adjusted_eps*attack_mask*grad.sign().detach()
        #hpf_adv_images = torch.clamp(hpf_inter_adv_images, min=0, max=1).detach()
        
        # Extract crominance info from orig adv, extract luminance from hpf adv
        adv_img_luminance = self.to_grey(adv_images)
        hpf_adv_img_luminance = self.to_grey(hpf_adv_images)
        
        crominance_adv_img = adv_images - adv_img_luminance

        # Combine crominance from orig adv and luminance from hpf adv
        final_adv_imgs = crominance_adv_img + hpf_adv_img_luminance
        final_adv_imgs = torch.clamp(final_adv_imgs, min=0, max=1)

        return final_adv_imgs