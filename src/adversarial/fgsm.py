import torch
import torchvision.transforms as T
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
        
