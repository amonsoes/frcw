import torch
import torchattacks
import csv
import os
import torchvision

from torchvision import transforms as T
from torchvision.io import encode_jpeg, decode_image
from src.model.pretrained import CNNLoader
from src.adversarial.hpf_mask import HPFMasker
from src.adversarial.black_box.boundary_attack import BoundaryAttack, HPFBoundaryAttack
from src.adversarial.black_box.nes_attack import NESAttack
from src.adversarial.black_box.square_attack import SquareAttack, HPFSquareAttack 
from src.adversarial.black_box.pg_rgf import PriorRGFAttack, HPFPriorRGFAttack
from src.adversarial.iqm import ImageQualityMetric
from src.adversarial.custom_cw import CW, WRCW, YCW
from src.adversarial.robust_cw import VarRCW, RCW
from src.adversarial.perc_cw import PerC_CW
from src.adversarial.cvfgsm import CVFGSM
from src.adversarial.jpeg_ifgm import JIFGSM
from src.adversarial.unipgd import PGD, UniPGD
from src.adversarial.uvmifgsm import UVMIFGSM, VMIFGSM
from src.adversarial.uap_attack import UAP
from src.adversarial.dct_attack import DCTCW
from src.adversarial.adversarial_rounding import FastAdversarialRounding
from src.adversarial.fgsm import FGSM, HpfFGSM

class AttackLoader:
    
    def __init__(self, 
                attack_type, 
                device,
                model,
                surrogate_model, 
                dataset_type, 
                model_trms,
                surrogate_model_trms,
                hpf_mask_params,
                input_size,
                jpeg_compr_obj=None,
                *args, 
                **kwargs):
        self.device = device
        self.dataset_type = dataset_type
        self.model = model
        self.model_trms = model_trms
        self.surrogate_model = self.get_model(surrogate_model)
        self.surrogate_model_trms = surrogate_model_trms
        self.jpeg_compr_obj = jpeg_compr_obj
        self.attack = self.load_attack(attack_type, hpf_mask_params, input_size, *args, **kwargs)
        
    def load_attack(self, attack_type, hpf_mask_params, input_size, *args, **kwargs):
        # if hpf version create hpf_masker
        if attack_type in ['hpf_fgsm',
                            'hpf_bim',
                            'hpf_pgd',
                            'hpf_pgdl2',
                            'hpf_vmifgsm',
                            'ycbcr_hpf_fgsm',
                            'hpfcw']:
            hpf_masker = HPFMasker(self.device, input_size=input_size, **hpf_mask_params)
        elif attack_type in ['hpf_boundary_attack', 'hpf_square_attack', 'hpf_pg_rgf']:
            hpf_masker = HPFMasker(self.device, input_size=input_size, is_black_box=True, **hpf_mask_params)
        else:
            hpf_masker = None
        # load white or black box
        if attack_type in ['fgsm',
                           'bim',
                           'jifgsm',
                           'grad_gaussian_bim',
                           'pgd',
                           'pgdl2',
                           'unipgd',
                           'vmifgsm',
                           'cvfgsm',
                           'hpf_fgsm',
                           'hpf_bim',
                           'hpf_pgd',
                           'hpf_pgdl2',
                           'hpf_vmifgsm',
                           'uvmifgsm',
                           'ycbcr_hpf_fgsm',
                           'varsinifgsm',
                           'cw',
                           'ssimcw',
                           'madcw',
                           'psnrcw',
                           'hpfcw',
                           'distscw',
                           'vifcw',
                           'msssimcw',
                           'rcw',
                           'wrcw',
                           'ycw',
                           'varrcw',
                           'perccw',
                           'ercw',
                           'uap',
                           'dctcw',
                           'far']:
            attack = WhiteBoxAttack(attack_type=attack_type,
                                    surrogate_model=self.surrogate_model,
                                    device=self.device,
                                    input_size=self.input_size,
                                    dataset_type=self.dataset_type,
                                    surrogate_model_trms=self.surrogate_model_trms,
                                    hpf_masker=hpf_masker,
                                    jpeg_compr_obj=self.jpeg_compr_obj,
                                    *args, 
                                    **kwargs)
        elif attack_type in ['boundary_attack',
                            'nes', 
                            'hpf_boundary_attack',
                            'square_attack',
                            'hpf_square_attack',
                            'pg_rgf',
                            'hpf_pg_rgf']:
            attack = BlackBoxAttack(attack_type,
                                    model=self.model,
                                    surrogate_model=self.surrogate_model,
                                    device=self.device, 
                                    model_trms=self.model_trms,
                                    surrogate_model_trms=self.surrogate_model_trms,
                                    hpf_masker=hpf_masker,
                                    input_size=input_size,
                                    *args, 
                                    **kwargs)
        else:
            raise ValueError('ATTACK NOT RECOGNIZED. Change spatial_adv_type in options')
        return attack

    def get_model(self, surrogate_model):
        adv_training_protocol = None
        if self.dataset_type == '140k_flickr_faces':
            if surrogate_model == 'resnet':
                surrogate_path = './saves/models/FlickrCNN/2024-10-15_2/pretrained_resnet_pretrained_flick.pt'
                n_classes = 2
            else:
                raise ValueError('ATTACK CURRENTLY ONLY WITH RESNET. change surrogate_model in options')
        elif self.dataset_type == 'nips17':
            if surrogate_model == 'adv-resnet-fbf':
                surrogate_path = './saves/models/Adversarial/fbf_models/imagenet_model_weights_2px.pth.tar'
                adv_training_protocol = 'fbf'
            elif surrogate_model == 'adv-resnet-pgd':
                surrogate_path = './saves/models/Adversarial/pgd_models/imagenet_linf_4.pt'
                adv_training_protocol = 'pgd'
            else:
                surrogate_path = ''
            n_classes = 1000
        elif self.dataset_type == 'cifar10':
            n_classes = 10
            surrogate_path = '' # CIFAR models are loaded from the model hub
        
        loader = CNNLoader(surrogate_path, adv_training_protocol)
        cnn, self.input_size = loader.transfer(surrogate_model, n_classes, feature_extract=False, device=self.device)
        cnn.model_name = surrogate_model
        model = cnn
        model.eval()
        return model 

    def get_l2(self, orig_img, adv_img):
        distance = (orig_img - adv_img).pow(2).sum().sqrt()
        return (distance / orig_img.max()).item()

class BlackBoxAttack:
    
    def __init__(self, 
                attack_type, 
                model, 
                surrogate_model, 
                model_trms, 
                surrogate_model_trms, 
                hpf_masker,
                input_size,
                *args, 
                **kwargs):
        self.black_box_attack = self.load_attack(attack_type,
                                                model,
                                                model_trms,
                                                surrogate_model,
                                                surrogate_model_trms,
                                                hpf_masker,
                                                input_size,
                                                *args, 
                                                **kwargs)
        self.to_float = T.ConvertImageDtype(torch.float32)
        self.l2_norm = []
        self.num_queries_lst = []
        self.mse_list = []
        self.n = 1
    
    def load_attack(self, 
                    attack_type, 
                    model, 
                    model_trms,
                    surrogate_model,
                    surrogate_model_trms,
                    hpf_masker,
                    input_size,
                    *args, 
                    **kwargs):
        if attack_type == 'boundary_attack':
            attack = BoundaryAttack(model, *args, **kwargs)
        elif attack_type == 'hpf_boundary_attack':
            attack = HPFBoundaryAttack(model, hpf_masker, *args, **kwargs)
        elif attack_type == 'nes':
            attack = NESAttack(model, *args, **kwargs)
        elif attack_type == 'square_attack':
            attack = SquareAttack(model, *args, **kwargs)
        elif attack_type == 'hpf_square_attack':
            attack = HPFSquareAttack(model, hpf_masker, *args, **kwargs)
        elif attack_type == 'pg_rgf':
            attack = PriorRGFAttack(model=model,
                                    model_trms=model_trms, 
                                    surrogate_model=surrogate_model, 
                                    surrogate_model_trms=surrogate_model_trms,
                                    input_size=input_size,
                                    num_classes=1000,
                                    *args, 
                                    **kwargs)
        elif attack_type == 'hpf_pg_rgf':
            attack = HPFPriorRGFAttack(hpf_masker,
                                    model=model,
                                    model_trms=model_trms, 
                                    surrogate_model=surrogate_model, 
                                    surrogate_model_trms=surrogate_model_trms,
                                    input_size=input_size,
                                    num_classes=1000,
                                    *args, 
                                    **kwargs)
            
        return attack

    def get_l2(self, orig_img, adv_img):
        distance = (orig_img - adv_img).pow(2).sum().sqrt()
        return (distance / orig_img.max()).item()
    
    def __call__(self, x, target_y):
        orig_x = x
        x = self.to_float(x)
        perturbed_x, num_queries, mse = self.black_box_attack(x, target_y)
        perturbed_x = perturbed_x.squeeze(0).cpu()
        self.l2_norm.append(self.get_l2(orig_x.cpu(), (perturbed_x*255).to(torch.uint8)))
        self.num_queries_lst.append(num_queries)
        self.mse_list.append(mse)
        return perturbed_x
    
    def set_up_log(self, path_log_dir):
        self.path_log_dir = path_log_dir
    
    def get_attack_metrics(self):
        avg_num_queries = sum(self.num_queries_lst) / len(self.num_queries_lst)
        avg_mse = sum(self.mse_list) / len(self.mse_list)
        return avg_num_queries, avg_mse
        
        
class WhiteBoxAttack:
    
    def __init__(self, 
                attack_type, 
                surrogate_model, 
                device, 
                input_size, 
                dataset_type, 
                surrogate_model_trms, 
                hpf_masker,
                internal_jpeg_quality=80,
                jpeg_compr_obj=None,
                *args, 
                **kwargs):
        self.model = surrogate_model
        self.device = torch.device(device)
        surrogate_loss = torch.nn.CrossEntropyLoss()
        self.model_trms = surrogate_model_trms
        self.input_size = input_size
        self.dataset_type = dataset_type
        self.internal_compression_rate = internal_jpeg_quality
        self.attack = self.load_attack(self.model, attack_type, surrogate_loss, hpf_masker, *args, **kwargs)
        self.to_tensor = T.ConvertImageDtype(torch.float32)
        self.jpeg_compr_obj = jpeg_compr_obj
        if kwargs.get('protocol_file') is None:
            protocol_file = ''
        else:
            protocol_file = kwargs['protocol_file']
        self.image_metric = ImageQualityMetric(['mad', 'ssim', 'dists'], protocol_file)
        self.save_dir = f"./data/survey_data/{attack_type.split('_')[0]}/vanilla" if len(attack_type.split('_')) == 1 else f"./data/survey_data/{attack_type.split('_')[1]}/hpf"
        self.check_save_dir_path()
        self.orig_save_dir = "./data/survey_data/orig"
        self.l2_norm = []
        self.mad_score = []
        self.ssim_score = []
        self.dists_score = []
        self.n = 1
        if attack_type == 'rcw':
            self.call_fn = self.attack_rcw
        elif attack_type == 'far':
            self.call_fn = self.attack_far
        else:
            self.call_fn = self.attack_mc

            
    def __call__(self, x, y):
        x = self.call_fn(x, y)
        x = torch.clamp(x, min=0.0, max=1.0)
        return x

    def check_save_dir_path(self):
        dirs = []
        paths = self.save_dir.split('/')
        for i in range(len(paths)-2):
            dirs.append('/'.join(paths[:i+3]))
        for path in dirs:
            if not os.path.exists(path):
                os.mkdir(path)
    
    def set_report_dir(self, run_name):
        self.report_dir = run_name
        with open(self.report_dir + '/' + 'target_labels.csv', 'w') as target_labels_f:
            target_labels_obj = csv.writer(target_labels_f)
            target_labels_obj.writerow(['target labels'])
    
    def report_target_label(self, target_label):
        with open(self.report_dir + '/' + 'target_labels.csv', 'a') as target_labels_f:
            target_labels_obj = csv.writer(target_labels_f)
            for t in target_label:
                target_labels_obj.writerow([t.item()])
    
    def attack_rcw(self, x, y):
        with torch.enable_grad():
            x = self.to_tensor(x)
            orig_x = x.clone().detach()
            x = x.to(self.device)
            self.model.zero_grad()
            #y = torch.LongTensor([y])
            # issue here
            cqe_image = self.jpeg_compr_obj(x.to('cpu'))
            cqe_image = cqe_image.to(self.device)
            perturbed_x = self.attack(x, y, cqe_image=cqe_image)
            if isinstance(perturbed_x, tuple):
                perturbed_x, target_label = perturbed_x
                self.report_target_label(target_label)
            perturbed_x = perturbed_x.cpu()
            self.l2_norm.extend([l2.item() for l2 in self.get_l2(orig_x, perturbed_x, )])
            if self.image_metric is not None:
                mad_r, ssim_r, dists_r = self.image_metric(orig_x, perturbed_x)
                self.mad_score.extend(mad_r)
                self.ssim_score.extend(ssim_r)
                self.dists_score.extend(dists_r)
            #torchvision.utils.save_image(orig_x, f'{self.orig_save_dir}/{self.n}.png', format='PNG')
            #torchvision.utils.save_image(perturbed_x, f'{self.save_dir}/{self.n}.png', format='PNG')
            self.n += 1
        return perturbed_x
    
    def attack_mc(self, x, y):
        with torch.enable_grad():
            x = self.to_tensor(x)
            orig_x = x.clone().detach()
            x = x.to(self.device)
            self.model.zero_grad()
            perturbed_x = self.attack(x, y)
            if isinstance(perturbed_x, tuple):
                perturbed_x, target_label = perturbed_x
                self.report_target_label(target_label)
            perturbed_x = perturbed_x.cpu()
            # TODO: insert perturbation for FAR
            self.l2_norm.extend([l2.item() for l2 in self.get_l2(orig_x, perturbed_x, )])
            if self.image_metric is not None:
                mad_r, ssim_r, dists_r = self.image_metric(orig_x, perturbed_x)
                self.mad_score.extend(mad_r)
                self.ssim_score.extend(ssim_r)
                self.dists_score.extend(dists_r)
            #torchvision.utils.save_image(orig_x, f'{self.orig_save_dir}/{self.n}.png', format='PNG')
            #torchvision.utils.save_image(perturbed_x, f'{self.save_dir}/{self.n}.png', format='PNG')
            self.n += 1
        return perturbed_x

    def attack_far(self, x, y):
        with torch.enable_grad():
            x = self.to_tensor(x)
            orig_x = x.clone().detach()
            x = x.to(self.device)
            self.model.zero_grad()
            perturbed_x = self.attack(x, y)
            if isinstance(perturbed_x, tuple):
                perturbed_x, target_label = perturbed_x
                self.report_target_label(target_label)
            perturbed_x = perturbed_x.squeeze(0).cpu()
            orig_x = self.attack.to_jpeg(orig_x.detach(), quality=self.internal_compression_rate)
            self.l2_norm.extend([l2.item() for l2 in self.get_l2(orig_x, perturbed_x)])
            if self.image_metric is not None:
                mad_r, ssim_r, dists_r = self.image_metric(orig_x, perturbed_x)
                self.mad_score.extend(mad_r)
                self.ssim_score.extend(ssim_r)
                self.dists_score.extend(dists_r)
            #torchvision.utils.save_image(orig_x, f'{self.orig_save_dir}/{self.n}.png', format='PNG')
            #torchvision.utils.save_image(perturbed_x, f'{self.save_dir}/{self.n}.png', format='PNG')
            self.n += 1
        return perturbed_x
    
    def load_attack(self, model, attack_type, surrogate_loss, hpf_masker, *args, **kwargs):
        if attack_type in ['vmifgsm', 'hpf_vmifgsm', 'uvmifgsm', 'cvfgsm', 'mvmifgsm', 'varsinifgsm']:
            attack = self.load_blackbox(model, attack_type, surrogate_loss, hpf_masker, *args, **kwargs)
        else:
            attack = self.load_whitebox(model, attack_type, surrogate_loss, hpf_masker, *args, **kwargs)
        return attack
    
    def load_blackbox(self, model, attack_type, surrogate_loss, hpf_masker, *args, **kwargs):
        if attack_type == 'vmifgsm':
            attack = VMIFGSM(model, surrogate_loss, model_trms=self.model_trms, *args, **kwargs)
        elif attack_type == 'cvfgsm':
            attack = CVFGSM(model, surrogate_loss, model_trms=self.model_trms, *args, **kwargs)
        elif attack_type == 'uvmifgsm':
            attack = UVMIFGSM(model, surrogate_loss, model_trms=self.model_trms, *args, **kwargs)
        elif attack_type == 'hpf_vmifgsm':
            attack = torchattacks.attacks.vmifgsm.HpfVMIFGSM(model, surrogate_loss=surrogate_loss, model_trms=self.model_trms, hpf_masker=hpf_masker, *args, **kwargs)
        elif attack_type == 'mvmifgsm':
            attack = torchattacks.attacks.vmifgsm.MVMIFGSM(model, surrogate_loss, *args, **kwargs)
        elif attack_type == 'varsinifgsm':
            attack = torchattacks.attacks.sinifgsm.VarSINIFGSM(model, surrogate_loss, *args, **kwargs)
        else:
            raise ValueError('ADVERSARIAL ATTACK NOT RECOGNIZED FROM TYPE. Change spatial_adv_type in options')
        return attack        
    
    def load_whitebox(self, model, attack_type, surrogate_loss, hpf_masker, *args, **kwargs):
        if attack_type == 'fgsm':
            attack = FGSM(model, surrogate_loss, model_trms=self.model_trms, *args, **kwargs)
        elif attack_type == 'hpf_fgsm':
            attack = HpfFGSM(model, surrogate_loss=surrogate_loss, model_trms=self.model_trms, hpf_masker=hpf_masker, *args, **kwargs)
        elif attack_type == 'ycbcr_hpf_fgsm':
            attack = torchattacks.attacks.fgsm.YcbcrHpfFGSM(model, surrogate_loss=surrogate_loss, model_trms=self.model_trms, hpf_masker=hpf_masker, *args, **kwargs)
        elif attack_type == 'bim':
            attack = torchattacks.attacks.bim.BIM(model, surrogate_loss, model_trms=self.model_trms, *args, **kwargs)
        elif attack_type == 'jifgsm':
            attack = JIFGSM(model, surrogate_loss, model_trms=self.model_trms, *args, **kwargs)
        elif attack_type == 'grad_gaussian_bim':
            attack = torchattacks.attacks.bim.GradGaussianBIM(model, surrogate_loss, model_trms=self.model_trms, *args, **kwargs)
        elif attack_type == 'hpf_bim':
            attack = torchattacks.attacks.bim.HpfBIM(model, surrogate_loss=surrogate_loss, model_trms=self.model_trms, hpf_masker=hpf_masker, *args, **kwargs)
        elif attack_type == 'pgd':
            attack = PGD(model, surrogate_loss, model_trms=self.model_trms, *args, **kwargs)
        elif attack_type == 'unipgd':
            attack = UniPGD(model, surrogate_loss, model_trms=self.model_trms, *args, **kwargs)
        elif attack_type == 'hpf_pgd':
            attack = torchattacks.attacks.pgd.HpfPGD(model, surrogate_loss=surrogate_loss, model_trms=self.model_trms, hpf_masker=hpf_masker, *args, **kwargs)
        elif attack_type == 'pgdl2':
            attack = torchattacks.attacks.pgdl2.PGDL2(model, surrogate_loss, model_trms=self.model_trms, *args, **kwargs)
        elif attack_type == 'uap':
            attack = UAP(model, surrogate_loss, model_trms=self.model_trms, *args, **kwargs)
        elif attack_type == 'hpf_pgdl2':
            attack = torchattacks.attacks.pgdl2.HpfPGDL2(model, surrogate_loss=surrogate_loss, model_trms=self.model_trms, hpf_masker=hpf_masker, *args, **kwargs) #diff for 299(inception) and 224(rest)
        elif attack_type == 'ycbcr_hpf_pgdl2':
            attack = torchattacks.attacks.pgdl2.YcbcrHpfPGDL2(model, input_size=self.input_size, surrogate_loss=surrogate_loss, *args, **kwargs) #diff for 299(inception) and 224(rest)
        elif attack_type == 'sinifgsm':
            attack = torchattacks.attacks.sinifgsm.SINIFGSM(model, surrogate_loss, *args, **kwargs)
        elif attack_type == 'far':
            attack = FastAdversarialRounding(model, surrogate_loss=surrogate_loss, model_trms=self.model_trms, img_size=self.input_size, *args, **kwargs)
        elif attack_type == 'cw':
            attack = CW(model, model_trms=self.model_trms, *args, **kwargs)
        elif attack_type == 'rcw':
            if self.dataset_type == 'nips17':
                cqe_init = 'random'
            elif self.dataset_type == 'cifar10':
                cqe_init = 'center'
            attack = RCW(model, model_trms=self.model_trms, cqe_init=cqe_init, *args, **kwargs)
        elif attack_type == 'varrcw':
            attack = VarRCW(model, model_trms=self.model_trms, *args, **kwargs)
        elif attack_type == 'wrcw':
            attack = WRCW(model, model_trms=self.model_trms, *args, **kwargs)
        elif attack_type == 'ycw':
            attack = YCW(model, model_trms=self.model_trms, *args, **kwargs)
        elif attack_type == 'perccw':
            attack = PerC_CW(model, model_trms=self.model_trms, *args, **kwargs)
        elif attack_type == 'dctcw':
            attack = DCTCW(model, model_trms=self.model_trms, *args, **kwargs)
        else:
            raise ValueError('ADVERSARIAL ATTACK NOT RECOGNIZED FROM TYPE. Change spatial_adv_type in options')
        return attack

    def get_l2(self, orig_img, adv_img):
        if orig_img.max() > 1:
            raise ValueError('original image is not 0 < x < 1')
        if adv_img.max() > 1:
            raise ValueError('adv image is not 0 < x < 1')
        distance = (orig_img - adv_img).pow(2).sum(dim=(1,2,3)).sqrt()
        return distance / orig_img.max()
        
    # Deprecated
    def get_l2norm(self, orig_x, perturbed_x):
        orig_x = orig_x / 255
        perturbed_x = perturbed_x / 255
        
        return torch.linalg.norm(orig_x - perturbed_x)
        
    def get_avg_l2norm(self):
        return self.l2_norm / self.n
    
class Augmenter:
    
    def __init__(self, kernel_size=5, compression_rate=40):
        self.blur = T.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 3.0))
        self.compression_rate = compression_rate
        
    def __call__(self, img):
        img = self.blur(img)
        img = self.jpeg_compression(img)
        return img

    def jpeg_compression_for_img(self, img):
        adjusted_for_jpeg = False
        if img.dtype != torch.uint8:
            """ig_max, ig_min = img.max().item(), img.min().item()
            img = (img - ig_min) / (ig_max - ig_min)"""
            img = (img * 255).to(torch.uint8)
            adjusted_for_jpeg = True
        compressed = encode_jpeg(img, quality=self.compression_rate)
        compressed_img = decode_image(compressed)
        if adjusted_for_jpeg:
            compressed_img = compressed_img / 255
            #compressed_img = compressed_img*(ig_max-ig_min)+ig_min
        return compressed_img

    def jpeg_compression(self, imgs):
        igs = [self.jpeg_compression_for_img(img) for img in imgs]
        return torch.stack(igs)

class Patchify:
    
    def __init__(self, img_size, patch_size, n_channels):
        self.img_size = img_size
        self.n_patches = (img_size // patch_size) ** 2

        assert (img_size // patch_size) * patch_size == img_size
        
    def __call__(self, x):
        p = x.unfold(1, 8, 8).unfold(2, 8, 8).unfold(3, 8, 8) # x.size() -> (batch, model_dim, n_patches, n_patches)
        self.unfold_shape = p.size()
        p = p.contiguous().view(-1,8,8)
        return p
    
    def inverse(self, p):
        if not hasattr(self, 'unfold_shape'):
            raise AttributeError('Patchify needs to be applied to a tensor in ordfer to revert the process.')
        x = p.view(self.unfold_shape)
        output_h = self.patchify.unfold_shape[1] * self.patchify.unfold_shape[4]
        output_w = self.patchify.unfold_shape[2] * self.patchify.unfold_shape[5]
        x = x.permute(0,1,4,2,5,3).contiguous()
        x = x.view(3, output_h, output_w)
        return x 
        
        