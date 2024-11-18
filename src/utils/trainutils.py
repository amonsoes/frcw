import torch
import os
import csv
import matplotlib.pyplot as plt

from datetime import date
from torch.optim import RAdam, SGD, Adam, lr_scheduler
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, Recall, Precision, AveragePrecision, F1Score
from src.adversarial.asr_metric import ASR, ConditionalAverageRate
from src.datasets.data import Nips17ImgNetData
from src.adversarial.robust_cw import RCW


def get_optim(model, optim_args):

    if optim_args['optim'] == 'sgdn':

        optim = SGD(model.parameters(), 
                    lr=optim_args['lr'], 
                    momentum=optim_args['momentum'], 
                    nesterov=optim_args['nesterov'], 
                    weight_decay=optim_args['weight_decay'])
    
    elif  optim_args['optim'] == 'radam':

        optim = RAdam(model.parameters(), 
                    lr=optim_args['lr'], 
                    betas=(0.9,0.999),
                    weight_decay=optim_args['weight_decay'])
    
    elif optim_args['optim'] == 'adam':
        
        optim = Adam(model.parameters(),
                    lr=optim_args['lr'],
                    weight_decay=optim_args['weight_decay'])
    
    elif optim_args['optim'] == 'sgd':
    
        optim = SGD(model.parameters(), 
                    lr=optim_args['lr'], 
                    weight_decay=optim_args['weight_decay'])

    else:
        raise ValueError('Wrong Input for optimizer')
    
    optim.optim_args = optim_args
    
    return optim


class TrainUtils:

    def __init__(self,
                opt,
                data,
                optim, 
                model_name, 
                model_type,
                lr,
                lr_gamma,
                epochs,
                num_classes,
                device,
                log_result,
                patience_scheduler=3, 
                patience_stopper=5,
                adversarial_opt=None,
                adversarial_training_opt=None):
        
        self.model_type = model_type
        self.run_name = None
        self.lr_gamma = lr_gamma
        self.transform_type = data.transform_type
        if isinstance(data, Nips17ImgNetData):
            self.analysis = 'imgnet_classification'
        else:
            raise ValueError('ERROR: unknown Dataset type. Add type to analysis type in trainutils.py')
        if adversarial_opt == None:
            adversarial = None
        else:
            adversarial = adversarial_opt.adversarial
        self.logger = Logger(opt,
                            model_name=model_name,
                            transform=self.transform_type,
                            lr=lr,
                            epochs=epochs,
                            batch_size=data.batch_size,
                            analysis_type=self.analysis,
                            adversarial_opt=adversarial_opt,
                            adversarial_training_opt=adversarial_training_opt,
                            log_result=log_result,
                            num_classes=num_classes)
        self.observed_schedule = [optim.param_groups[0]['lr']]
        self.scheduler = self.get_scheduler(optim=optim, patience=patience_scheduler)
        self.stopper = EarlyStopping(scheduler=self.scheduler, tolerance=patience_stopper)
        if num_classes == 1:
            self.metrics = MetricCollection([
                Accuracy(task='binary').to(device),
                Precision(task='binary', num_classes=num_classes, average='micro').to(device),
                Recall(task='binary', num_classes=num_classes, average='micro').to(device),
                AveragePrecision(task='binary', num_classes=num_classes, average='micro').to(device),
                F1Score(task='binary', num_classes=num_classes, average='micro').to(device),
            ])
        else:
            self.metrics = MetricCollection([
                Accuracy(task='multiclass', num_classes=num_classes).to(device),
                Precision(task='multiclass', num_classes=num_classes, average='macro').to(device),
                Recall(task='multiclass', num_classes=num_classes, average='macro').to(device),
                AveragePrecision(task='multiclass', num_classes=num_classes, average='macro').to(device),
                F1Score(task='multiclass', num_classes=num_classes, average='macro').to(device)
            ])
        if not os.path.exists('./saves/'):
            os.mkdir('./saves/')
            os.mkdir('./saves/models/')

    def get_scheduler(self, optim, patience):
        if optim.optim_args['scheduler'] == 'exp_lr':
            scheduler = lr_scheduler.ExponentialLR(optim, gamma=self.lr_gamma, verbose=True)
        elif optim.optim_args['scheduler'] == 'reduce_on_plateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optim, mode='min', patience=patience, verbose=True)
        return scheduler

    def plot_schedule(self):
        plt.plot(self.observed_schedule)
        plt.ylabel('lr per epoch')
        plt.show()
    
    def perform_lr_step(self, epoch_val_loss):
        if isinstance(self.scheduler, lr_scheduler.ExponentialLR):
            self.scheduler.step()
        else:
            self.scheduler.step(epoch_val_loss)

    def resolve_collision(self, path, run_name):
        enum = 1
        run_name_orig = run_name
        while os.path.exists(path+run_name):
            run_name = run_name_orig
            run_name = run_name + '_' + str(enum)
            enum += 1
        return run_name
    
    def save_model_shape(self, path, save_opt, result, is_pretraining_result=False):
        with open(f'{path}/model_params.txt', 'w') as f:
            f.write('CHOSEN PARAMS FOR MODEL\n\n')
            for k, v in save_opt.__dict__.items():
                f.write(f'{k}:{v}\n')
            f.write('\n\nRESULTS OBTAINED\n\n')
            for k, v in result.items():
                f.write(f'{k}:{v}\n')
        print('\nsaved model args.\n')
            
    def save_model(self, model, save_opt, result, optim=None):
        
        run_name = date.today().isoformat() 

        if self.model_type == 'pretrained':
            if save_opt.dataset_type == '140k_flickr_faces':
                if not os.path.exists('./saves/models/FlickrCNN/'):
                    os.mkdir('./saves/models/FlickrCNN/')
                self.run_name = self.resolve_collision('./saves/models/FlickrCNN/', run_name) if self.run_name == None else self.run_name
                save_path = f'./saves/models/FlickrCNN/{self.run_name}' 
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                self.save_model_shape(save_path, save_opt, result)
                torch.save(model.state_dict(), f'{save_path}/{save_opt.model_out}_{model.model_name}_{self.transform_type}_{self.analysis[:5]}.pt')
                print(f'\n\nsaved model at: {save_path}/{save_opt.model_out}_{model.model_name}_{self.transform_type}_{self.analysis[:5]}.pt\n\n')
            else:
                if not os.path.exists('./saves/models/ImgnetCNN/'):
                    os.mkdir('./saves/models/ImgnetCNN/')
                self.run_name = self.resolve_collision('./saves/models/ImgnetCNN/', run_name)  if self.run_name == None else self.run_name
                save_path = f'./saves/models/ImgnetCNN/{self.run_name}'
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                self.save_model_shape(save_path, save_opt, result)
                torch.save(model.state_dict(), f'{save_path}/{save_opt.model_out}_{model.model_name}_{self.transform_type}_{self.analysis[:5]}.pt')
                print(f'\n\nsaved model at: {save_path}/{save_opt.model_out}_{model.model_name}_{self.transform_type}_{self.analysis[:5]}.pt\n\n')

        elif self.model_type == 'bi_hpf':
            if not os.path.exists('./saves/models/BiHPF/'):
                os.mkdir('./saves/models/BiHPF/')
            self.run_name = self.resolve_collision('./saves/models/BiHPF/', run_name)  if self.run_name == None else self.run_name
            save_path = f'./saves/models/BiHPF/{self.run_name}'
            if not os.path.exists(save_path):
                    os.mkdir(save_path)
            self.save_model_shape(save_path, save_opt, result)
            torch.save(model.state_dict(), f'{save_path}/{save_opt.model_out}_{model.model_name}_{self.transform_type}_{self.analysis[:5]}.pt')
            print(f'\n\nsaved model at: {save_path}/{save_opt.model_out}_{model.model_name}_{self.transform_type}_{self.analysis[:5]}.pt\n\n')

        elif self.model_type == 'bi_attncnn':
            if not os.path.exists('./saves/models/BiAttnCNN/'):
                os.mkdir('./saves/models/BiAttnCNN/')
            self.run_name = self.resolve_collision('./saves/models/BiAttnCNN/', run_name)  if self.run_name == None else self.run_name
            save_path = f'./saves/models/BiAttnCNN/{self.run_name}'
            if not os.path.exists(save_path):
                    os.mkdir(save_path)
            self.save_model_shape(save_path, save_opt, result)
            torch.save(model.state_dict(), f'{save_path}/{save_opt.model_out}_{self.transform_type}_{self.analysis[:5]}.pt')
            print(f'\n\nsaved model at: {save_path}/{save_opt.model_out}_{self.transform_type}_{self.analysis[:5]}.pt\n\n')

        elif self.model_type == 'attncnn':
            if not os.path.exists('./saves/models/AttnCNN/'):
                os.mkdir('./saves/models/AttnCNN/')
            self.run_name = self.resolve_collision('./saves/models/AttnCNN/', run_name)  if self.run_name == None else self.run_name
            save_path = f'./saves/models/AttnCNN/{self.run_name}'
            if not os.path.exists(save_path):
                    os.mkdir(save_path)
            self.save_model_shape(save_path, save_opt, result)
            torch.save(model.state_dict(), f'{save_path}/{save_opt.model_out}_{self.transform_type}_{self.analysis[:5]}.pt')
            print(f'\n\nsaved model at: {save_path}/{save_opt.model_out}_{self.transform_type}_{self.analysis[:5]}.pt\n\n')

        elif self.model_type == 'synmixer_base':
            if not os.path.exists('./saves/models/SynMixer/base/'):
                os.mkdir('./saves/models/SynMixer/base/')
            self.run_name = self.resolve_collision('./saves/models/SynMixer/base/', run_name)  if self.run_name == None else self.run_name
            save_path = f'./saves/models/SynMixer/base/{self.run_name}'
            if not os.path.exists(save_path):
                    os.mkdir(save_path)
            self.save_model_shape(save_path, save_opt, result)
            torch.save(model.state_dict(), f'{save_path}/{save_opt.model_out}-{self.transform_type}-{self.analysis[:5]}.pt')
            print(f'\n\nsaved model at: {save_path}/{save_opt.model_out}-{self.transform_type}-{self.analysis[:5]}.pt\n\n')

        elif self.model_type == 'synmixer_target':
            if not os.path.exists('./saves/models/SynMixer/target/'):
                os.mkdir('./saves/models/SynMixer/target/')
            self.run_name = self.resolve_collision('./saves/models/SynMixer/target/', run_name)  if self.run_name == None else self.run_name
            save_path = f'./saves/models/SynMixer/target/{self.run_name}'
            if not os.path.exists(save_path):
                    os.mkdir(save_path)
            self.save_model_shape(save_path, save_opt, result)
            torch.save(model.state_dict(), f'{save_path}/{save_opt.model_out}.pt')
            print(f'\n\nsaved model at: {save_path}/{save_opt.model_out}.pt\n\n')

        elif self.model_type == 'coatnet':
            if not os.path.exists('./saves/models/CoAtNet/'):
                os.mkdir('./saves/models/CoAtNet/')
            self.run_name = self.resolve_collision('./saves/models/CoAtNet/', run_name)  if self.run_name == None else self.run_name
            save_path = f'./saves/models/CoAtNet/{self.run_name}'
            if not os.path.exists(save_path):
                    os.mkdir(save_path)
            self.save_model_shape(save_path, save_opt, result)
            torch.save(model.state_dict(), f'{save_path}/{save_opt.model_out}_{self.transform_type}_{self.analysis[:5]}.pt')
            print(f'\n\nsaved model at: {save_path}/{save_opt.model_out}_{self.transform_type}_{self.analysis[:5]}.pt\n\n')

        elif self.model_type == 'bi_coatnet':
            if not os.path.exists('./saves/models/BiCoAtNet/'):
                os.mkdir('./saves/models/BiCoAtNet/')
            self.run_name = self.resolve_collision('./saves/models/BiCoAtNet/', run_name)  if self.run_name == None else self.run_name
            save_path = f'./saves/models/BiCoAtNet/{self.run_name}' 
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            self.save_model_shape(save_path, save_opt, result)
            torch.save(model.state_dict(), f'{save_path}/{save_opt.model_out}_{self.transform_type}_{self.analysis[:5]}.pt')
            print(f'\n\nsaved model at: {save_path}/{save_opt.model_out}_{self.transform_type}_{self.analysis[:5]}.pt\n\n')

        elif self.model_type == 'pretraining':
            if not os.path.exists('./saves/pretraining'):
                os.mkdir('./saves/pretraining/')
            if not os.path.exists('./saves/pretraining/pt_models/'):
                os.mkdir('./saves/pretraining/pt_models/')
            if not os.path.exists(f'./saves/pretraining/pt_models/{self.model_type}'):
                os.mkdir(f'./saves/pretraining/pt_models/{self.model_type}')
            self.run_name = self.resolve_collision('./saves/pretraining/pt_models/', run_name)  if self.run_name == None else self.run_name
            save_path = f'../saves/pretraining/pt_models/{self.run_name}' 
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            self.save_model_shape(save_path, save_opt, result, is_pretraining_result=True)
            torch.save(model.state_dict(), f'{save_path}/{save_opt.model_out}_{self.transform_type}_{self.analysis[:5]}.pt')
            torch.save(optim.state_dict(), f'{save_path}/{save_opt.model_out}_{self.transform_type}_{self.analysis[:5]}.pt')
            print(f'\n\nsaved model at: {save_path}/{save_opt.model_out}_{self.transform_type}_{self.analysis[:5]}.pt\n\n')
            
        else:
            if self.transform_type == 'band_cooccurence':
                if not os.path.exists('./saves/models/CooccurrenceCNN/'):
                    os.mkdir('./saves/models/CooccurrenceCNN/')
                self.run_name = self.resolve_collision('./saves/models/CooccurrenceCNN/', run_name)  if self.run_name == None else self.run_name
                save_path = f'./saves/models/CooccurrenceCNN/{self.run_name}'
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                self.save_model_shape(save_path, save_opt, result)
                model.to_state_dict(f'{save_path}/{save_opt.model_out}_{self.analysis}.pt', self.transform_type, self.analysis)
            elif self.transform_type in ['real_nd_fourier', 'augmented_nd_fourier']:
                if not os.path.exists('./saves/models/SpectralCNN/'):
                    os.mkdir('./saves/models/SpectralCNN/')
                self.run_name = self.resolve_collision('./saves/models/SpectralCNN/', run_name)  if self.run_name == None else self.run_name
                save_path = f'./saves/models/SpectralCNN/{self.run_name}'
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                self.save_model_shape(save_path, save_opt, result)
                model.to_state_dict(f'{save_path}/{save_opt.model_out}_{self.analysis}.pt', self.transform_type, self.analysis)
            else:
                if not os.path.exists('./saves/models/BaseCNN/'):
                    os.mkdir('./saves/models/BaseCNN/')
                self.run_name = self.resolve_collision('./saves/models/BaseCNN/', run_name)  if self.run_name == None else self.run_name
                save_path = f'./saves/models/BaseCNN/{self.run_name}' 
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                self.save_model_shape(save_path, save_opt, result)
                model.to_state_dict(f'{save_path}/{save_opt.model_out}_{self.analysis}.pt', self.transform_type, self.analysis)
    
    def save_checkpoint(self, model, save_opt, result, optim, e):
        save_path = f'./saves/pretraining/pt_models/{self.model_type}/{self.run_name}/checkpoints/'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        torch.save({
            'epoch' : e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict()
        }, save_path)
        self.save_model_shape(save_path, save_opt, result)
    
    def load_checkpoint(self, save_path, model, optim, rank):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['model_state_dict'], map_location=map_location)
        optim.load_state_dict(checkpoint['optimizer_state_dict'], map_location=map_location)
        e = checkpoint['epoch']
        model.train()
        return model, optim, e
        

class Logger:

    def __init__(self,
                opt,
                model_name, 
                transform, 
                lr, 
                epochs, 
                batch_size, 
                analysis_type, 
                adversarial_opt,
                adversarial_training_opt,
                log_result,
                num_classes,
                create_log_file=True):
        self.opt = opt
        self.analysis = analysis_type
        self.log_result = log_result
        self.adversarial_opt = adversarial_opt
        self.adversarial_training_opt = adversarial_training_opt
        self.create_log_file = create_log_file
        self.model_name = model_name
        self.transform = transform
        self.lr = lr
        self.report_fn = self.write_to_report_binary if num_classes == 1 else self.write_to_report_mc 
        
        if create_log_file:
            if not os.path.exists(f'./saves/reports/{self.analysis}'):
                os.mkdir(f'./saves/reports/{self.analysis}')
            self.report_dir = f'./saves/reports/{self.analysis}'
            self.mk_report_dir(epochs, batch_size)
            
    def resolve_name_collision(self, path):
        enum = 0
        ori_path = path
        while os.path.exists(path):
            enum += 1
            path = ori_path + '_' + str(enum)
        return path
    
    def mk_report_dir(self, epochs, batch_size):
        
        run_name = date.today().isoformat()
        run_name += '_' + self.model_name
        if not self.adversarial_opt.adversarial:
            run_name += '_' + 'base'
        else:
            self.base_run_name = "_".join(run_name.split("_")[1:]) + '_' + 'base'
            if self.adversarial_opt.spatial_adv_type in ['cw',
                                                         'ssimcw',
                                                         'psnrcw',
                                                         'madcw',
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
                                                         'dctcw']:    
                run_name += '_' + self.adversarial_opt.spatial_adv_type + '_'+ str(self.adversarial_opt.spatial_attack_params.c) +'_' + str(self.adversarial_opt.spatial_attack_params.attack_lr)
            elif self.adversarial_opt.spatial_adv_type in ['uap']:
                run_name += '_' + self.adversarial_opt.spatial_adv_type + '_' + str(self.adversarial_opt.spatial_attack_params.eps)
            else:
                run_name += '_' + self.adversarial_opt.spatial_adv_type + '_' + str(self.adversarial_opt.spatial_attack_params.eps)
        run_name = self.resolve_name_collision(f'{self.report_dir}/{run_name}')
        os.mkdir(run_name)
        self.run_name = run_name
        
        with open(f'{run_name}/run_params.txt', 'w') as f:
            f.write('CHOSEN PARAMS FOR RUN\n\n')
            f.write(f'analysis_type : {self.analysis}\n')
            f.write(f'model_name : {self.model_name}\n')
            f.write(f'transform : {self.transform}\n')
            f.write(f'lr : {self.lr}\n')
            f.write(f'epochs : {epochs}\n')
            f.write(f'batch_size : {batch_size}\n')
            f.write(f'adversarial_training : {self.adversarial_training_opt.adversarial_training}\n')
            f.write(f'jpeg compression: {self.opt.jpeg_compression}\n')
            f.write(f'jpeg compression rate: {self.opt.jpeg_compression_rate}\n')
            if self.adversarial_training_opt.adversarial_training:
                f.write(f'adv_training_type : {self.adversarial_training_opt.adv_training_type}\n')
                f.write(f'attacks for training : {self.adversarial_training_opt.attacks_for_training}\n')
            if self.adversarial_opt.adversarial:
                f.write(f'adversarial : {self.adversarial_opt.adversarial}\n')
                f.write(f'adversarial_model : {self.adversarial_opt.spatial_adv_type}\n')
                f.write(f'surrogate_model : {self.adversarial_opt.surrogate_model_params.surrogate_model}\n')
                f.write(f'eps : {self.adversarial_opt.spatial_attack_params.eps}\n')
                if self.adversarial_opt.spatial_adv_type == 'jifgsm':
                    f.write(f'diff jpeg type : {self.adversarial_opt.spatial_attack_params.jifgsm_compr_type}\n')
                if self.adversarial_opt.spatial_adv_type in ['cw',
                                                             'ssimcw',
                                                             'psnrcw',
                                                             'madcw',
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
                                                             'dctcw']:
                    f.write(f'attack_lr : {self.adversarial_opt.spatial_attack_params.attack_lr}\n')
                    if self.adversarial_opt.spatial_adv_type in ['rcw', 'wrcw', 'ycw', 'varrcw', 'ercw']:
                        f.write(f'rcw_comp_lower_bound : {self.adversarial_opt.spatial_attack_params.rcw_comp_lower_bound}\n')
                    if self.adversarial_opt.spatial_adv_type in ['varrcw', 'rcw']:
                        f.write(f'ablation : {self.adversarial_opt.spatial_attack_params.ablation}\n')
                        if self.adversarial_opt.spatial_adv_type == 'varrcw':
                            f.write(f'constraint type : {self.adversarial_opt.spatial_attack_params.iq_loss}\n')
                elif self.adversarial_opt.spatial_adv_type in ['uap']:
                    f.write(f'uap_path : {self.adversarial_opt.spatial_attack_params.uap_path}')
                elif self.adversarial_opt.spatial_adv_type in ['unipgd', 'uvmifgsm']:
                    f.write(f'uap_path : {self.adversarial_opt.spatial_attack_params.uap_path}\n')
                    f.write(f'init_method : {self.adversarial_opt.spatial_attack_params.init_method}\n')
                f.write(f'attack_compression : {self.adversarial_opt.attack_compression}\n')
                f.write(f'compression_rate : {self.adversarial_opt.compression_rate[0]}\n')
                if self.adversarial_opt.spatial_attack_params.hpf_mask_params != None:
                    f.write(f'use_sal_mask : {self.adversarial_opt.spatial_attack_params.hpf_mask_params["use_sal_mask"]}\n')
                    f.write(f'sal_mask_only : {self.adversarial_opt.spatial_attack_params.hpf_mask_params["sal_mask_only"]}\n')
                    f.write(f'lf_boosting : {self.adversarial_opt.spatial_attack_params.hpf_mask_params["lf_boosting"]}\n')
                if hasattr(self.adversarial_opt.spatial_attack_params, 'gaussian'):
                    f.write(f'gaussian : {self.adversarial_opt.spatial_attack_params.gaussian}\n')
                    if self.adversarial_opt.spatial_attack_params.gaussian:
                        f.write(f'kernel_size : {self.adversarial_opt.spatial_attack_params.gauss_kernel}\n')
                        f.write(f'gaussian_sigma : {self.adversarial_opt.spatial_attack_params.gauss_sigma}\n')
                if hasattr(self.adversarial_opt.spatial_attack_params, 'alpha'):
                    f.write(f'alpha : {self.adversarial_opt.spatial_attack_params.alpha}\n')

        with open(f'{self.run_name}/report.csv', 'a') as report_file:
            report_obj = csv.writer(report_file)
            report_obj.writerow(['filepath', 'prediction', 'label', 'mad', 'ssim', 'dists', 'l2'])
            
        print('\nsaved run args.\n')
        
        
    def write_to_report(self, filenames, class_results, ground_truth, mad_scores=None, ssim_scores=None, l2_norms=None):
        return self.report_fn(filenames, class_results, ground_truth, mad_scores, ssim_scores, l2_norms)

    def write_to_report_binary(self, filenames, class_results, ground_truth, l2_norms):
        if l2_norms == None or not l2_norms:
            l2_norms = [0.0] * len(filenames)
        class_results = torch.sigmoid(class_results) > 0.5
        class_results = class_results.T.flatten().int()
        with open(f'{self.run_name}/report.csv', 'a') as report_file:
            report_obj = csv.writer(report_file)
            for f_n, c_r, g_t, l2_norm in zip(filenames, class_results.tolist(),  ground_truth.tolist(), l2_norms):
                report_obj.writerow([f_n, c_r, g_t, l2_norm])

    def write_to_report_mc(self, filenames, class_results, ground_truth, mad_scores, ssim_scores, l2_norms):
        if l2_norms == None or not l2_norms:
            l2_norms = [0.0] * len(filenames)
        if mad_scores == None or not mad_scores:
            mad_scores = [0.0] * len(filenames)
        if ssim_scores == None or not ssim_scores:
            ssim_scores = [0.0] * len(filenames)
        
        class_results = torch.softmax(class_results, dim=1).argmax(dim=1)
        with open(f'{self.run_name}/report.csv', 'a') as report_file:
            report_obj = csv.writer(report_file)
            for f_n, c_r, g_t, mad_score, ssim_score, l2_norm in zip(filenames, class_results.tolist(),  ground_truth.tolist(), mad_scores, ssim_scores, l2_norms):
                report_obj.writerow([f_n, c_r, g_t, mad_score, ssim_score, l2_norm])
    
    def log_black_box_metrics(self, attack_obj):
        avg_queries, avg_mse = attack_obj.get_attack_metrics()
        with open(f'{self.run_name}/black_box_metrics.txt', 'a') as report_file:
            report_file.write('BLACK BOX METRICS\n\n')
            report_file.write(f'AVG QUERIES : {avg_queries}\n')
            report_file.write(f'AVG MSE : {avg_mse}\n')
    
    def log_iqa(self, attack_obj):
        avg_mad = attack_obj.image_metric.get_avg_mad()
        avg_ssim = attack_obj.image_metric.get_avg_ssim()
        avg_dists = attack_obj.image_metric.get_avg_dists()
        with open(f'{self.run_name}/image_quality_metrics.txt', 'a') as report_file:
            report_file.write('IMAGE QUALITY METRICS\n\n')
            report_file.write(f'Average MAD : {avg_mad}\n')
            report_file.write(f'Average SSIM : {avg_ssim}\n')
            report_file.write(f'Average DISTS : {avg_dists}\n')
        if isinstance(attack_obj.attack, RCW):
            attack_obj.attack.log_cqe_results(f'{self.run_name}/cqe_results.txt')

    def log_eval_results(self, epoch_train_loss, epoch_val_loss, epoch_result):
        pass

    def log_test_results(self, result, is_targeted=False, hfm=None):
        
        if self.create_log_file:
            print(result)
            
            # Add ASR computation here
            with open(f'{self.run_name}/results.txt', 'w') as f:
                f.write('RESULTS\n\n')
                for k in result:
                    f.write(f'{k}:{result[k]}\n')
                if self.adversarial_opt.adversarial:
                    base_path = "/".join(self.run_name.split('/')[:-1])
                    asr_metric = ASR(self.run_name, f'{base_path}/{self.base_run_name}', is_targeted)
                    cad_metric = ConditionalAverageRate(self.run_name, f'{base_path}/{self.base_run_name}', scale_for_asp=self.adversarial_opt.scale_cad)
                    #spr_metric = SPR(self.run_name, f'{base_path}/{self.base_run_name}')
                    asr_r = asr_metric()
                    cad_r = cad_metric()
                    #spr_r = spr_metric(asr_r)
                    f.write(f'ASR:{asr_r}\n')
                    f.write(f'ConditionalAverageRate:{cad_r}\n')
                    f.write(f'ASP:{asr_r / (cad_r+ 0.00000001)}\n')
                    #f.write(f'SPR:{spr_r}\n')
                    
                    if hfm != None:
                        f.write(f'HFM:{hfm}\n')
                    
                
    def exit(self):
        if self.log_result:
            pass


class EarlyStopping:

    # call after validation but before scheduler

    def __init__(self, scheduler, tolerance=5):
        self.tolerance = tolerance
        self.min_val_loss = torch.inf
        self.counter = 0
        self.early_stop = False
        self.init_lr = scheduler.optimizer.param_groups[0]['lr']
    
    def __call__(self, validation_loss, scheduler):
        if validation_loss < self.min_val_loss:
            self.min_val_loss = validation_loss
            self.counter = 0
            self.init_lr = scheduler.optimizer.param_groups[0]['lr']
        elif validation_loss >=  self.min_val_loss:
            self.counter +=1
            if self.counter >= self.tolerance and self.check_lr_reduced(scheduler):
                print('\n\nINFO: \nInduced EARLY STOPPING due to validation loss not improving\n\n')
                print('\nstopping training early...\n\n')
                return True
        return False 

    def check_lr_reduced(self, scheduler):
        if self.init_lr != scheduler.optimizer.param_groups[0]['lr']:
            return True
        return False


if __name__ == '__main__':
    pass
        

                    