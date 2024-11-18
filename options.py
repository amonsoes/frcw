import argparse
import sys
import numpy as np

from src.utils.argutils import str2dict_conv
from src.utils.argutils import set_up_args

def build_args():

    parser = argparse.ArgumentParser()

    # ========= SHARED OPTIONS =========

    #script options
    parser.add_argument('--optimization', type=lambda x: x in ['True', 'true', 'TRUE', '1', 'yes', 'y'], default=False, help='if True performs genetic algorithm')
    parser.add_argument('--device', type=str, default='cpu', help='set gpu or cpu')
    parser.add_argument('--optim', type=str, default='sgd', help='set to adam or sgd. sgd: sgd-nesterov, adam: radam')
    parser.add_argument('--log_result', type=lambda x: x in ['True', 'true', 'TRUE', '1', 'yes', 'y'], default=False, help='log result to wandb')

    #adversarial options
    parser.add_argument('--adversarial', type=lambda x: x in ['True', 'true', 'TRUE', '1', 'yes', 'y'], default=False, help='activate adversarial processing')
    parser.add_argument('--spatial_adv_type', type=str, default='fgsm', help='choose available spatial attack')
    parser.add_argument('--eps', type=float, default=0.04, help='set epsilon for attack boundary')
    parser.add_argument('--scale_cad_for_asp', type=lambda x: x in ['True', 'true', 'TRUE', '1', 'yes', 'y'], default=False, help='set if CAD should be scaled')
    parser.add_argument('--test_cas', type=lambda x: x in ['True', 'true', 'TRUE', '1', 'yes', 'y'], default=False, help='test the compression adaptation search')
    parser.add_argument('--q_search_type', type=str, default='cqe', help='set cqe search type for RCW')
    
    
    # adv: white-box
    parser.add_argument('--surrogate_model', type=str, default='resnet', help='set the Image Net model you want to transfer for FGSM')
    parser.add_argument('--surrogate_input_size', type=int, default=224, help='surrogate model input size')
    parser.add_argument('--surrogate_transform', type=str, default='pretrained', help='which surrogate transform to perform on imgs')
    parser.add_argument('--alpha', type=float, default=0.0, help='set alpha for step size of iterative fgsm methods')
    parser.add_argument('--log_mu', type=float, default=0.4, help='determines the weight of the LoG mask for the final HPF mask')
    parser.add_argument('--N', type=int, default=10, help='set number of samples to be drawn from the eps-neighborhood of adversary gradient for mean calc')
    parser.add_argument('--diagonal', type=int, default=-5, help='set how much low frequency information should be added to dct coefficient comutation. middle_ground:0, less:>0 more:<0')
    parser.add_argument('--lf_boosting', type=float, default=0.0, help='set lf_boosting to boost low frequencies by the amount in hpf settings')
    parser.add_argument('--mf_boosting', type=float, default=0.0, help='set mf_boosting to boost low frequencies by the amount in hpf settings')
    parser.add_argument('--use_sal_mask', type=lambda x: x in ['True', 'true', 'TRUE', '1', 'yes', 'y'], default=True, help='set to False to disable salient mask in HPF computation')
    parser.add_argument('--sal_mask_only', type=lambda x: x in ['True', 'true', 'TRUE', '1', 'yes', 'y'], default=False, help='set to False to disable salient mask in HPF computation')
    parser.add_argument('--hpf_mask_tau', type=float, default=0.7, help='set binary variable to define hpf mask and saliency mask tradeoff')
    parser.add_argument('--is_targeted', type=lambda x: x in ['True', 'true', 'TRUE', '1', 'yes', 'y'], default=False, help='activate/deactivate target mode')
    parser.add_argument('--uap_path', type=str, default='', help='define the path to the UAP')
    parser.add_argument('--init_method', type=str, default='', help='init method for attacks. One of uap/eps/he/xavier')
    parser.add_argument('--random_uap', type=lambda x: x in ['True', 'true', 'TRUE', '1', 'yes', 'y'], default=False, help='random init for uap attack')
    parser.add_argument('--jifgsm_compr_type', type=str, default='shin', help='one of ["shin","reich"]')
    parser.add_argument('--eta', type=float, default=0.6, help='set eta percentile in FAR to decide on the most important DCT coeffs')
    parser.add_argument('--far_jpeg_quality', type=int, default=70, help='set internal jpeg quality for fast adv rounding')
        
    # optimization-based attacks
    parser.add_argument('--c', type=float, default=1.0, help='tradeoff variable that balances delta minimization with adv loss')
    parser.add_argument('--kappa', type=float, default=0, help='high kappa ensures misclassification with high confidence')
    parser.add_argument('--steps', type=int, default=5000, help='set nr of optim steps. Also used in boundary attack')
    parser.add_argument('--attack_lr', type=float, default=0.0001, help='adjustment rate taken by optimizer (step size)')
    
    # AUC functionality
    parser.add_argument('--run_auc_test',type=lambda x: x in ['True', 'true', 'TRUE', '1', 'yes', 'y'], default=False, help='activates AUC computation for attacks and disables the normal run')
    
    # optim attacks test environments
    parser.add_argument('--run_cw_test',type=lambda x: x in ['True', 'true', 'TRUE', '1', 'yes', 'y'], default=False, help='activates CW test environment in run_pretrained.py and disables the normal run')
    parser.add_argument('--n_c', type=int, default=20, help='amount of c to be tested during the evaluation')
    parser.add_argument('--n_starts', type=int, default=2, help='n of random samples around x for optim')
    parser.add_argument('--n_datapoints', type=int, default=-1, help='n of samples from data for test. -1 equals to entire dataset size')
    parser.add_argument('--test_robustness',type=lambda x: x in ['True', 'true', 'TRUE', '1', 'yes', 'y'], default=False, help='activates CW test environment with compression')
    parser.add_argument('--verbose_cw', type=lambda x: x in ['True', 'true', 'TRUE', '1', 'yes', 'y'], default=False, help='activate/deactivate print statements in optim')
    parser.add_argument('--target_mode', type=str, default='random', help='set target mode to either "most_likely" or "least_likely". Defaults to "random"')
    parser.add_argument('--rcw_comp_lower_bound', type=int, default=75, help='set compression rate lower bound for rcw attack')
    parser.add_argument('--rcw_beta', type=float, default=1.0, help='set robustness loss')
    parser.add_argument('--iq_loss', type=str, default='l2', help='set iq loss for VarRCW attack. Set one of ["l2","ciede2000","hpf"]')
    parser.add_argument('--dct_type', type=str, default='full', help='set DCT type for dct cw. Set one of ["full","patched"]')
    parser.add_argument('--ablation',type=lambda x: x in ['True', 'true', 'TRUE', '1', 'yes', 'y'], default=False, help='activates VarRCW in ablation mode')
    
    
    # adv: compression and counter-compression
    parser.add_argument('--attack_compression', type=lambda x: x in ['True', 'true', 'TRUE', '1', 'yes', 'y'], default=False, help='activate JPEG compression on spatial attack')
    parser.add_argument('--attack_compression_rate', type=lambda x: [int(i) for i in x.split(',')], default=[40], help='set rates of JPEG compression on attack')
    # in the case of consecutive_attack_compr, attack_compression_rate should be list
    parser.add_argument('--consecutive_attack_compr', type=lambda x: x in ['True', 'true', 'TRUE', '1', 'yes', 'y'], default=False, help='applies consecutive compressions to image')
    
    parser.add_argument('--gaussian', type=lambda x: x in ['True', 'true', 'TRUE', '1', 'yes', 'y'], default=False, help='set to True if you want to apply gaussian after attack computation')
    parser.add_argument('--gauss_kernel', type=int, default=15, help='size of gaussian kernel. should be odd')
    parser.add_argument('--gauss_sigma', type=float, default=2.5, help='set sigma for gaussian kernel')
    
    # adv: black-box: boundary
    parser.add_argument('--max_queries', type=int, default=2, help='boundary attack, pg_rgf: set max queries')
    parser.add_argument('--p', type=str, default='l2', help='boundary/nes/pg_rgf: l-norm')
    parser.add_argument('--spherical_step', type=float, default=0.008, help='boundary attack: spherical step size')
    parser.add_argument('--source_step', type=float, default=0.0016, help='boundary attack: step size towards source')
    parser.add_argument('--source_step_convergence', type=float, default=0.000001, help='boundary attack: threshold for convergence (eps)')
    parser.add_argument('--step_adaptation', type=int, default=1000, help='boundary attack: if step size should be adapted')
    parser.add_argument('--update_stats_every_k', type=int, default=30, help='boundary attack: every k times epherical and source step are updated')
    
    # adv: black-box: PG-RGF
    # max_queries
    parser.add_argument('--samples_per_draw', type=int, default=10, help='pg-rgf: number of samples (rand vecs) to estimate the gradient.')
    parser.add_argument('--method', type=str, default='fixed_biased', help='pg-rgf: methods used in the attack. uniform: RGF, biased: P-RGF (\lambda^*), fixed_biased: P-RGF (\lambda=0.5)')
    parser.add_argument('--dataprior', type=lambda x: x in ['True', 'true', 'TRUE', '1', 'yes', 'y'], default=False, help='pg-rgf: whether to use data prior in the attack.')
    parser.add_argument('--sigma', type=float, default=0.0224, help='pg-rgf: float value sampling variance')
    parser.add_argument('--learning_rate', type=float, default=2.0, help='pg-rgf: adjustment rate of adversarial sample')
    
    # adv: black-box: NES
    parser.add_argument('--max_loss_queries', type=int, default=1000, help='nes attack: maximum nr of calls allowed to approx. grad ')
    parser.add_argument('--fd_eta', type=float, default=0.001, help='nes attack: step size of forward difference step')
    parser.add_argument('--nes_lr', type=float, default=0.005, help='nes attack: learning rate of NES step')
    parser.add_argument('--q', type=int, default=20, help='number of noise samples per NES step')
    
    # adv: black-box: SquareAttack
    parser.add_argument('--p_init', type=float, default=0.008, help='square attack: percentage of pixels to be attacked')
    
    # adv: spectral attack
    parser.add_argument('--power_dict_path', type=str, default='./src/adversarial/adv_resources/', help='set from where you want to load the power dict')
    parser.add_argument('--spectral_delta_path', type=str, default='./src/adversarial/adv_resources/', help='set from where you want to load the delta')

    # adv: adversarial training options
    parser.add_argument('--adversarial_training', type=lambda x: x in ['True', 'true', 'TRUE', '1', 'yes', 'y'], default=False, help='activate adversarial training')
    parser.add_argument('--adv_training_type', type=str, default='', help='set type of adv training')
    parser.add_argument('--attacks_for_training', type=lambda x: x.split(','), default='bim', help='comma-separate attack you want to use for training')
    parser.add_argument('--training_eps', type=float, default=8.0, help='set epsilon for attack computation during training')
        
    # transform options
    parser.add_argument('--cross_offset', type=lambda x: tuple(map(int, x.split(', '))), default=(0,0), help='which cross-band pixel correlation offset to use. enter in form x,x')
    parser.add_argument('--jpeg_compression', type=lambda x: x in ['True', 'true', 'TRUE', '1', 'yes', 'y'], default=False, help='activate JPEG compression')
    parser.add_argument('--jpeg_compression_rate', type=int, default=99, help='set rate of JPEG compression on attack')

    # Dataset options
    parser.add_argument('--dataset', type=str, default='140k_flickr_faces', help='which dataset to use')
    parser.add_argument('--greyscale_fourier', type=lambda x: x in ['True', 'true', 'TRUE', '1', 'yes', 'y'], default=False, help='Fourier transform with 1(True) channel or 3(False)')
    parser.add_argument('--greyscale_processing', type=lambda x: x in ['True', 'true', 'TRUE', '1', 'yes', 'y'], default=False, help='Image transform with 1(True) channel or 3(False)')
    

    filename = sys.argv[0].split('/')[-1]


    # ========= BASECNN OPTIONS =========


    if filename == 'run_basecnn.py':
        
        #basecnn options
        parser.add_argument('--pretrained', type=str, default='', help='set wether to load weights')
        parser.add_argument('--conv1', type=str2dict_conv, default={'size':5, 'out':17, 'pad':1, 'stride':1, 'dil':1}, help='set input, size, output, padding, stride and dilation in this order by comma-separated values')
        parser.add_argument('--conv2', type=str2dict_conv, default={'size':6, 'out':10, 'pad':4, 'stride':1, 'dil':1}, help='set input, size, output, padding, stride and dilation in this order by comma-separated values')
        parser.add_argument('--pool1', type=int, default=3, help='set pool2 size')
        parser.add_argument('--pool2', type=int, default=3, help='set pool1 size')
        parser.add_argument('--fc2', type=int, default=400, help='set size of dense layer')
        parser.add_argument('--dropout', type=float, default=0.4, help='set dropout')
        parser.add_argument('--batchsize', type=int, default=2, help='set batch size')
        parser.add_argument('--lr', type=float, default=0.0005, help='set learning rate')
        parser.add_argument('--epochs', type=int, default=7, help='set number of trainig epochs')
        parser.add_argument('--transform', type=str, default='pretrained', help='which transform to perform on imgs')
        parser.add_argument('--input_size', type=int, default=224, help='set input size for BaseCnn')


    # ========= PRETRAINED OPTIONS =========


    elif filename == 'run_pretrained.py':

        #transfer model options
        parser.add_argument('--model_name', type=str, default='resnet', help='set the Image Net model you want to transfer')
        parser.add_argument('--as_ft_extractor', type=lambda x: x in ['True', 'true', 'TRUE', '1', 'yes', 'y'], default=False, help='wether to freeze layers of transferred model')
        parser.add_argument('--pretrained', type=str, default='', help='set wether to load weights')
        parser.add_argument('--batchsize', type=int, default=8, help='set batch size')
        parser.add_argument('--lr', type=float, default=0.00005, help='set learning rate')
        parser.add_argument('--epochs', type=int, default=15, help='set number of trainig epochs')
        parser.add_argument('--model_out', type=str, default='pretrained', help='set the name of the output model')
        parser.add_argument('--transform', type=str, default='pretrained', help='which transform to perform on imgs')
        
        # adversarial pretrained options
        parser.add_argument('--adversarial_pretrained', type=lambda x: x in ['True', 'true', 'TRUE', '1', 'yes', 'y'], default=False, help='transfer adversarially pretrained')
        parser.add_argument('--adv_pretrained_protocol', type=str, default='fbf', help='transfer adversarially pretrained')


    args = parser.parse_args()
    args = set_up_args(args, filename)
    
    return args

args = build_args()

if __name__ == '__main__':
    pass
