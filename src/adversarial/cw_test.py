
import os
import torch
import csv
import matplotlib.pyplot as plt
import pandas as pd

from datetime import date
from src.adversarial.robust_cw import RCW, EnsembleRCW

class CWTestEnvironment:
    
    def __init__(self,
                cw_type,
                model, 
                optim_steps, 
                n_datapoints,
                target_mode,
                dataset_type,
                test_robustness,
                iq_loss='l2'):
        """
        Experiment environment for CW-based attacks. Runs CW subtype for
        a number of c, a number of random seeds around data point x, a 
        number of optim steps for a number of datapoints

        Args:
            cw_type (string): which cw type shall be tested
            n_c (int): number of c to be tested
            n_starts (int): n of random samples around x for optim
            optim_steps (int): n of steps by the optim
            n_datapoints (int): n of samples for test from the dataset
            target_mode (str): one of ('random', 'least_likely', 'most_likely')
            dataset_type (str): one of ('cifar10' or 'nips17')
            test_robustness (bool): if true tests on several JPEG settings
        """
        print('\nINFO: CWTestEnvironment is initialized so c and attack_lr arguments will be overwritten\n')
        self.model = model
        self.cw_type = cw_type
        self.c_list = torch.tensor([0.5, 1.0, 2.0, 3.5, 6.0])
        #self.attack_lr_lst = [0.00001, 0.001, 0.01]
        self.attack_lr_lst = [0.00001, 0.0001, 0.001, 0.01]
        self.n_starts = 1 #we want to find out how robust the algorithm is to different c so no 
        self.optim_steps = optim_steps
        self.n_datapoints = n_datapoints
        self.config_list = self.get_config_list()
        self.eps_for_division = 1e-07 # this is not the perturbation restriction but division-by-zero-eps
        self.target_mode = target_mode
        self.dataset_type = dataset_type
        self.test_robustness = test_robustness
        self.asp_list = []
        
        if test_robustness:
            # those determine, how much quality will be available after compression
            # i.e. 100 -> no quality loss
            # values below therefore test mild compression (as found in most apps today)
            self.compression_range = [[70], [80], [90]]
        
        if cw_type == 'rcw':
            self.cw_cls = RCW
        elif cw_type == 'ensemblercw':
            self.cw_cls = EnsembleRCW

            
        else:
            raise ValueError('cw_type not recognized!')
        
        report_base = './saves/reports/cw_reports'
        if not os.path.exists(report_base):
            os.mkdir(report_base)
        
        self.report_dir = report_base
        self.report_dir += '/'
        self.report_dir += date.today().isoformat() + '_'
        self.report_dir += self.cw_type  + '_' + str(optim_steps) + '_' + target_mode
        self.report_dir += '_' + f'compression:{test_robustness}'
        self.report_dir = self.resolve_name_collision(self.report_dir)
        os.mkdir(self.report_dir)
        if self.test_robustness:
            for compression_value in self.compression_range:
                os.mkdir(f'{self.report_dir}/compr-{compression_value}')
                main_losses_csv = f'{self.report_dir}/compr-{compression_value}/run_losses.csv'
                with open(main_losses_csv, 'a') as main_file:
                    main_obj = csv.writer(main_file)
                    main_obj.writerow(['c', 'attack_lr', 'avg_iq_loss', 'avg_f_loss', 'avg_cost', 'asr', 'cad', 'asp'])
                os.mkdir(f'{self.report_dir}/compr-{compression_value}/loss_reports')
            self.main_losses_csv = f'{self.report_dir}/compr-{self.compression_range[0]}/run_losses.csv'
            self.auc_metric = AUC(f'{self.report_dir}/compr-{self.compression_range[0]}/auc_result.txt')
        else:
            os.mkdir(f'{self.report_dir}/loss_reports/')
            self.main_losses_csv = f'{self.report_dir}/run_losses.csv'
            self.auc_metric = AUC(f'{self.report_dir}/auc_result.txt')
            with open(self.main_losses_csv, 'a') as main_file:
                main_obj = csv.writer(main_file)
                main_obj.writerow(['c', 'attack_lr', 'avg_iq_loss', 'avg_f_loss', 'avg_cost', 'asr', 'cad', 'asp'])
        
        with open(f'{self.report_dir}/run_params.txt', 'w') as f:
            f.write('CHOSEN PARAMS FOR RUN\n\n')
            f.write(f'cw_type : {cw_type}\n')
            f.write(f'optimizer steps per start : {optim_steps}\n')
            f.write(f'number of data points : {n_datapoints}\n')
            f.write(f'target_mode : {self.target_mode}\n')
            
            if cw_type == 'varrcw':
                f.write(f'constraint type : {self.iq_loss}\n')
        
        
        print('Running CW test with the following parameters:\n\n')
        print(f'\t c values: {self.c_list}')
        print(f'\t attack lrs: {self.attack_lr_lst}')
        print(f'\t random seeds per sample: {self.n_starts}')
        print(f'\t steps per seed: {self.optim_steps}')
        print(f'\t num of samples from dataset: {self.n_datapoints}')
        print(f'\t target_mode: {self.target_mode}')
        
    def reset(self):
        self.asp_list = []
    
    def get_compr_dir(self, compression_val):
        path_compr_dir = f'{self.report_dir}/compr-{compression_val}'
        self.main_losses_csv = path_compr_dir + '/' + 'run_losses.csv'
        auc_metric_target_path = path_compr_dir + '/' + 'auc_result.txt'
        self.auc_metric.update_target_path(auc_metric_target_path)
        return path_compr_dir
    
    def set_dataset_and_model_trms(self, data_obj):
        self.data = data_obj

    def resolve_name_collision(self, path):
        enum = 0
        ori_path = path
        while os.path.exists(path):
            enum += 1
            path = ori_path + '_' + str(enum)
        return path
    
    def get_config_list(self):
        # multiple restarts will be implemented in the CW class
        config_list = []
        for c in self.c_list:
            for attack_lr in self.attack_lr_lst:
                config_list.append([c, attack_lr])
        return config_list

    def write_to_protocol_dir(self, c, attack_lr, run_csv, compression, logger_run_file):
        total_iq_loss, total_f_loss, total_cost = 0.0, 0.0, 0.0
        n = 0
        with open(run_csv, 'r') as run_file:
            run_obj = csv.reader(run_file)
            next(run_obj)
            for row in run_obj:
                total_iq_loss += float(row[1])
                total_f_loss += float(row[2])
                total_cost += float(row[3])
                n += 1
        avg_iq_loss = total_iq_loss / (n + self.eps_for_division) 
        avg_f_loss = total_f_loss / (n + self.eps_for_division)
        avg_cost = total_cost / (n + self.eps_for_division)
        if isinstance(c, torch.Tensor):
            c = c.item()
        asr_run, cad_run, asp_run = self.get_asr_from_run(logger_run_file)
        with open(self.main_losses_csv, 'a') as main_file:
            main_obj = csv.writer(main_file)
            main_obj.writerow([c, attack_lr, avg_iq_loss, avg_f_loss, avg_cost, asr_run, cad_run, asp_run])
    
    def get_asr_from_run(self, run_dir):
        
        with open(run_dir + '/' + 'results.txt', 'r') as results_file:
            for line in results_file:
                if line.startswith('ASR'):
                    _, asr = line.strip().split(':')
                elif line.startswith('ConditionalAverageRate'):
                    _, cad = line.strip().split(':')
                elif line.startswith('ASP'):
                    _, asp = line.strip().split(':')

        return float(asr), float(cad), float(asp)
    
    def get_avg_asp_per_c(self):
        # this calculates the avg asp per c over all learning rates
        df = pd.read_csv(self.main_losses_csv)
        avg_asp_per_c = df.groupby('c')['asp'].mean()
        d_repr = avg_asp_per_c.to_dict().items()
        c_vals = [i[0] for i in d_repr]
        asp_vals = [i[1] for i in d_repr]
        target_path = '/'.join(self.main_losses_csv.split('/')[:-1]) + '/' + f'{self.cw_type}_c-asp_ROC.png'
        self.plot_roc(c_vals, asp_vals, target_path=target_path)
        return c_vals, asp_vals

        
    
    def plot_cw_test_results(self, compression_val=False):
        # plot 1
        c_list = []
        asr_list = []
        cad_list = []
        
        # plot 2
        attack_lr_list = []
        # asr
        # cad
        
        # plot 3
        # c_list
        avg_cost_list = []
        # asr
        
        # plot 4
        # attack_lr_list
        # avg_cost_list
        # asr
        
        with open(self.main_losses_csv, 'r') as main_losses:
            main_losses_obj = csv.reader(main_losses)
            next(main_losses_obj)
            for line in main_losses_obj:
                c, attack_lr, avg_iq_loss, avg_f_loss, avg_cost, asr, avg_l2, asp = line
                c_list.append(float(c))
                attack_lr_list.append(float(attack_lr))
                avg_cost_list.append(float(avg_cost))
                asr_list.append(float(asr))
                cad_list.append(float(avg_l2))
        
        fig, ax = plt.subplots(4, figsize=(8,16))
        ax[0].set_facecolor('#F7FCFC')
        ax[1].set_facecolor('#F7FCFC')
        ax[2].set_facecolor('#F7FCFC')
        ax[3].set_facecolor('#F7FCFC')
        color1 = 'tab:red'
        color2 = 'tab:blue'
        color3 = 'tab:cyan'
        plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)
        
        # plot 1
        # line 1 
        ax[0].set_xlabel('c')
        ax[0].set_ylabel('ASR', color=color2)
        line1, = ax[0].plot(c_list, asr_list, color=color2, label='ASR')
        ax[0].tick_params(axis='y', labelcolor=color2)
        
        ax1 = ax[0].twinx()
        
        # line 2
        ax1.set_ylabel('L2', color=color1)
        line2, = ax1.plot(c_list, cad_list,'--', color=color1, label='L2')
        ax1.tick_params(axis='y', labelcolor=color1)

        # plot 2
        # line 1 
        ax[1].set_xlabel('learning rate')
        ax[1].set_ylabel('ASR', color=color2)
        line3, = ax[1].plot(attack_lr_list, asr_list, color=color2, label='ASR')
        ax[1].tick_params(axis='y', labelcolor=color2)
        
        ax2 = ax[1].twinx()
        
        # line 2
        ax2.set_ylabel('L2', color=color1)
        line4, = ax2.plot(attack_lr_list, cad_list,'--', color=color1)
        ax2.tick_params(axis='y', labelcolor=color1)



        # plot 4
        # line 1 
        ax[3].set_xlabel('learning rate')
        ax[3].set_ylabel('avg cost')
        line6 = ax[3].plot(attack_lr_list, avg_cost_list, color=color3, label='cost')
        ax[3].tick_params(axis='y')
        
        lines = [line1, line2]
        labels = ['ASR', 'L2', 'Cost']
        fig.legend(lines, labels)
    
        plt.suptitle(f'CW Test for type:  {self.cw_type}', weight='bold')
        if compression_val:
            plt.savefig(f'{self.report_dir}/compr-{compression_val}_plots.png')
        else:    
            plt.savefig(f'{self.report_dir}/plots.png')
        #plt.show()

    def plot_roc(self, x_list, y_list, target_path=False):
    
        plt.plot(x_list, y_list)
        plt.xlabel('c')
        plt.ylabel('ASP')
        plt.title(f'ASP curve for type: {self.cw_type}', weight='bold')
        
        if target_path != False:
            plt.savefig(target_path)

if __name__ == '__main__':
    
    path = '/home/amon/git_repos/adv-attacks/saves/reports/cw_reports/2024-06-12_varrcw_3_1_2_most_likely_compression:False/run_losses.csv'
    
    df = pd.read_csv(path)
    print('done')
    
    