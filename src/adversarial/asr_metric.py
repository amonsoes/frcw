import csv
import math
import torch

class ASR:
    
    def __init__(self, path, basepath, is_targeted=False):
        if is_targeted:
            self.target_report = path + '/' + 'target_labels.csv'
        used_attack_compression, compression_value = self.check_for_compression(path)
        self.path = path + '/' + 'report.csv'
        if used_attack_compression:
            basepath += f'_compr-{compression_value}'
        self.base_path = basepath + '/' + 'report.csv'
        self.n = 0
        self.success = 0
        self.eps = 1e-7
        self.call_fn = self.compute_targeted if is_targeted else self.compute_untargeted

    def check_len(self, results_f, base_f):
        with open(self.path, 'r') as results_f:
            with open(self.base_path) as base_f:
                results_obj = csv.reader(results_f)
                base_obj = csv.reader(base_f)
                row_count_results = sum(1 for row in results_obj)
                row_count_base = sum(1 for row in base_obj)
                if row_count_base != row_count_results:
                    print(f'####\nWARNING: reports have different lenght base:{row_count_base} report:{row_count_results}\n####')
        return row_count_results
        
    def check_for_compression(self, path):
        used_attack_compression = False
        compression_value = 0
        with open(path + '/' + 'run_params.txt', 'r') as run_params_f:
            next(run_params_f)
            for line in run_params_f:
                if line != '\n':
                    param, value = line.split(':')
                    if param.strip() == 'attack_compression' and value.strip() == 'True':
                        used_attack_compression = True
                    elif param.strip() == 'compression_rate':
                        compression_value = value.strip()
        return used_attack_compression, compression_value
                
    def __call__(self):
        asr = self.call_fn()
        return asr
    
    def compute_untargeted(self):
        with open(self.path, 'r') as results_f:
            with open(self.base_path, 'r') as base_f:
                row_count_results = self.check_len(results_f, base_f)
                results_obj = csv.reader(results_f)
                base_obj = csv.reader(base_f)
                next(results_obj)
                next(base_f)
                for r_line, b_line in zip(results_obj, base_obj):
                    if b_line[1] == b_line[2]: # check if base model predicted correctly
                        if r_line[-1] != '0.0': # check if an attack actually happened 
                            if r_line[1] != r_line[2]: # check if adv model forced misclassification
                                self.success +=1
                            self.n += 1
                return self.success / (self.n + self.eps)
        
    def compute_targeted(self):
        with open(self.path, 'r') as results_f:
            with open(self.base_path, 'r') as base_f:
                with open(self.target_report, 'r') as target_f:
                    row_count_results = self.check_len(results_f, base_f)
                    results_obj = csv.reader(results_f)
                    base_obj = csv.reader(base_f)
                    targets_obj = csv.reader(target_f)
                    next(results_obj)
                    next(base_f)
                    next(targets_obj)
                    for r_line, b_line, t_line in zip(results_obj, base_obj, targets_obj):
                        if b_line[1] == b_line[2]: # check if base model predicted correctly
                            if r_line[-1] != '0.0': # check if an attack actually happened 
                                if r_line[1] != r_line[2]: # check if adv model forced misclassification
                                    if r_line[1] == t_line[0]: # check if model predicted target class from attack
                                        self.success += 1
                                self.n += 1
                    return self.success / (self.n + self.eps)
        

"""class ConditionalAverageRate:
    
    def __init__(self, path, basepath):
        self.path = path + '/' + 'report.csv'
        self.base_path = basepath + '/' + 'report.csv'
        self.acc_dist = 0.0
        self.n = 0
        self.eps = 0.000001
    
    def check_len(self, results_f, base_f):
        with open(self.path, 'r') as results_f:
            with open(self.base_path) as base_f:
                results_obj = csv.reader(results_f)
                base_obj = csv.reader(base_f)
                row_count_results = sum(1 for row in results_obj)
                row_count_base = sum(1 for row in base_obj)
                if row_count_base != row_count_results:
                    print(f'####\nWARNING: reports have different lenght base:{row_count_base} report:{row_count_results}\n####')
        
    def __call__(self):
        with open(self.path, 'r') as results_f:
            with open(self.base_path) as base_f:
                self.check_len(results_f, base_f)
                results_obj = csv.reader(results_f)
                base_obj = csv.reader(base_f)
                self.check_len(results_obj, base_obj)
                next(results_obj)
                next(base_f)
                for r_line, b_line in zip(results_obj, base_obj):
                    if r_line[-1] != '0.0': # check if adv attack was applied
                        if b_line[1] == b_line[2]: # check if base model predicted correctly
                            if r_line[1] != r_line[2]: # check if adv model forced misclassification
                                self.acc_dist += float(r_line[-1])
                                self.n += 1
                return self.acc_dist / (self.n + self.eps)"""


class ConditionalAverageRate:
    
    def __init__(self, path, basepath, eps=None, scale_for_asp=False):
        self.path = path + '/' + 'report.csv'
        self.base_path = basepath + '/' + 'report.csv'
        used_attack_compression, compression_value = self.check_for_compression(path)
        if used_attack_compression:
            basepath += f'_compr-{compression_value}'
        self.acc_dist = 0.0
        self.n = 0
        if eps == None:
            eps, dim = self.get_epsilon(path)
        else:
            _, dim = self.get_epsilon(path)
        self.scale_for_asp = scale_for_asp
        delta_eps = torch.full((3, dim, dim), eps)
        self.max_pert_dist = delta_eps.pow(2).sum().sqrt()
        self.mu = 0.000001
        
    
    def get_epsilon(self, path):
        dim = 224 # a std value to avoid chrashing
        with open(path + '/' + 'run_params.txt', 'r') as f:
            for line in f.readlines():
                if line.startswith('eps'):
                    _, value = line.strip().split(':')
                    eps = value
                elif line.startswith('surrogate_model'):
                    _, value = line.strip().split(':')
                    if value == 'resnet':
                        dim = 224
                    elif value == 'inception':
                        dim = 299
                
        return float(value), dim

    def check_for_compression(self, path):
        used_attack_compression = False
        compression_value = 0
        with open(path + '/' + 'run_params.txt', 'r') as run_params_f:
            next(run_params_f)
            for line in run_params_f:
                if line != '\n':
                    param, value = line.split(':')
                    if param.strip() == 'attack_compression' and value.strip() == 'True':
                        used_attack_compression = True
                    elif param.strip() == 'compression_rate':
                        compression_value = value.strip()
        return used_attack_compression, compression_value
    
    def check_len(self, results_f, base_f):
        with open(self.path, 'r') as results_f:
            with open(self.base_path) as base_f:
                results_obj = csv.reader(results_f)
                base_obj = csv.reader(base_f)
                row_count_results = sum(1 for row in results_obj)
                row_count_base = sum(1 for row in base_obj)
                if row_count_base != row_count_results:
                    print(f'####\nWARNING: reports have different lenght base:{row_count_base} report:{row_count_results}\n####')
        
    def __call__(self):
        with open(self.path, 'r') as results_f:
            with open(self.base_path) as base_f:
                self.check_len(results_f, base_f)
                results_obj = csv.reader(results_f)
                base_obj = csv.reader(base_f)
                self.check_len(results_obj, base_obj)
                next(results_obj)
                next(base_f)
                for r_line, b_line in zip(results_obj, base_obj):
                    if r_line[-1] != '0.0': # check if adv attack was applied
                        if b_line[1] == b_line[2]: # check if base model predicted correctly
                            if r_line[1] != r_line[2]: # check if adv model forced misclassification
                                self.acc_dist += (float(r_line[-1]) / self.max_pert_dist).item() if self.scale_for_asp else float(r_line[-1])
                                self.n += 1
                return self.acc_dist / (self.n + self.mu)


class SPR:
    
    def __init__(self, path, basepath):
        """
        ASR / tanh(10*eps*AvgMAD)

        Args:
            path (str)
            basepath (str)
            
        Extracts the average MAD by going over the report CSV.
        Calculates SPR based on above Formula
        """
        self.path = path + '/' + 'report.csv'
        self.base_path = basepath + '/' + 'report.csv'
        self.mad = 0.0
        self.n = 0
        self.eps, _ = self.get_epsilon(path)
        self.mu = 0.000001
        
    def get_epsilon(self, path):
        dim = 224 # a std value to avoid chrashing
        with open(path + '/' + 'run_params.txt', 'r') as f:
            for line in f.readlines():
                if line.startswith('eps'):
                    _, value = line.strip().split(':')
                    eps = value
                elif line.startswith('surrogate_model'):
                    _, value = line.strip().split(':')
                    if value == 'resnet':
                        dim = 224
                    elif value == 'inception':
                        dim = 299
        
        if path.split('/')[-1].split('_')[1].startswith('CIFARCNN'):
            dim = 32        
        return float(value), dim

    def check_len(self, results_f, base_f):
        with open(self.path, 'r') as results_f:
            with open(self.base_path) as base_f:
                results_obj = csv.reader(results_f)
                base_obj = csv.reader(base_f)
                row_count_results = sum(1 for row in results_obj)
                row_count_base = sum(1 for row in base_obj)
                if row_count_base != row_count_results:
                    print(f'####\nWARNING: reports have different lenght base:{row_count_base} report:{row_count_results}\n####')
        return row_count_results

    def get_agv_mad(self):
        with open(self.path, 'r') as results_f:
            with open(self.base_path) as base_f:
                self.check_len(results_f, base_f)
                results_obj = csv.reader(results_f)
                base_obj = csv.reader(base_f)
                self.check_len(results_obj, base_obj)
                next(results_obj)
                next(base_f)
                for r_line, b_line in zip(results_obj, base_obj):
                    if r_line[-1] != '0.0': # check if adv attack was applied
                        if b_line[1] == b_line[2]: # check if base model predicted correctly
                            if r_line[1] != r_line[2]: # check if adv model forced misclassification
                                self.mad += float(r_line[3])
                                self.n += 1
                return self.mad / (self.n + self.mu)

    def __call__(self, asr):
        avg_mad = self.get_agv_mad()
        return asr / (math.tanh(10*self.eps*avg_mad)) + 0.00000001


class AUC:
    
    def __init__(self, target_path=False):
        """
        This method calculates the overall performance of an attack (regardless of eps)
        by using the trapezoidal rule numerical integration method:
        
        ( xi+1 - xi ) * ( yi + yi+1 ) / 2   
        """
        self.target_path = target_path
    
    def update_target_path(self, new_path):
        self.target_path = new_path
    
    def __call__(self, x_list, y_list):
        auc = 0.0
        break_point = len(x_list) - 1
        for e, (eps, asp) in enumerate(zip(x_list, y_list)):
            if e >= break_point:
                break
            next_val_x = x_list[e+1]
            next_val_y = y_list[e+1]
            trapez_area = ((next_val_x - eps)*(asp + next_val_y))/2
            auc += trapez_area
        if self.target_path != False:
            with open(self.target_path, 'w') as auc_file:
                auc_file.write(f'AUC result : {auc}')
        return auc
        
            

if __name__ == '__main__':
    pass