import torch
import csv
import torch.optim as optim
import matplotlib.pyplot as plt
import time

from diff_jpeg import diff_jpeg_coding
from src.adversarial.custom_cw import CW
from src.utils.datautils import apply_along_dim
from torchvision.io import encode_jpeg, decode_image



class RCW(CW):
    
    def __init__(self, 
                model, 
                model_trms, 
                rcw_comp_lower_bound, 
                rcw_beta,
                cqe_init='random',
                q_search_type='cqe',
                ablation=False,
                *args, 
                **kwargs):
        """Robust CW employs a differentiable approximation
        in the constrained optimization in the form of T
        to improve the robustness in compression scenarios.
        
        ||x - x_hat||_2 + b * f(JPEG(x_hat, q), theta)
        
        jpeg_quality argument should be in range [0,1]
        they will be internally projected to 255
        and after compression back to 1.0
        
        self.compress(x) should be applied directly on image 
        CQE(Compression Quality Estimation) is a binary search algorithm
        to find the quality setting of a JPEG algorithm
        
        """
        super().__init__(model, model_trms, *args, **kwargs)
        self.compression_lower_bound = rcw_comp_lower_bound
        self.cqe_log = {}
        self.cqe_steps_log = []
        self.n = 0
        if q_search_type == 'cqe':
            self.q_search_type = self.compression_quality_estimation
            self.cqe_init = self.cqe_init_random if cqe_init == 'random' else self.cqe_init_center
        elif ablation:
            self.q_search_type = self.return_fixed
        else:
            self.q_search_type = self.brute_force_search_for_q
        
    def compress(self, img, jpeg_quality):
        img = img * 255
        compressed =  diff_jpeg_coding(image_rgb=img, jpeg_quality=jpeg_quality)
        return (compressed / 255).clip(min=0., max=1.)

    def compress_for_q_search(self, img, jpeg_quality):
        img = (img * 255).to(torch.uint8)
        compressed = apply_along_dim(img, funcs_tuple=(encode_jpeg, decode_image), jpeg_quality=jpeg_quality, dim=0)
        compressed_img = compressed / 255
        return compressed_img.clip(min=0., max=1.)

    def get_l2_loss(self, adv_images, images, attack_mask):
        #current_iq_loss = self.loss(self.flatten(adv_images,), self.flatten(images)).sum(dim=1)
        current_iq_loss = (adv_images - images).pow(2).sum(dim=(1,2,3)).sqrt()
        iq_loss = current_iq_loss.sum()
        return iq_loss, current_iq_loss
    
    def cqe_init_random(self, batch_size):
        return torch.randint(low=2, high=98, size=(1, batch_size))
    
    def cqe_init_center(self, batch_size):
        return torch.full((1, batch_size), 50)

    def search_for_q(self, cqe_image, ref_image):
        return self.q_search_type(cqe_image, ref_image)

    def return_fixed(self, cqe_image, ref_image):
        batch_size = cqe_image.shape[0]
        best_q = torch.full((1,batch_size), 80)
        return best_q, 0
    
    def avg_n_steps_for_cqe(self):
        return sum(self.cqe_steps_log) / self.n

    def compression_quality_estimation(self, cqe_image, ref_image, acceptance_threshold=0.1):
        """
        determines JPEG compression quality based on binary search.
        
        cqe_image: output of JPEG algo we want to compare against
        ref_image: uncompressed image used to find the right compression val
        schedule: constant thta scales down the step size. Gets smaller as n_steps increases
        q: JPEG quality setting we want to change to find the true compression setting
        exploration_step: keep exploring direction even if previous step resulted in worse l2
        
        """
        print('######## INIT JPEG SETTING SEARCH ####### \n')
        batch_size = cqe_image.shape[0]
        step_size = torch.ones((1, batch_size)).int()
        ref_image = ref_image.to('cpu')
        cqe_image = cqe_image.to('cpu')
        direction = torch.full((1, batch_size), -1)
        schedule = 1.0
        q = self.cqe_init(batch_size)
        #q = 98
        current_l2 = torch.full((1, batch_size), 1e07)
        best_l2 = torch.full((1, batch_size), 1e07)
        best_q = q
        break_criterion = torch.zeros((1, batch_size))
        n_steps = torch.zeros((batch_size,))
        # keep exploring direction even if previous step resulted in worse l2
        exploration_steps = torch.zeros((1, batch_size))
        lower_than_acceptance_mask = torch.zeros((1,batch_size)).bool()

        compressed_img = self.compress_for_q_search(ref_image, jpeg_quality=q.squeeze(0))
        last_l2 = (compressed_img - cqe_image).pow(2).sum(dim=(1,2,3)).sqrt()
        
        while torch.any(best_l2 > acceptance_threshold):
            steps = torch.clamp(step_size, min=1.).int()
            steps[lower_than_acceptance_mask] = step_size[lower_than_acceptance_mask]
            q += direction * torch.round((schedule * steps)).int()
            q = torch.clamp(q, min=1, max=99)
            compressed_img = self.compress_for_q_search(ref_image, jpeg_quality=q.squeeze(0))
            current_l2 = (compressed_img - cqe_image).pow(2).sum(dim=(1,2,3)).sqrt()

            # parallelize exploration steps logic
            is_worse_mask = (current_l2 >= last_l2).unsqueeze(0)
            exploration_steps[is_worse_mask] += 1
            break_criterion[is_worse_mask] += 1
            chg_dir_mask = exploration_steps > 2
            direction[chg_dir_mask] *= -1
            exploration_steps[chg_dir_mask] = 0

            # parallelize better d logic
            is_better_mask = ~is_worse_mask
            break_criterion[is_better_mask] = 0

            #parallelize best_d logic
            is_best_mask = (current_l2 < best_l2)
            best_q[is_best_mask] = q[is_best_mask]
            best_l2[is_best_mask] = current_l2[is_best_mask.squeeze(0)]
            
            exploration_steps[is_better_mask] = 0

            lower_than_acceptance_mask = current_l2 <= acceptance_threshold
            higher_than_acceptance_mask = ~lower_than_acceptance_mask
            step_size = torch.round(current_l2).int()
            last_l2 = current_l2
            schedule *= 0.99
            n_steps[higher_than_acceptance_mask] += 1

            if torch.all(break_criterion >= 10) or n_steps.max() >=150:
                # best q has not changed for 10 iterations
                break
            print(f'step : {n_steps}, q : {q}, current_l2 : {current_l2}, best q : {best_q}\n')
        print(f'BEST q FOUND: {best_q}')
        n_steps = n_steps.sum().item()
        return best_q, n_steps

    def brute_force_search_for_q(self, cqe_image, ref_image, acceptance_threshold=0.1):
        print('######## INIT JPEG SETTING SEARCH ####### \n')
        batch_size = cqe_image.shape[0]
        ref_image = ref_image.to('cpu')
        cqe_image = cqe_image.to('cpu')
        q_range = list(reversed(range(70, 100)))
        last_l2 = torch.full((batch_size,), 1e07)
        best_l2 = torch.full((1, batch_size), 1e07)
        best_q = torch.full((1, batch_size), 100)
        n_steps = 0

        for q in q_range:
            q = torch.full((1, batch_size), q)
            compressed_img = self.compress_for_q_search(ref_image, jpeg_quality=q.squeeze(0))
            current_l2 = (compressed_img - cqe_image).pow(2).sum(dim=(1,2,3)).sqrt()
            is_worse_mask = (current_l2 >= best_l2)
            is_better_mask = ~is_worse_mask
            best_q[is_better_mask] = q[is_better_mask]
            best_l2[is_better_mask] = current_l2[is_better_mask.squeeze(0)]
            last_l2 = current_l2
            n_steps += 1
            print(f'step : {n_steps}, q : {q}, current_l2 : {current_l2}, best q : {best_q}\n')
        return best_q, n_steps * batch_size

    def log_cqe_results(self, path):
        with open(path, 'w') as log_file:
            csv_obj = csv.writer(log_file)
            csv_obj.writerow(['QUALITY', 'OCCURENCES'])
            for k,v in self.cqe_log.items():
                csv_obj.writerow([k,v])
            n_steps = self.avg_n_steps_for_cqe()
            csv_obj.writerow(['n_steps',str(n_steps)])
            
        q_setting, _ = max(self.cqe_log.items(), key= lambda x: x[1])
        q_setting = int(q_setting)
        # we are assuming that the desired setting is the maximum chosen value
        #sorted_items = sorted(items, key=lambda x: x[0])
        x_list = list(range(q_setting - 10, q_setting + 10 + 1))
        y_list = [self.cqe_log.get(str(x), 0) for x in x_list]
        
        y_sum = sum(y_list)
        y_list = [y / y_sum for y in y_list]
        
        plt.figure(figsize=(8, 6))
        plt.bar(x_list, y_list, color='skyblue')
        plt.xticks(x_list)
        plt.xlabel('Compression Values')
        plt.ylabel('y')
        plt.title('Compression Adaptation Results')
        
        path_dir = '/'.join(path.split('/')[:-1]) + '/' + 'cqe_fig.png'
        plt.savefig(path_dir)
        plt.show()
        
    def get_ro_loss(self, adv_images, labels, target_labels):
    
        compressed_img = self.compress(adv_images, jpeg_quality=self.determined_quality.to(self.device))
        
        #outputs = self.get_outputs(adv_images)
        compressed_outputs = self.get_outputs(compressed_img) 
        
        if self.targeted:
            f_compressed = self.f(compressed_outputs, target_labels)
            f_compressed *= self.c
        else:
            f_compressed = self.f(compressed_outputs, labels).sum()
            f_compressed *= self.c
        return f_compressed, compressed_outputs

    def forward(self, images, labels, cqe_image):
        r"""
        Overridden.
        """
        start_time = time.perf_counter()
        print(f'\nCalculating adversarial example for images...\n\n')
        #images = images / 255
        
        self.c = self.original_c
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        
        self.determined_quality, n_steps = self.search_for_q(cqe_image=cqe_image, ref_image=images)
        self.determined_quality = self.determined_quality.squeeze(0)
        self.cqe_steps_log.append(n_steps)
        self.n += images.shape[0]
        for quality in self.determined_quality.tolist():
            if str(quality) in self.cqe_log.keys():
                self.cqe_log[str(quality)] += 1
            else:
                self.cqe_log[str(quality)] = 1

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
                iq_loss, current_iq_loss = self.get_iq_loss(adv_images, images, attack_mask)

                # Calculate adversarial loss including robustness loss
                ro_loss, comp_outputs = self.get_ro_loss(adv_images, labels, target_labels)

                cost = (current_iq_loss + ro_loss).sum() # ro_loss holds c tradeoff computation
                
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
                
                print(f'\n{step} - iq_loss: {iq_loss.item()}, r_loss: {ro_loss.sum().item()} cost: {cost.item()}')
                end_time = time.perf_counter()
                elapsed_time = (end_time - start_time) / batch_size
                self.loop_times.append(elapsed_time)
                start_time = time.perf_counter()

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
            self.write_to_protocol_dir(iq_loss, ro_loss, cost)
            self.write_runtime_to_protocol_dir()
        self.write_timing('cqe', self.n)
        return (best_adv_images_from_starts, target_labels)





class EnsembleRCW(CW):

    def __init__(self, 
                model, 
                model_trms, 
                N,
                *args, 
                **kwargs):
        """Robust CW employs a differentiable approximation
        in the constrained optimization in the form of T
        to improve the robustness in compression scenarios.
        
        ||x - x_hat||_2 + b * f(JPEG(x_hat, q), theta)
        
        jpeg_quality argument should be in range [0,1]
        they will be internally projected to 255
        and after compression back to 1.0
        
        self.compress(x) should be applied directly on image 
        CQE(Compression Quality Estimation) is a binary search algorithm
        to find the quality setting of a JPEG algorithm
        
        """
        
        super().__init__(model, model_trms, *args, **kwargs)
        self.n = 0
        self.compression_rates = []
        compression = 99
        self.N_comp_rates = N
        for i in range(N): # in 5-step decrements
            self.compression_rates.append(compression)
            compression -= 5
        

    def ensemble_grad(self, w, labels, target_labels, ori_images):

        ensemble_grad = torch.zeros_like(w).detach().to(self.device)    
        grad_list = []
        loss_tensor = torch.zeros(self.N_comp_rates)
        ro_loss_tensor = torch.zeros(self.N_comp_rates)
        
        for e, compression_rate in enumerate(self.compression_rates):
            w_i = w.clone().detach()
            w_i.requires_grad = True
            adv_images_i = self.tanh_space(w_i)
            iq_loss, current_iq_loss = self.get_iq_loss(adv_images_i, ori_images)
            ro_loss, comp_outs = self.get_ro_loss(adv_images_i, labels, target_labels, q=torch.full((w.shape[0],), compression_rate, dtype=torch.uint8))
            if e == len(self.compression_rates) // 2:
                comp_outs_mean = comp_outs

            cost = (ro_loss + iq_loss).mean()
            # Update adversarial images
            grad = torch.autograd.grad(cost, w_i,
                                    retain_graph=False,
                                    create_graph=False)[0]
            
            grad_list.append(grad)
            loss_tensor[e] = cost
            ro_loss_tensor[e] = ro_loss.mean()
        
        total_cost_exp = loss_tensor.exp().sum()
        for cost, grad in zip(loss_tensor, grad_list):
            ensemble_grad +=  (1 - (torch.exp(cost)) / total_cost_exp) * grad
        return ensemble_grad, comp_outs_mean, loss_tensor.mean(), ro_loss_tensor.mean(), current_iq_loss, adv_images_i, iq_loss
                
        
    def compress(self, img, jpeg_quality):
        img = img * 255
        compressed =  diff_jpeg_coding(image_rgb=img, jpeg_quality=jpeg_quality)
        return (compressed / 255).clip(min=0., max=1.)

    def get_l2_loss(self, adv_images, images, attack_mask=None):
        current_iq_loss = (adv_images - images).pow(2).sum(dim=(1,2,3)).sqrt()
        iq_loss = current_iq_loss.sum()
        return iq_loss, current_iq_loss
    
        
    def get_ro_loss(self, adv_images, labels, target_labels, q):
    
        compressed_img = self.compress(adv_images, jpeg_quality=q.to(self.device))
        
        compressed_outputs = self.get_outputs(compressed_img) 
        
        if self.targeted:
            f_compressed = self.f(compressed_outputs, target_labels)
            f_compressed *= self.c
        else:
            f_compressed = self.f(compressed_outputs, labels).sum()
            f_compressed *= self.c
        return f_compressed, compressed_outputs

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        print(f'\nCalculating adversarial example for images...\n\n')
        #images = images / 255
        
        self.c = self.original_c
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        

        self.n += images.shape[0]

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

    
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
            start_time = time.perf_counter() 
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


                # Calculate adversarial loss including robustness loss and get grad
                w.grad, comp_outputs, cost, ro_loss, current_iq_loss, adv_images, iq_loss = self.ensemble_grad(w, labels, target_labels, images)

                
                # Update adversarial images
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
                
                print(f'\n{step} - iq_loss: {iq_loss.item()}, ro_loss: {ro_loss.sum().item()}, cost: {cost.sum().item()}') 
                end_time = time.perf_counter()
                elapsed_time = (end_time - start_time) / batch_size
                self.loop_times.append(elapsed_time)
                start_time = time.perf_counter() 

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
            self.write_to_protocol_dir(iq_loss, cost, cost)
            self.write_runtime_to_protocol_dir()
        
        self.write_timing('ensemble', self.n)
        return (best_adv_images_from_starts, target_labels)

