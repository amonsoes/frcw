import torch
import csv
import torch.optim as optim
import matplotlib.pyplot as plt

from random import randint
from diff_jpeg import diff_jpeg_coding
from src.adversarial.custom_cw import CW
from src.adversarial.perc_cw import CIEDE2000Loss
from src.utils.datautils import apply_along_dim
from torchvision.io import encode_jpeg, decode_image


class VarRCW(CW):
    
    def __init__(self, 
                model, 
                model_trms, 
                rcw_comp_lower_bound, 
                rcw_beta,
                iq_loss='l2',
                N=4,
                ablation=False,
                *args,
                **kwargs):
        """Variance-tuned RCW
        
        adds a variance term to the gradient calculation
        to account for compression variance.
        This should enhance the portability of the attack 
        inbetween compression rates.
        N - number of compression rates

        """
        super().__init__(model, model_trms, *args, **kwargs)
        self.original_c = self.c
        self.compression_lower_bound = rcw_comp_lower_bound
        self.mu = 1.0

        self.compression_rates = []
        compression = 100
        self.N = N
        self.ablation = ablation
        for i in range(N): # in 5-step decrements
            compression -= 5
            self.compression_rates.append(compression)

        if ablation:
            self.forward_fn = self.forward_ablation
            self.main_comp_rate = 80
        else:
            # strongest compression should be used to compute the loss for variance computations
            self.main_comp_rate = self.compression_rates.pop(-1)

        
        if iq_loss == 'l2':
            self.iq_loss_fn = self.get_l2_loss
        elif iq_loss == 'ciede2000':
            self.ciede_loss = CIEDE2000Loss(device=self.device, batch_size=1)
            self.iq_loss_fn = self.get_ciede2000_loss
        else:
            raise ValueError('value for iq_loss not recognized. Choose one of ["l2","ciede2000","hpf"]')
        self.iq_loss_type = iq_loss


    def get_iq_loss(self, adv_images, images, attack_mask):
        iq_loss, current_iq_loss = self.iq_loss_fn(adv_images, images, attack_mask)
        return iq_loss, current_iq_loss
    
    def get_l2_loss(self, adv_images, images, attack_mask):
        #current_iq_loss = self.loss(self.flatten(adv_images,), self.flatten(images)).sum(dim=1)
        current_iq_loss = (adv_images - images).pow(2).sum(dim=(1,2,3)).sqrt()
        iq_loss = current_iq_loss.sum()
        return iq_loss, current_iq_loss

    def get_ciede2000_loss(self, adv_images, images, attack_mask):
        current_iq_loss = self.ciede_loss(adv_images, images)
        iq_loss = current_iq_loss.sum()
        return iq_loss, current_iq_loss
            
    def compress(self, img, jpeg_quality):
        img = img * 255
        compressed =  diff_jpeg_coding(image_rgb=img, jpeg_quality=jpeg_quality)
        return (compressed / 255).clip(min=0., max=1.)
    
    def get_ro_loss(self, adv_images, labels, target_labels):
        compressed_img = self.compress(adv_images, jpeg_quality=torch.tensor([self.main_comp_rate]*self.batch_size).to(self.device))
        
        #outputs = self.get_outputs(adv_images)
        compressed_outputs = self.get_outputs(compressed_img) 
        
        if self.targeted:
            f_compressed = self.f(compressed_outputs, target_labels)
            f_compressed *= self.c
        else:
            f_compressed = self.f(compressed_outputs, labels).sum()
            f_compressed *= self.c
        return f_compressed, compressed_outputs
    
    def get_loss(self, adv_images, labels, target_labels):
    
        #outputs = self.get_outputs(adv_images)
        compressed_outputs = self.get_outputs(adv_images) 
        
        if self.targeted:
            f_compressed = self.f(compressed_outputs, target_labels)
            f_compressed *= self.c
        else:
            f_compressed = self.f(compressed_outputs, labels).sum()
            f_compressed *= self.c
        return f_compressed, compressed_outputs

    
    def update_v(self, w, labels, target_labels, w_grad, current_iq_loss):
        """This function updates the gradient variance
        First iteration computes gradients of compressed w variants at various rates
        Second iteration calculates the variance:
        
        V(x) = 1/N * SUM_xi^N(grad_xi - grad_x)

        Args:
            w (_type_): original input to the optimizer
            labels (_type_): ground truth
            target_labels (_type_): target labels
            w_grad (_type_): gradient of w that was calculated in main iter

        Returns:
            _type_: _description_
        """
        v_list = []
        GV_grad = torch.zeros_like(w).detach().to(self.device) # images.shape == w.shape
        
        for rate in self.compression_rates:
            #self.optimizer.zero_grad()
            w_i = w.clone().detach() # copy w to obtain w_i to keep the graph of w intact
            w_i.requires_grad = True 
            img_i = self.tanh_space(w_i)  # get image back for compression and forward pass
            compressed_img = self.compress(img_i, jpeg_quality=torch.tensor([rate]*self.batch_size).to(self.device))
            f_compressed, _ = self.get_loss(compressed_img, labels, target_labels)
            cost = (current_iq_loss + f_compressed).sum()
            compressed_grad = torch.autograd.grad(cost, w_i,
                                               retain_graph=False, create_graph=False)[0]
            v_list.append(compressed_grad)
        
        for vi_grad in v_list:
            GV_grad += (vi_grad - w_grad)
        GV_grad = GV_grad / self.N
        return GV_grad

    def grad_var_momentum(self, w_grad_old, w_hat_grad, GV_grad):
        denom = (w_hat_grad + GV_grad).abs().sum()
        sum_grads = w_hat_grad + GV_grad
        w_grad = self.mu * w_grad_old + sum_grads / denom
        return w_grad
    
    def forward_regular(self, images, labels):
        raise ValueError('Not implemented yet')
            
        
    def forward_ablation(self, images, labels):
        r"""
        Overridden.
        """
        print(f'\nCalculating adversarial example for images...\n\n')
        #images = images / 255
        self.c = torch.full((images.shape[0],), self.original_c).to(self.device)
        
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        
        if self.iq_loss_type == 'hpf':
            #project x to x_ll for the constraint
            images_for_const , _ = self.hpf_dwt(images)
        else:
            images_for_const = images
        
        # define momentum and gradient variant as tensors with 0
        #v = torch.zeros_like(images).detach().to(self.device)
        #w_grad = torch.zeros_like(images).detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)
        
        #implement outer step like in perccw

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
                iq_loss, current_iq_loss = self.get_iq_loss(adv_images, images_for_const, 1)
                
                # Calculate adversarial loss including robustness loss
                ro_loss, comp_outputs = self.get_ro_loss(adv_images, labels, target_labels)

                cost = (current_iq_loss + ro_loss).sum() # ro_loss holds c tradeoff computation
                
                # Update adversarial images
                self.optimizer.zero_grad()
                cost.backward()
                
                #w_hat_grad = w.grad
                #w_grad = self.grad_var_momentum(w_grad, w_hat_grad, v)
                #w.grad = w_grad
                
                #edit gradient wrt w with variance list
                #v = self.update_v(w=w, labels=labels, target_labels=target_labels, w_grad=w_grad, current_iq_loss=current_iq_loss)
                
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
        
            # from perccw.py
            ######################   
                            
        if self.protocol_file:
            self.write_to_protocol_dir(iq_loss, ro_loss, cost)
            self.write_runtime_to_protocol_dir()
        return (best_adv_images_from_starts, target_labels)


    def forward_verbose(self, images, labels):
        r"""
        Overridden.
        """
        print(f'\nCalculating adversarial example for images...\n\n')
        #images = images / 255
        self.c = torch.full((images.shape[0],), self.original_c).to(self.device)
        
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        
        if self.iq_loss_type == 'hpf':
            #project x to x_ll for the constraint
            images_for_const , _ = self.hpf_dwt(images)
        else:
            images_for_const = images
        
        # define momentum and gradient variant as tensors with 0
        v = torch.zeros_like(images).detach().to(self.device)
        w_grad = torch.zeros_like(images).detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)
        
        #implement outer step like in perccw

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
                iq_loss, current_iq_loss = self.get_iq_loss(adv_images, images_for_const, 1)
                
                # Calculate adversarial loss including robustness loss
                ro_loss, comp_outputs = self.get_ro_loss(adv_images, labels, target_labels)

                cost = (current_iq_loss + ro_loss).sum() # ro_loss holds c tradeoff computation
                
                # Update adversarial images
                self.optimizer.zero_grad()
                cost.backward()
                
                w_hat_grad = w.grad
                w_grad = self.grad_var_momentum(w_grad, w_hat_grad, v)
                w.grad = w_grad
                
                #edit gradient wrt w with variance list
                v = self.update_v(w=w, labels=labels, target_labels=target_labels, w_grad=w_grad, current_iq_loss=current_iq_loss)
                
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
        
            # from perccw.py
            ######################   
                            
        if self.protocol_file:
            self.write_to_protocol_dir(iq_loss, ro_loss, cost)
            self.write_runtime_to_protocol_dir()
        return (best_adv_images_from_starts, target_labels)


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
        
        while torch.any(current_l2 > acceptance_threshold):
            steps = torch.clamp(step_size, min=1.).int()
            steps[lower_than_acceptance_mask] = step_size[lower_than_acceptance_mask]
            q += direction * torch.round((schedule * steps)).int()
            q = torch.clamp(q, min=1, max=99)
            compressed_img = self.compress_for_q_search(ref_image, jpeg_quality=q.squeeze(0))
            current_l2 = (compressed_img - cqe_image).pow(2).sum(dim=(1,2,3)).sqrt()

            # parallelize exploration steps logic
            is_worse_mask = (current_l2 >= last_l2).unsqueeze(0)
            exploration_steps[is_worse_mask] += 1
            chg_dir_mask = exploration_steps > 2
            direction[chg_dir_mask] *= -1
            exploration_steps[chg_dir_mask] = 0

            # parallelize criterion logic
            is_better_mask = ~is_worse_mask
            break_criterion[is_better_mask] = 0

            #parallelize best_q logic
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
            break_criterion += 1
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
    
        #random_quality = random.randint(self.compression_lower_bound, 99)
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
        return (best_adv_images_from_starts, target_labels)