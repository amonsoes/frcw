
import torch
import math


from diff_jpeg import diff_jpeg_coding
from src.adversarial.perc_cw import PerC_AL, ciede2000_diff, quantization, rgb2lab_diff
from src.utils.datautils import apply_along_dim
from torchvision.io import encode_jpeg, decode_image


class RAL(PerC_AL):

    def __init__(self,                 
                model, 
                model_trms, 
                cqe_init='random',
                q_search_type='cqe',
                *args, 
                **kwargs):
        super().__init__(model, model_trms, *args, **kwargs)
        self.cqe_log = {}
        self.cqe_steps_log = []
        self.n = 0
        if q_search_type == 'cqe':
            self.q_search_type = self.compression_quality_estimation
            self.cqe_init = self.cqe_init_random if cqe_init == 'random' else self.cqe_init_center
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

    def cqe_init_random(self, batch_size):
        return torch.randint(low=2, high=98, size=(1, batch_size))
    
    def cqe_init_center(self, batch_size):
        return torch.full((1, batch_size), 50)

    def search_for_q(self, cqe_image, ref_image):
        return self.q_search_type(cqe_image, ref_image)

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

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor, cqe_image: torch.Tensor) -> torch.Tensor:
        """
        Performs the adversary of the model given the inputs and labels.

        Parameters
        inputs : torch.Tensor
            Batch of image examples in the range of [0,1].
        labels : torch.Tensor
            Original labels if untargeted, else labels of targets.

        Returns
        -------
        torch.Tensor
            Batch of image samples modified to be adversarial
        """
        self.kappa = self.original_kappa
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        ori_labels = labels

        self.determined_quality, n_steps = self.search_for_q(cqe_image=cqe_image, ref_image=inputs)
        self.determined_quality = self.determined_quality.squeeze(0)
        self.cqe_steps_log.append(n_steps)

        if self.targeted:
            comp_inputs = self.compress(inputs, jpeg_quality=self.determined_quality.to(self.device))
            labels = self.get_target_label(comp_inputs, labels)

        if inputs.min() < 0 or inputs.max() > 1: raise ValueError('Input values should be in the [0, 1] range.')



        # set for schedule and targeting
        alpha_l_min=self.alpha_l_init/100
        alpha_c_min=self.alpha_c_init/10
        multiplier = -1 if self.targeted else 1

        # init attack params
        X_adv_round_best=inputs.clone()
        batch_size=inputs.shape[0]
        delta=torch.zeros_like(inputs, requires_grad=True)
        inputs_LAB=rgb2lab_diff(inputs,self.device)
        mask_isadv= torch.zeros(batch_size,dtype=torch.uint8).to(self.device)
        color_l2_delta_bound_best=(torch.ones(batch_size)*100000).to(self.device)

        # set cases according to target mode
        if (self.targeted==False) and self.kappa!=0:
            labels_onehot = torch.zeros(labels.size(0), 1000, device=self.device)
            labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
            labels_infhot = torch.zeros_like(labels_onehot).scatter_(1, labels.unsqueeze(1), float('inf'))
        
        # start optimization
        for i in range(self.max_iterations):

            # cosine annealing for alpha_l and alpha_c 
            alpha_c=alpha_c_min+0.5*(self.alpha_c_init-alpha_c_min)*(1+math.cos(i/self.max_iterations*math.pi))
            alpha_l=alpha_l_min+0.5*(self.alpha_l_init-alpha_l_min)*(1+math.cos(i/self.max_iterations*math.pi))

            # get cross-entropy loss for adv sample, scale CE-grad to unit length, update delta by adversarial gradient
            compressed_imgs = self.compress(inputs+delta, jpeg_quality=self.determined_quality.to(self.device))
            loss = multiplier * (self.loss(self.model(self.model_trms(compressed_imgs)), labels) + self.kappa)
            loss.backward()
            grad_a=delta.grad.clone()
            delta.grad.zero_()
            delta.data[~mask_isadv]=delta.data[~mask_isadv]+alpha_l*(grad_a.permute(1,2,3,0)/torch.norm(grad_a.view(batch_size,-1),dim=1)).permute(3,0,1,2)[~mask_isadv]  
            
            # compute CIEDE2000 difference and get fidelity gradients, scale, update delta by color gradient
            d_map=ciede2000_diff(inputs_LAB, rgb2lab_diff(inputs+delta,self.device),self.device).unsqueeze(1)
            color_dis=torch.norm(d_map.view(batch_size,-1),dim=1)
            color_loss=color_dis.mean()
            color_loss
            color_loss.backward()
            grad_color=delta.grad.clone()
            delta.grad.zero_()
            delta.data[mask_isadv]=delta.data[mask_isadv]-alpha_c* (grad_color.permute(1,2,3,0)/torch.norm(grad_color.view(batch_size,-1),dim=1)).permute(3,0,1,2)[mask_isadv]
            delta.data=(inputs+delta.data).clamp(0,1)-inputs

            # quantize image (not included in any backward comps) & check if samples are adversarial
            X_adv_round=inputs+delta.data
            mask_isadv = self.check_if_adv(X_adv_round, labels)

            # update adversarial image if: (1) color dist is less (2) images are adversarial
            mask_best=(color_dis.data<color_l2_delta_bound_best)
            mask=mask_best * mask_isadv
            color_l2_delta_bound_best[mask]=color_dis.data[mask]
            X_adv_round_best[mask]=X_adv_round[mask]
            print(f'adv_loss:{loss}, color_loss:{color_loss}, is_adv:{mask_isadv}, adjusted: {mask}')


        return X_adv_round_best
    
    def check_if_adv(self, X_adv_round, labels):
            X_adv_round = self.compress(X_adv_round, jpeg_quality=self.determined_quality.to(self.device))
            outputs = self.model(self.model_trms(X_adv_round))
            one_hot_labels = torch.eye(len(outputs[0]), device=self.device)[labels].to(self.device)

            i, _ = torch.max((1-one_hot_labels)*outputs, dim=1) # get the largest logit
            #outputs[outputs.argmax(dim=1)] = -1e10

            if self.targeted:
                j = torch.masked_select(outputs, one_hot_labels.bool()) # get the target logit
                adv_loss = (i - j) + self.kappa
                is_adv = adv_loss <= 0.0
                #return torch.clamp((i-j), min=-self.kappa)
            else:
                label_mask = torch.full_like(labels.view(-1,1), -1e+03).to(torch.float32)
                outputs.scatter_(1, labels.view(-1,1), label_mask)
                j, _ = torch.max(outputs, dim=1)
                adv_loss = (i - j) + self.kappa
                is_adv = adv_loss <= 0.0
            return is_adv

