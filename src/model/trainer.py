import torch

from src.model.training import CNNTraining
from src.model.train_tools.adv_training import PGDAdversarialTraining, EnsembleAdversarialTraining, FBFAdversarialTraining

class Trainer:
    
    def __init__(self, 
                opt,
                model, 
                model_name, 
                data,
                num_classes,
                optim_args, 
                epochs, 
                model_type, 
                log_result, 
                adversarial_training_opt,
                lr_gamma=0.9):
        
        """if adversarial_training_opt.adversarial_training:
            self.train_model_fn = self.train_model_adv
        else:
            self.train_model_fn = self.train_model_normal"""
        
        if adversarial_training_opt.adversarial_training:
            
            if torch.cuda.is_available():
                torch.cuda.set_device(data.device) # this is necessary due to how devices are chosen in ART
            
            if adversarial_training_opt.adv_training_type == 'base':
                self.training = PGDAdversarialTraining(attacks=adversarial_training_opt.attacks_for_training,
                                                model=model, 
                                                model_name=model_name, 
                                                data=data,
                                                num_classes=num_classes, 
                                                optim_args=optim_args, 
                                                epochs=epochs, 
                                                model_type=model_type, 
                                                log_result=log_result,
                                                lr_gamma=lr_gamma)
            elif adversarial_training_opt.adv_training_type == 'ensemble':
                self.training = EnsembleAdversarialTraining(model=model, 
                                                model_name=model_name, 
                                                data=data,
                                                num_classes=num_classes,
                                                optim_args=optim_args, 
                                                epochs=epochs, 
                                                model_type=model_type, 
                                                log_result=log_result,
                                                lr_gamma=lr_gamma)
            elif adversarial_training_opt.adv_training_type == 'fbtf':
                self.training = FBFAdversarialTraining(eps=adversarial_training_opt.training_eps,
                                                model=model, 
                                                model_name=model_name, 
                                                data=data, 
                                                optim_args=optim_args, 
                                                epochs=epochs, 
                                                model_type=model_type, 
                                                log_result=log_result,
                                                lr_gamma=lr_gamma)
                
        else:
            self.training = CNNTraining(opt,
                                    model, 
                                    model_name, 
                                    data,
                                    num_classes,
                                    optim_args, 
                                    epochs, 
                                    model_type, 
                                    log_result,
                                    lr_gamma=0.9)
    
    """def train_model(self, save_opt):
        return self.train_model_fn(save_opt)"""
    
    def train_model(self, save_opt):
        model, best_acc = self.training.train_model(save_opt)
        self.best_acc = best_acc
        return best_acc
    
    """def train_model_normal(self, save_opt):
        model, best_acc = self.training.train_model(save_opt)
        self.best_acc = best_acc
        return model"""
    
    def test_model(self):
        return self.training.test_model()


if __name__ == '__main__':
    pass