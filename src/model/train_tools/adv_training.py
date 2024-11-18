import torch


from art.estimators.classification import PyTorchClassifier
from art.data_generators import PyTorchDataGenerator
from art.defences.trainer import AdversarialTrainer, AdversarialTrainerFBFPyTorch
from art.attacks.evasion import FastGradientMethod, BasicIterativeMethod, ProjectedGradientDescent
from tqdm import tqdm

from src.model.training import Training
from src.model.xception import Xception


class BaseAdversarialTraining(Training):
    
    # model should output logits
    
    def __init__(self, ratio=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.model.model_name in ['xception', 'inception']:
            input_shape = (self.data.batch_size, 3, 299, 299)
        else:
            input_shape = (self.data.batch_size, 3, 224, 224)
        device_type = 'gpu' if torch.cuda.is_available() else 'cpu'
        self.classifier = PyTorchClassifier(self.model, 
                                            loss=self.criterion, 
                                            input_shape=input_shape, 
                                            nb_classes=self.num_classes,
                                            optimizer=self.optim,
                                            device_type=device_type)
        print(f'\nperforming training on: {self.classifier.device_type}\n')
        self.art_train = PyTorchDataGenerator(self.data.train, size=len(self.data.train.dataset), batch_size=self.data.batch_size)
        self.art_test = PyTorchDataGenerator(self.data.test, size=len(self.data.train.dataset), batch_size=self.data.batch_size)
        self.ratio = ratio

    def train_model(self, save_opt):
        best_acc = 0.0
        self.training.fit_generator(self.art_train, nb_epochs=self.epochs)
        self.model = self.training.get_classifier().model
        return self.model, best_acc
    
    def test_model(self, save_opt):
        with torch.no_grad():
            self.model.eval()
            self.utils.metrics.reset()
            print(f'\nTest Model\n')
            for x, y, paths in tqdm(self.data.test):
                x, y = x.to(self.device), y.to(self.device)
                x_hat = self.model(x)
                self.utils.metrics(x_hat, y)
                self.report(x_hat, y, paths)
            result = self.utils.metrics.compute()
            self.utils.save_model(model=self.model, save_opt=save_opt, result=result)
            self.utils.logger.log_test_results(result)
            print('\nTEST COMPLETED. RESULTS:\n')
            print(result)
            print('\n\n')
            self.utils.logger.exit()
            return result['MulticlassAccuracy'].item()

class PGDAdversarialTraining(BaseAdversarialTraining):
    
    # model should output logits
    
    def __init__(self, attacks=['PGD'], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attacks = self.load_attacks_for_training(attacks)
        self.training = AdversarialTrainer(self.classifier, attacks=list(self.attacks.values()), ratio=self.ratio)
        
class EnsembleAdversarialTraining(BaseAdversarialTraining):

    # Ensemble Adversarial Training with Multiple surrogate models
    # https://arxiv.org/abs/1705.07204
    
    def __init__(self, surrogate_models, attacks, ratio, *args, **kwargs):
        super().__init__(attacks=attacks, ratio=ratio, *args, **kwargs)
        self.surrogate_models = surrogate_models # models that provide the gradient for the AS

    def load_attacks_for_training(self, attacks):
        attacks_dict = {}
        # possible attacks: FGSM, BIM, PGD
        if any([x in attacks for x in ['fgsm', 'FGSM']]):
            fgsm = FastGradientMethod(self.classifier, eps=0.3, eps_step=0.01, max_iter=40)
            attacks_dict['FGSM'] = fgsm
        if any([x in attacks for x in ['bim', 'BIM']]):
            bim = BasicIterativeMethod(self.classifier, eps=0.3, eps_step=0.01, max_iter=40)
            attacks_dict['BIM'] = bim
        if any([x in attacks for x in ['pgd', 'PGD']]):
            pgd = ProjectedGradientDescent(self.classifier, eps=0.3, eps_step=0.01, max_iter=40)
            attacks_dict['PGD'] = pgd
        return attacks_dict
        

class FBFAdversarialTraining(BaseAdversarialTraining):
    
    # "Fast is better than Free" Adversarial Training
    # https://arxiv.org/pdf/2001.03994.pdf
    
    def __init__(self, eps, *args, **kwargs):
        """
        The effectiveness of this protocol is found to be sensitive to the use 
        of techniques like data augmentation, gradient clipping and learning rate
        schedules. Optionally, the use of mixed precision arithmetic operation via 
        apex library can significantly reduce the training time making this 
        one of the fastest adversarial training protocol.
        """
        super().__init__(*args, **kwargs)
        self.training = AdversarialTrainerFBFPyTorch(self.classifier, eps=eps)
        
        
        
if __name__ == '__main__':
    pass