import csv

from torch.utils.data import DataLoader, random_split
from src.datasets.subsets import Nips17Subset
from src.datasets.data_transforms.img_transform import PreTransforms, PostTransforms

class Data:
    
    def __init__(self, dataset_name, *args, **kwargs):
        self.dataset = self.loader(dataset_name, *args, **kwargs)
    
    def loader(self, dataset_name, *args, **kwargs):
        if dataset_name == 'nips17':
            dataset = Nips17ImgNetData(*args, **kwargs)
        else:
            raise ValueError('Dataset not recognized')
        return dataset

class BaseDataset:
    
    def __init__(self,
                dataset_name,
                model,
                device,
                batch_size,
                transform,
                adversarial_opt,
                adversarial_training_opt,
                jpeg_compression,
                jpeg_compression_rate,
                target_transform=None,
                input_size=224):


        self.transform_type = transform
        self.transforms = PreTransforms(device=device,
                                    target_transform=target_transform, 
                                    input_size=input_size, 
                                    dataset_type=dataset_name,
                                    adversarial_opt=adversarial_opt)
        self.post_transforms = PostTransforms(transform,
                                            adversarial_opt=adversarial_opt,
                                            input_size=input_size,
                                            jpeg_compression=jpeg_compression,
                                            jpeg_compression_rate=jpeg_compression_rate,
                                            dataset_type=dataset_name,
                                            model=model,
                                            device=device,
                                            val_base_trm_size=self.transforms.val_base_trm_size)
        self.adversarial_opt = adversarial_opt
        self.adversarial_training_opt = adversarial_training_opt
        self.device = device
        self.batch_size = batch_size
        self.x, self.y = input_size, input_size


class Nips17ImgNetData(BaseDataset):

    def __init__(self, n_datapoints, *args,**kwargs):
        super().__init__('nips17', *args, **kwargs)
        
        self.categories = self.get_categories()
        self.dataset_type = 'nips17'

        self.test_data = self.get_data(transform_val=self.transforms.transform_val, 
                                    target_transform=self.transforms.target_transform)
        if n_datapoints == -1:
            self.test = self.train = self.validation =  DataLoader(self.test_data, batch_size=self.batch_size)
        else:
            self.test_data, _ = random_split(self.test_data, [n_datapoints, len(self.test_data)-n_datapoints])
            self.test = self.train = self.validation =  DataLoader(self.test_data, batch_size=self.batch_size)

    def get_data(self, transform_val, target_transform):
        path_test = './data/nips17/'
        path_labels = path_test + 'images.csv'
        path_images = path_test + 'images/'
        test = Nips17Subset(label_path=path_labels, 
                            img_path=path_images, 
                            transform=transform_val, 
                            target_transform=target_transform, 
                            adversarial=self.adversarial_opt.adversarial, 
                            is_test_data=True)
        return test
        
    def get_categories(self):
        categories = {}
        path = './data/nips17/categories.csv'
        with open(path, 'r') as cats:
            filereader = csv.reader(cats)
            next(filereader)
            for ind, cat in filereader:
                categories[int(ind) - 1] = cat
        return categories

        
if __name__ == '__main__':
    pass
