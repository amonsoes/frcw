import torch

from torch import nn
from torch.nn import Conv2d, Linear, MaxPool2d, ReLU, Dropout, BatchNorm2d
from torch import flatten


class BaseCnn(nn.Module):

    def __init__(self,
                dim_inp_x,
                dim_inp_y, 
                n_channels,
                conv1_dict,
                conv2_dict, 
                pool1,
                pool2,
                fc2, 
                dropout,
                batch_size,
                device,
                pretrained):
        print('\n initializing model ...\n')
        super().__init__()
        self.device = torch.device(device)
        self.conv1 = Conv2d(in_channels=n_channels, out_channels=conv1_dict['out'], kernel_size=conv1_dict['size'], padding=conv1_dict['pad'], dilation=conv1_dict['dil'])
        self.conv2 = Conv2d(in_channels=conv1_dict['out'], out_channels=conv2_dict['out'], kernel_size=conv2_dict['size'], padding=conv2_dict['pad'], dilation=conv2_dict['dil'])
        self.pool1 = MaxPool2d(kernel_size=pool1)
        self.pool2 = MaxPool2d(kernel_size=pool2)
        self.batch_norm1 = BatchNorm2d(conv1_dict['out'])
        self.batch_norm2 = BatchNorm2d(conv2_dict['out'])
        self.dropout = Dropout(p=dropout)
        self.relu = ReLU()
        
        fc1 = self.infer_fc_size(n_channels, dim_inp_x, dim_inp_y)

        self.fc1 = Linear(in_features=fc1, out_features=fc2)
        self.fc2 = Linear(in_features=fc2, out_features=1)
        self.batch_size = batch_size
        self.to(self.device)
        print('\n model initialized \n\n')
        print(self)

        if pretrained:
            self.from_state_dict(pretrained)

    @classmethod
    def init_from_dict(cls, params, pretrained_path):
        model = cls(dim_inp_x=params['input_size'],
                    dim_inp_y=params['input_size'],
                    n_channels=params['n_channels'],
                    conv1_dict=params['conv1'],
                    conv2_dict=params['conv2'],
                    pool1=params['pool1'],
                    pool2=params['pool2'],
                    fc2=params['fc2'],
                    dropout=0.0,
                    batch_size=1,
                    device='cpu',
                    pretrained=pretrained_path)
        model.eval()
        return model
    
    def forward(self, x):
        # inp = batch_size x n_channels x width x height

        x = self.conv1(x)
        x = self.pool1(x)
        x = self.batch_norm1(self.relu(x))

        x = self.conv2(x)
        x = self.pool2(x)
        x = self.batch_norm2(self.relu(x))

        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        logits = self.fc2(x)

        return logits
    
    
    def get_fully_connected_dim(self, 
                                inp_dim_x, 
                                inp_dim_y, 
                                conv1_dict,
                                conv2_dict, 
                                pool1,
                                pool2):
        """ substracts filter parameters from image dimensions to calculate model parameters for flattening
        """
        out_x1 = self.get_conv_ouput_params(inp_dim_x, conv1_dict)
        out_x2 = self.get_pool_ouput_params(out_x1, pool1)
        out_x3 = self.get_conv_ouput_params(out_x2, conv2_dict)
        out_x4 = self.get_pool_ouput_params(out_x3, pool2)
        
        out_y1 = self.get_conv_ouput_params(inp_dim_y,conv1_dict)
        out_y2 = self.get_pool_ouput_params(out_y1, pool1)
        out_y3 = self.get_conv_ouput_params(out_y2, conv2_dict)
        out_y4 = self.get_pool_ouput_params(out_y3, pool2)
        return int(conv2_dict['out']*out_x4*out_y4)
    
    def get_conv_ouput_params(self, input_size, f_dict):
        return int((input_size + 2*f_dict['pad'] - f_dict['dil']*(f_dict['size'] - 1) - 1)/f_dict['stride'] + 1)

    def get_pool_ouput_params(self, input_size, pool_size):
        # if stride is undefined, it is the same as 'size'
        # padding is 0 by default
        # dilation is 1 by default
        return int((input_size + 2*0 - 1*(pool_size - 1) - 1)/pool_size + 1)

    def infer_fc_size(self, n_channels, inp_dim_x, inp_dim_y):
        x = torch.randn((1, n_channels, inp_dim_y, inp_dim_x))
        
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = flatten(x, 1)
        return x.shape[1]

    def to_state_dict(self, path, transform, analysis):
        torch.save(self.state_dict(), path)
        with open(path+f'_params_{transform}_{analysis[:5]}.txt', 'w') as f:
            f.write(str(self)) 
        print(f'\n\nsaved model at: {path}\n\n')

    def from_state_dict(self, path):
        print(f'\n\nload model from: {path}\n\n')
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.to(self.device)
        self.eval()


if __name__ == '__main__':
    pass
    


