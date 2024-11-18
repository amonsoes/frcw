import torch
import torch_dct as dct

from torchvision import transforms as T

def plot_tensor(tensor):
    to_pil = T.ToPILImage()
    image = to_pil(tensor)
    image.show()

def apply_along_dim(batch, funcs_tuple, jpeg_quality, dim):
    encode, decode = funcs_tuple
    tensors = torch.unbind(batch, dim=dim)
    enc_tensors = [encode(tensor, quality=q) for tensor, q in zip(tensors, jpeg_quality)]
    tensors = [decode(tensor) for tensor in enc_tensors]
    batch = torch.stack(tensors, dim=0)
    return batch


class Patchify:
    
    def __init__(self, img_size, patch_size, n_channels):
        self.img_size = img_size
        self.n_patches = (img_size // patch_size) ** 2

        assert (img_size // patch_size) * patch_size == img_size
        
    def __call__(self, x):
        p = x.unfold(1, 8, 8).unfold(2, 8, 8).unfold(3, 8, 8) # x.size() -> (batch, model_dim, n_patches, n_patches)
        self.unfold_shape = p.size()
        p = p.contiguous().view(-1,8,8)
        return p
    
    def inverse(self, p):
        if not hasattr(self, 'unfold_shape'):
            raise AttributeError('Patchify needs to be applied to a tensor in ordfer to revert the process.')
        x = p.view(self.unfold_shape)
        output_h = self.unfold_shape[1] * self.unfold_shape[4]
        output_w = self.unfold_shape[2] * self.unfold_shape[5]
        x = x.permute(0,1,4,2,5,3).contiguous()
        x = x.view(3, output_h, output_w)
        return x 

class DCT:
    
    def __init__(self, img_size=224, patch_size=8, n_channels=3, diagonal=0):
        """
        diagonal parameter will decide how much cosine components will be taken into consideration
        while calculating fgsm patches.
        """
        print('DCT class transforms on 3d tensors')
        self.patchify = Patchify(img_size=224, patch_size=8, n_channels=3)
        self.normalize = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        self.mask = torch.flip(torch.triu(torch.ones((8,8)), diagonal=diagonal), dims=[0])
        
    def __call__(self, tensor):
        p, fgsm_coeffs = self.patched_dct(tensor)
        dct_coeffs = self.patchify.inverse(p)
        fgsm_coeffs = self.patchify.inverse(fgsm_coeffs)
        fgsm_coeffs = fgsm_coeffs / fgsm_coeffs.max()
        return dct_coeffs, fgsm_coeffs
    
    def patched_dct(self, tensor):
        p = self.patchify(tensor)
        fgsm_coeff_tensor = torch.zeros(p.shape, dtype=torch.float32)
        for e, patch in enumerate(p):
            
            dct_coeffs = dct.dct_2d(patch, norm='ortho')
            dct_coeffs[0][0] = 0.0
            fgsm_coeffs = self.calculate_fgsm_coeffs(dct_coeffs)
            fgsm_coeff_tensor[e] = fgsm_coeffs
            p[e] = dct_coeffs
        return p, fgsm_coeff_tensor
    
    def calculate_fgsm_coeffs(self, patch):
        sum_patch = sum(patch[self.mask == 1].abs())
        return torch.full((8,8), fill_value=sum_patch)


def identity(x):
    return x

if __name__ == '__main__':
    pass