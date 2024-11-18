import os
import gdown
import torchattacks
import torch_dct
import IQA_pytorch
import argparse

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--nips17_dataset_path', type=str, default='default', help='the path to your nips17 dataset zip')
    parser.add_argument('--extend_modules', type=lambda x: x in ['True', 'true', '1', 'yes'], default=True, help='set to true if you want to extend modules')
    args = parser.parse_args()
    
    print('\n========= INIT REPO SETUP... ==========\n')
    
    
    # get data
    if not os.path.exists('./data/nips17') or 'default' in args:
        print(args)
        print('\t(1) create data folder and place datasets\n')
        if not os.path.exists('./data'):
            os.mkdir('./data')
        os.chdir('./data')
        
        name_nips17 = args.nips17_dataset_path.split('/')[-1]
        target_name_nips17 = 'nips17'
        
        os.system(f'mkdir {target_name_nips17}')
        os.system(f'mv {args.nips17_dataset_path} ./{target_name_nips17}')
        os.system(f'unzip ./{target_name_nips17}/{name_nips17} -d ./{target_name_nips17}')
        os.chdir('../')
    else:
        print('\t(1) passing dataset setup. \n')
    
    
    # get models
    if not os.path.exists('./saves/models'):
        os.chdir('./saves')
        print('\n\t(2) downloading models: this might take a while depending on bandwidth\n')
        try:
            models_url = 'https://drive.google.com/uc?id=1tewpsKAbpud6RTwGTT8_fDa5EFYTdbRW'
            models_output = 'models.zip' 
            gdown.download(models_url, models_output, quiet=False)
            os.system('unzip models.zip')
            os.system('rm models.zip')
            os.chdir('../')
        except Exception as e:
            print('ERROR: There is most likely a connection error. Check below message: \n\n')
            print('==================================================================\n')
            print(f'Exception:\n{e}')
            print('\n==================================================================')
            print('\t(2) passing pretrained model setup. \n')
            os.chdir('../')
    else:
        print('\t(2) passing pretrained model setup. \n')
    
    # extend modules
    if args.extend_modules:
        print('\n\t(3) extending torchattacks and other modules\n')
        
        torchattacks_path = torchattacks.__path__[0] 
        torch_dct_path = torch_dct.__path__[0]
        iqa_pytorch_path = IQA_pytorch.__path__[0]
        
        os.system(f'rm -rf {torchattacks_path+"/"+"attacks"}')
        os.system(f'cp -rf ./package_extensions/torchattacks/attacks {torchattacks_path}')
        os.system(f'cp -rf ./package_extensions/torch_dct/_dct.py {torch_dct_path}/')
        os.system(f'cp -rf ./package_extensions/IQA_pytorch/MAD.py {iqa_pytorch_path}/')
        
    else:
        print('\t(3) passing torchattacks extension.\n')   
    
    # install pytorch-colors
    try:
        import pytorch_colors
    except:
        print('\n\t(4) downloading and installing pytorch-colors\n')
        os.system('git clone https://github.com/jorge-pessoa/pytorch-colors')
        os.chdir('./pytorch-colors/')
        os.system('python3 setup.py install')
        os.chdir('../')
        
    print('\n===== DONE. Refer to README.md to reproduce experiments =====\n')