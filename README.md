# FastRCW

This repository holds the code for the paper "Towards JPEG-compression invariance for adversarial optimization". It contains commands to replicate all experiments and the survey results.

<br />


## Supported Datasets

- Nips2017 Adversarial Challenge ImageNet Subset. [Find Here](https://www.kaggle.com/competitions/nips-2017-defense-against-adversarial-attack/overview)

<br />

## Models

- ResNet: Pretrained IMGNet model. [He et al 2016](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)
- Inception: Pretrained IMGNet model with depthwise-separable convolutions. [Szegedy et al 2015](https://arxiv.org/pdf/1512.00567.pdf)


<br />

## Requirements

(1) **Install module requirements**

All experiments performed on Python 3.10, torch 2.4.0+cu124 and torchvision 0.19.0+cu124
Download and install torch + torchvision [here](https://pytorch.org/)

Install remaining modules:

```bash
pip install -r requirements.txt
```

(2) **Download Datasets**


- Nips2017 Adversarial Challenge ImageNet Subset. [Find Here](https://www.kaggle.com/competitions/nips-2017-defense-against-adversarial-attack/overview)

Download the datasets for the specified sources.

(3) **Data set-up**

Download the dataset and place it in a folder called 'data' in the root of the repository. The tree to the data set should look like this './data/nips17/'

(4) **Pretrained model set-up**

Pretrained models can be downloaded [here](https://drive.google.com/drive/folders/1G1WvO9NylgV6aUKNEjoatv073KGpM8mk?usp=sharing).
Put the downloaded folder in the folder './saves/'.

(5) **Extend torchattacks and other modules**

- Replace MAD.py in your IQA_pytorch package with the one in the folder package_extensions.
- Replace DISTS_pt.py in your DISTS_pytorch package with the one in the folder package_extensions.

You can use these commands. Replace path/to/package with the path to your package.

```bash
cp package_extensions/IQA_pytorch/MAD.py path/to/IQA_pytorch
```

```bash
cp package_extensions/DISTS_pytorch/DISTS_pt.py path/to/DISTS_pytorch
```

<br />

## Replicate Experiments

Available attacks:

- FastRCW: Reliable attack based on a JPEG approximation and Adaptive Compression Search
- JPEG Iterative FGSM [Shin et al. 17](https://machine-learning-and-security.github.io/papers/mlsec17_paper_54.pdf)
- JPEG Iterative FGSM [Reich et al. 24](https://arxiv.org/abs/2309.06978)
- Fast Adversarial Rounding [Shi et al. 21](https://ieeexplore.ieee.org/document/9428243)


### (2) ASR/CAD comparison with the SOTA

ATTACKS:

This measures the success rate ASR and the average distortion CAD. To change the attack in the run command (see below), change the argument --spatial_adv_type to one of the following values:

- FastRCW: rcw
- JPEG IFGSM (Shin et al.): jifgsm
- JPEG IFGSM (Reich et al.): jifgsm (put --diff_jpeg_type=reich)
- Fast Adversarial Rounding (Shi et al.): far

To check for ASR and CAD, find the log of your run in ./saves/reports/imgnet_classification after your run, 
copy the name of your log directoy (looks like YYYY-DD-MM_ImgNetCNN_model_attack_setting) and run the following command:

```bash
python3 cad_asr.py your_log_dir
```

This will output the ASR and CAD of the run.

#### White-Box Attacks

FastRCW:
```bash
python3 run_pretrained.py --dataset nips17 --model_name resnet --transform pretrained --adversarial True --steps 10000 --spatial_adv_type rcw  --target_mode most_likely --surrogate_model resnet --surrogate_input_size 224 --batchsize 16 --device cuda:0 --is_targeted True  --attack_compression=True --attack_compression_rate 80 --attack_lr 0.00001 --c 0.5
```

Shin JPEG IFGSM:
```bash
python3 run_pretrained.py --dataset nips17 --model_name resnet --transform pretrained --adversarial True --spatial_adv_type jifgsm --surrogate_model resnet --surrogate_input_size 224 --batchsize 16 --device cuda:0 --jifgsm_compr_type shin --attack_compression True --attack_compression_rate 70 --N 6 --is_targeted true --target_mode most_likely --eps 0.0004 --steps 7

```

Reich JPEG IFGSM:
```bash
python3 run_pretrained.py --dataset nips17 --model_name resnet --transform pretrained --adversarial True --spatial_adv_type jifgsm --surrogate_model resnet --surrogate_input_size 224 --batchsize 16 --device cuda:0 --jifgsm_compr_type reich --attack_compression True --attack_compression_rate 70 --N 6 --is_targeted true --target_mode most_likely --eps 0.0004 --steps 7
```

Fast Adversarial Rounding:
```bash
python3 run_pretrained.py --dataset nips17 --model_name resnet --transform pretrained --adversarial True --spatial_adv_type far --surrogate_model resnet --surrogate_input_size 224 --batchsize 16 --device cuda:0 --attack_compression True --attack_compression_rate 70 --is_targeted true --target_mode most_likely --eps 9e-05 --far_jpeg_quality 80 --eta 0.3
```


#### Black-Box Attacks

FastRCW:
```bash
python3 run_pretrained.py --dataset nips17 --model_name inception --transform pretrained --adversarial True --steps 10000 --spatial_adv_type rcw  --target_mode most_likely --surrogate_model resnet --surrogate_input_size 224 --batchsize 16 --device cuda:0 --is_targeted True  --attack_compression=True --attack_compression_rate 70 --attack_lr 0.00001 --c 0.5
```

Shin JPEG IFGSM:
```bash
python3 run_pretrained.py --dataset nips17 --model_name inception --transform pretrained --adversarial True --spatial_adv_type jifgsm --surrogate_model resnet --surrogate_input_size 224 --batchsize 16 --device cuda:0 --jifgsm_compr_type shin --attack_compression True --attack_compression_rate 70 --N 6 --is_targeted true --target_mode most_likely --eps 0.0004 --steps 7
```

Reich JPEG IFGSM:
```bash
python3 run_pretrained.py --dataset nips17 --model_name inception --transform pretrained --adversarial True --spatial_adv_type jifgsm --surrogate_model resnet --surrogate_input_size 224 --batchsize 16 --device cuda:0 --jifgsm_compr_type reich --attack_compression True --attack_compression_rate 70 --N 6 --is_targeted true --target_mode most_likely --eps 0.0004 --steps 7
```

Fast Adversarial Rounding:
```bash
python3 run_pretrained.py --dataset nips17 --model_name inception --transform pretrained --adversarial True --spatial_adv_type far --surrogate_model resnet --surrogate_input_size 224 --batchsize 16 --device cuda:0 --attack_compression True --attack_compression_rate 70 --is_targeted true --target_mode most_likely --eps 9e-05 --far_jpeg_quality 80 --eta 0.3
```

#### Robust Models

*PGD-Adversarially-Trained Resnet*

FastRCW:
```bash
python3 run_pretrained.py --dataset nips17 --model_name adv-resnet-pgd --transform pretrained --adversarial True --steps 10000 --spatial_adv_type rcw  --target_mode most_likely --surrogate_model adv-resnet-pgd --surrogate_input_size 224 --batchsize 16 --device cuda:0 --is_targeted True  --attack_compression=True --attack_compression_rate 70 --attack_lr 0.00001 --c 0.5 --adversarial_pretrained=True --adv_pretrained_protocol=pgd
```

Shin JPEG IFGSM:
```bash
python3 run_pretrained.py --dataset nips17 --model_name adv-resnet-pgd --transform pretrained --adversarial True --spatial_adv_type jifgsm --surrogate_model adv-resnet-pgd --surrogate_input_size 224 --batchsize 16 --device cuda:0 --jifgsm_compr_type shin --attack_compression True --attack_compression_rate 70 --N 6 --is_targeted true --target_mode most_likely --eps 0.0008 --steps 7 --adversarial_pretrained=True --adv_pretrained_protocol=pgd
```

Reich JPEG IFGSM:
```bash
python3 run_pretrained.py --dataset nips17 --model_name adv-resnet-pgd --transform pretrained --adversarial True --spatial_adv_type jifgsm --surrogate_model adv-resnet-pgd --surrogate_input_size 224 --batchsize 16 --device cuda:0 --jifgsm_compr_type reich --attack_compression True --attack_compression_rate 70 --N 6 --is_targeted true --target_mode most_likely --eps 0.0008 --steps 7 --adversarial_pretrained=True --adv_pretrained_protocol=pgd
```

Fast Adversarial Rounding:
```bash
python3 run_pretrained.py --dataset nips17 --model_name adv-resnet-pgd --transform pretrained --adversarial True --spatial_adv_type far --surrogate_model adv-resnet-pgd --surrogate_input_size 224 --batchsize 16 --device cuda:0 --attack_compression True --attack_compression_rate 70 --is_targeted true --target_mode most_likely --eps 9e-04 --far_jpeg_quality 80 --eta 0.3 --adversarial_pretrained=True --adv_pretrained_protocol=pgd
```

*FBF-Adversarially-Trained Resnet*


FastRCW:
```bash
python3 run_pretrained.py --dataset nips17 --model_name adv-resnet-pgd --transform pretrained --adversarial True --steps 10000 --spatial_adv_type rcw  --target_mode most_likely --surrogate_model adv-resnet-pgd --surrogate_input_size 224 --batchsize 16 --device cuda:1 --is_targeted True  --attack_compression=True --attack_compression_rate 70 --attack_lr 0.00001 --c 0.5 --adversarial_pretrained=True --adv_pretrained_protocol=pgd
```

Shin JPEG IFGSM:
```bash
python3 run_pretrained.py --dataset nips17 --model_name adv-resnet-fbf --transform pretrained --adversarial True --spatial_adv_type jifgsm --surrogate_model adv-resnet-fbf --surrogate_input_size 224 --batchsize 16 --device cuda:1 --jifgsm_compr_type reich --attack_compression True --attack_compression_rate 70 --N 6 --is_targeted true --target_mode most_likely --eps 0.0004 --steps 7 --adversarial_pretrained=True --adv_pretrained_protocol=fbf
```

Reich JPEG IFGSM:
```bash
python3 run_pretrained.py --dataset nips17 --model_name adv-resnet-pgd --transform pretrained --adversarial True --spatial_adv_type jifgsm --surrogate_model adv-resnet-pgd --surrogate_input_size 224 --batchsize 16 --device cuda:0 --jifgsm_compr_type reich --attack_compression True --attack_compression_rate 70 --N 6 --is_targeted true --target_mode most_likely --eps 0.0004 --steps 7 --adversarial_pretrained=True --adv_pretrained_protocol=pgd
```

Fast Adversarial Rounding:
```bash
python3 run_pretrained.py --dataset nips17 --model_name adv-resnet-fbf --transform pretrained --adversarial True --spatial_adv_type far --surrogate_model adv-resnet-fbf --surrogate_input_size 224 --batchsize 16 --device cuda:1 --attack_compression True --attack_compression_rate 80 --is_targeted true --target_mode most_likely --eps 9e-05 --far_jpeg_quality 80 --eta 0.3 --adversarial_pretrained=True --adv_pretrained_protocol=fbf
```


#### Perceived Quality Experiments

All logs will contain the results of MAD and DISTS. Those can be conveniently printed with the tool cad_asr.py