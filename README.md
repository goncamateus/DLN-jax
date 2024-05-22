# DLN (Lighting Network for Low-Light Image Enhancement) - Jax/Flax implementation
Original: Li-Wen Wang, Zhi-Song Liu, Wan-Chi Siu, and Daniel P. K. Lun
Modified: Mateus G. Machado

This repo provides the Jax/Flax implementation of DLN based on the [original repo](https://github.com/WangLiwen1994/DLN).
I followed the same code structure, neural networks architecture, and hyper-parameters of the original PyTorch Code.

## Complete Architecture - From original Repo
The complete architecture of Deep Lighten Network (DLN) is shown as follows,
The rectangles and cubes denote the operations and feature maps respectively.

![](figures/architecture.png)

### Dataset
- Download the VOC2007 dataset and put it to "datasets/train/VOC2007/" and "datasets/test/VOC2007/".
- Download the LOL dataset and put it to "datasets/train/LOL" and "datasets/test/LOL".

### Training
It needs to manually switch the training dataset: 
1) first, train from the synthesized dataset, 
2) then, load the pretrained model and train from the real dataset
```bash
python train.py 
```

## BibTex
```
@ARTICLE{DLN2020,
  author={Li-Wen Wang and Zhi-Song Liu and Wan-Chi Siu and Daniel P.K. Lun},
  journal={IEEE Transactions on Image Processing}, 
  title={Lightening Network for Low-light Image Enhancement}, 
  year={2020},
  doi={10.1109/TIP.2020.3008396},
}
```