# Introduction
This repository is for **Contextual Transformer Networks for Visual Recognition**. 

CoT is a unified self-attention building block, and acts as an alternative to standard convolutions in ConvNet. As a result, it is feasible to replace convolutions with their CoT counterparts for strengthening vision backbones with contextualized self-attention.

<p align="center">
  <img src="images/framework.jpg" width="800"/>
</p>


# Usage
The code is mainly based on [timm](https://github.com/rwightman/pytorch-image-models).

### Requirement:
* PyTorch 1.8.0+
* Python3.7
* CUDA 10.1+
* [CuPy](https://cupy.dev/). 

### Clone the repository:
```
git clone https://github.com/JDAI-CV/CoTNet.git
```

### Train 
First, download the [ImageNet](https://github.com/facebookarchive/fb.resnet.torch/blob/master/INSTALL.md) dataset. To train CoTNet-50 on ImageNet on a single node with 8 gpus for 350 epochs run:
```
python -m torch.distributed.launch --nproc_per_node=8 train.py --folder ./experiments/cot_experiments/CoTNet-50-350epoch
```
The training scripts for CoTNet (e.g., CoTNet-50) can be found in the [cot_experiments](cot_experiments) folder.

# Inference Time vs. Accuracy
CoTNet models consistently obtain better top-1 accuracy with less inference time than other vision backbones across both default and advanced training setups. In a word, CoTNet models seek better inference time-accuracy trade-offs than existing vision backbones.

<p align="center">
  <img src="images/inference_time.jpg" width="800"/>
</p>

## Citing Contextual Transformer Networks
```
@article{cotnet,
  title={Contextual Transformer Networks for Visual Recognition},
  author={Li, Yehao and Yao, Ting and Pan, Yingwei and Mei, Tao},
  year={2021}
}
```

## Acknowledgements
Thanks the contribution of [timm](https://github.com/rwightman/pytorch-image-models) and awesome PyTorch team.