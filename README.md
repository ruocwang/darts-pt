# DARTS-PT
Code accompanying the paper:<br/>
[[ICLR'2021](https://iclr.cc/)] [[Outstanding Paper Award](https://iclr-conf.medium.com/announcing-iclr-2021-outstanding-paper-awards-9ae0514734ab)] [Rethinking Architecture Selection in Differentiable NAS](https://arxiv.org/abs/2108.04392v1)<br/>
Ruochen Wang, Minhao Cheng, Xiangning Chen, Xiaocheng Tang, Cho-Jui Hsieh

Differentiable Neural Architecture Search is one of the most popular Neural Architecture Search (NAS) methods for its search efficiency and simplicity, accomplished by jointly optimizing the model weight and architecture parameters in a weight-sharing supernet via gradient-based algorithms. At the end of the search phase, the operations with the largest architecture parameters will be selected to form the final architecture, with the implicit assumption that the values of architecture parameters reflect the operation strength. While much has been discussed about the supernet's optimization, the architecture selection process has received little attention. We provide empirical and theoretical analysis to show that the magnitude of architecture parameters does not necessarily indicate how much the operation contributes to the supernet's performance. We propose an alternative perturbation-based architecture selection that directly measures each operation's influence on the supernet. We re-evaluate several differentiable NAS methods with the proposed architecture selection and find that it is able to extract significantly improved architectures from the underlying supernets consistently. Furthermore, we find that several failure modes of DARTS can be greatly alleviated with the proposed selection method, indicating that much of the poor generalization observed in DARTS can be attributed to the failure of magnitude-based architecture selection rather than entirely the optimization of its supernet.


## Requirements

```
Python >= 3.7
PyTorch >= 1.5
tensorboard == 2.0.1
gpustat
```

## Experiments on NAS-Bench-201

The scripts for running experiments can be found in the `exp_scripts/` directory.

### Dataset preparation
1. Download the [NAS-Bench-201-v1_0-e61699.pth](https://drive.google.com/file/d/1SKW0Cu0u8-gb18zDpaAGi0f74UdXeGKs/view) and save it under `./data` folder.

2. Install NasBench201 via pip. (Note: We use the `[2020-02-25]` version of the NAS-Bench-201 API. If you have the newer version installed, you might add `hp="200"` to `api.query_by_arch()` in `nasbench201/train_search.py`)
```
pip install nas-bench-201
```


### Running DARTS-PT on NAS-Bench-201

#### Supernet training
The ckpts and logs will be saved to `./experiments/nasbench201/search-{script_name}-{seed}/`. For example, the ckpt dir would be `./experiments/nasbench201/search-darts-201-1/` for the command below.
```
bash darts-201.sh
```

#### Architecture selection (projection)
The projection script loads ckpts from `experiments/nasbench201/{resume_expid}`
```
bash darts-proj-201.sh --resume_epoch 100 --resume_expid search-darts-201-1
```

#### Fix-alpha version (blank-pt):
```
bash blank-201.sh
bash blank-proj-201.sh --resume_expid search-blank-201-1
```


## Experiments on S1-S4

#### Supernet training
The ckpts and logs will be saved to `./experiments/sota/{dataset}/search-{script_name}-{space_id}-{seed}/`. For example, `./experiments/sota/cifar10/search-darts-sota-s3-1/` (script: darts-sota, space: s3, seed: 1).
```
bash darts-sota.sh --space [s1/s2/s3/s4] --dataset [cifar10/cifar100/svhn]
```

#### Architecture selection (projection)
```
bash darts-proj-sota.sh --space [s1/s2/s3/s4] --dataset [cifar10/cifar100/svhn] --resume_expid search-darts-sota-[s1/s2/s3/s4]-2
```

#### Fix-alpha version (blank-pt):
```
bash blank-sota.sh --space [s1/s2/s3/s4] --dataset [cifar10/cifar100/svhn]
bash blank-proj-201.sh --space [s1/s2/s3/s4] --dataset [cifar10/cifar100/svhn] --resume_expid search-blank-sota-[s1/s2/s3/s4]-2
```

#### Evaluation
```
bash eval.sh --arch [genotype_name]
bash eval-c100.sh --arch [genotype_name]
bash eval-svhn.sh --arch [genotype_name]
```


## Expeirments on DARTS Space

#### Supernet training
```
bash darts-sota.sh
```

#### Archtiecture selection (projection)
```
bash darts-proj-sota.sh --resume_expid search-darts-sota-s5-2
```

#### Fix-alpha version (blank-pt)
```
bash blank-sota.sh
bash blank-proj-201.sh --resume_expid search-blank-sota-s5-2
```

#### Evaluation
```
bash eval.sh --arch [genotype_name]
```


## Citation
If you find this project helpful, please consider citing our paper:

Thanks!

```
@inproceedings{
  ruochenwang2021dartspt,
  title={Rethinking Architecture Selection in Differentiable NAS},
  author={Wang, Ruochen and Cheng, Minhao and Chen, Xiangning and Xiaocheng Tang and Hsieh, Cho-Jui},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2021},
  url={https://arxiv.org/abs/2108.04392}
}
```
