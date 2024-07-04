# Introduction

## Abstract

- ***18 May 2024*** models have been integrated into huggging face
- ***18 May 2024*** This is preliminary released code, including training code and checkpoints.

## Project file structure

```
├── dataset 
|  ├── AID
|  |  ├── class_1
|  |  |     └── images
|  |  ├── dataset.txt
|  |  ├── test.txt
|  |  └── train.txt
|  ├── NWPU_RESISC45
|  |  ├── class_1
|  |  |     └── images
|  |  ├── dataset.txt
|  |  ├── test.txt
|  |  └── train.txt
|  └── UCMD
|     ├── class_1
|     |     └── images
|     ├── dataset.txt
|     ├── test.txt
|     └── train.txt
├── dataset.py 
├── logs
|  └── base_distill
├── loss.py
├── models
|  ├── base_distill
|  |  ├── base_distall.py
|  |  ├── student.py
|  |  ├── teacher.py
|  |  ├── tool.py
|  ├── base_vit
|  |  ├── base_vit.py
|  |  └── tool.py
|  └── common.py
├── models_data
|  ├── base_distill
|  |  ├── config.yaml # train details config
|  |  ├── student
|  |  |  ├── config.json
|  |  |  └── TinyCLIP-ViT-8M-16-Text-3M-YFCC15M.pt
|  |  └── teacher
|  |  |  ├── trained_teacher_params.pt # such as base_vit_UCMD_128_0.9920.pt
|  |     └── config.json
|  └── base_vit
|     ├── config.yaml  # train details config
|     └── TinyCLIP-ViT-39M-16-Text-19M-YFCC15M
|        ├── TinyCLIP-ViT-39M-16-Text-19M-YFCC15M.pt
|        └── config.json
├── optimizer.py
├── ReadMe.md
├── requirements.txt
├── train.py
├── train_server.py
└── utils.py
```

## Running

### Preliminary

1. Configure the environment according to requirements.txt
2.  Download the dataset, such as UCMD, unzip it to the dataset UCMD directory
3. Download the pre-training weights: include **TinyCLIP-ViT-39M-16-Text-19M-YFCC15M.pt** and  **TinyCLIP-ViT-8M-16-Text-3M-YFCC15M.pt** from [here](https://github.com/wkcn/TinyCLIP) 

### Train Teacher Network 

1.  configure **base_vit/config.yaml**
2.  Run **train.py** if you have a single GPU environment, otherwise run **train_server.py** after configuring **CUDA_VISIBLE_DEVICES**
3.  Place the trained weights under **logs/base_vit/** to **model_data/base_distill/teacher/**

### Train Distill Network

1. configure **base_distill/config.yaml**
2.  change "opt = get_options(model_name="base_vit")" to "opt = get_options(model_name="base_distill")"
3. run it :wink:

**If you want to test other models, just add the model to model/model_name/ and the relevant weight configuration to model_data/model_name/,Note that the underlying network requires rewriting the show() function and encode() and forward()**




# Acknowledge

Our code is based on [SCFR](https://github.com/xdplay17/SCFR) [TinyClip](https://github.com/wkcn/TinyCLIP) [AMFMN](https://github.com/xiaoyuan1996/AMFMN)[CLIP](https://github.com/openai/CLIP), [OpenCLIP](https://github.com/mlfoundations/open_clip), [d2l.ai](https://zh-v2.d2l.ai/index.html) and [PyTorch](https://github.com/pytorch/pytorch). Thank contributors for their awesome contribution!

