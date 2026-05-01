<div align="center">
<h1>InterRVOS: Interaction-Aware Referring Video Object Segmentation</h1>

[**Woojeong Jin**](https://github.com/wooj0216)&emsp;
[**Seongchan Kim**](https://github.com/deep-overflow)&emsp;
[**Jaeho Lee**](https://github.com/jefflee0810)&emsp;
[**Seungryong Kim**](https://cvlab.kaist.ac.kr/members/faculty)&dagger;
<br>
KAIST AI
<br>
&dagger;: Corresponding Author

**CVPR 2026**

<a href="https://arxiv.org/abs/2506.02356v3">
  <img src="https://img.shields.io/badge/arXiv-2506.02356-B31B1B?logo=arxiv&logoColor=white">
</a>
<a href="https://cvlab-kaist.github.io/InterRVOS/">
  <img src="https://img.shields.io/badge/Project_Page-Available-1E90FF">
</a>
<a href="https://huggingface.co/wooj0216/ReVIOSa-4B">
  <img src="https://img.shields.io/badge/🤗 Huggingface Models-Available-6A5ACD" >
</a>
<a href="https://huggingface.co/datasets/wooj0216/InterRVOS-127K">
  <img src="https://img.shields.io/badge/Dataset-Available-20B2AA" >
</a>

</div>

## 📢 News

- [x] Upcoming: InterRVOS-127K dataset and ReVIOSa checkpoints
- [x] Upcoming : Data annotation pipeline
- [x] Released: Training code, inference & evaluation code
- [x] Released: InterRVOS on [ArXiv](https://arxiv.org/abs/2506.02356v3) and [Project Page](https://cvlab-kaist.github.io/InterRVOS/)


## 🎯 Release Progress

- [x] Model checkpoints
- [x] InterRVOS-127K dataset (Training & Evaluation)
- [x] Data annotation pipeline code
- [x] Inference & evaluation code
- [x] Training code


## Overview

This repository contains the code for the paper **InterRVOS: Interaction-aware Referring Video Object Segmentation**.

In this paper, we introduce **Interaction-aware Referring Video Object Segmentation (InterRVOS)**, a novel task that focuses on the modeling of interactions. 
It requires the model to segment the <b>actor</b> and <b>target</b> objects separately, reflecting their asymmetric roles in an interaction. Please refer to the [project page](https://cvlab-kaist.github.io/InterRVOS/) for detailed visualization results.

## Model Download

‼️ We release the pretrained **ReVIOSa-1B** and **ReVIOSa-4B** model on Hugging Face 🤗:
[**ReVIOSa-1B**](https://huggingface.co/wooj0216/ReVIOSa-1B) and [**ReVIOSa-4B**](https://huggingface.co/wooj0216/ReVIOSa-4B)

### 🚀 Quick Start
```
import torch
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import numpy as np
import os

# load the model and tokenizer
path = "wooj0216/ReVIOSa-4B"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

video_folder = "/PATH/TO/VIDEO_FOLDER"
images_paths = os.listdir(video_folder)
images_paths = [os.path.join(video_folder, image_path) for image_name in images_paths]
text_prompts = "<image>Please segment the child reaching out to man."
input_dict = {
    'video': images_paths,
    'text': text_prompts,
    'past_text': '',
    'mask_prompts': None,
    'tokenizer': tokenizer,
}
return_dict = model.predict_forward(**input_dict)
answer = return_dict["prediction"]
masks = return_dict['prediction_masks']
```

## Dataset

‼️ We release our dataset **InterRVOS-127K** model on Hugging Face 🤗:
[**wooj0216/InterRVOS-127K**](https://huggingface.co/datasets/wooj0216/InterRVOS-127K)

## Model Training & Inference

Instructions for training, inference, and evaluation are provided in **[ReVIOSa/README.md](ReVIOSa/README.md)**.

## Data Annotation

Our automatic data-annotation pipeline are provided in the **[data_annotation](data_annotation)**.

## Acknowledgement
This project is based on [Sa2VA](https://github.com/magic-research/Sa2VA). Many thanks to the authors for their great works!

## References
If you find this repository useful, please consider referring to the following paper:
```
@misc{jin2025interrvosinteractionawarereferringvideo,
    title={InterRVOS: Interaction-aware Referring Video Object Segmentation},
    author={Woojeong Jin and Seongchan Kim and Jaeho Lee and Seungryong Kim},
    year={2025},
    eprint={2506.02356},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2506.02356v3},
}
```
