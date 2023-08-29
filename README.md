# R2GenGPT: Radiology Report Generation with Frozen LLMs

## Introduction
![overview](https://github.com/wang-zhanyu/R2GenGPT/blob/main/images/align.png)

## Getting Started
### Installation

**1. Prepare the code and the environment**

Git clone our repository and install the requirements.

```bash
https://github.com/wang-zhanyu/R2GenGPT.git
cd R2GenGPT
pip install -r requirements.txt
```


**2. Prepare the training dataset**

We trained R2GenGPT using the same data as [R2Gen](https://github.com/cuhksz-nlp/R2Gen) on iu-xray and mimic-cxr datasets.

After downloading the data, place it in the ./data folder.

### Training

For shallow alignment

```bash
bash scripts/shallow_run.sh
```

For delta alignment

```bash
bash scripts/delta_run.sh
```

For deep alignment

```bash
bash scripts/shallow_run.sh
```


## Acknowledgement

+ [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) Some codes of this repo are based on MiniGPT-4.
+ [Llama2](https://github.com/facebookresearch/llama) The fantastic language ability of Llama-2 with only 7B parameters is just amazing.


## License
This repository is under [BSD 3-Clause License](LICENSE.md).
