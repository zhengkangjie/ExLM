# ExLM: Rethinking the Impact of `[MASK]` Tokens in Masked Language Models

This repository contains the **official implementation** of the paper:

> **ExLM: Rethinking the Impact of `[MASK]` Tokens in Masked Language Models**  
> *ICML 2025*  
> **Kangjie Zheng**, Junwei Yang, Siyue Liang, Bin Feng, Zequn Liu, Wei Ju, Zhiping Xiao, Ming Zhang  
> [[PDF]](https://arxiv.org/pdf/2501.13397.pdf)\

---

## Code Base

This implementation is built upon **[DA-Transformer](https://github.com/thu-coai/DA-Transformer)** and **[Fairseq](https://github.com/facebookresearch/fairseq)**.  
We adapt and extend their functionalities to implement our proposed ExLM framework.

---

## Installation

Clone this repository and install it locally:

```bash
git clone <this-repo-url>
cd <this-repo-dir>
python setup.py install
````

---

## Pre-training on Wikipedia and BookCorpus

Below is an example command to run **ExLM** pre-training on the Wikipedia and BookCorpus datasets using **two GPUs**.

```bash
#!/usr/bin/env bash
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=0,1

savepath=/path/to/save/checkpoints
logfile=${savepath}/train.log
datapath=/path/to/datasets

mkdir -p "${savepath}"
cp "$0" "${savepath}/"

echo ">> training"

fairseq-train  $datapath \
    --user-dir fs_plugins \
    --save-dir "${savepath}" \
    \
    --task masked_lm \
    --criterion nat_dag_loss_mlm \
    --arch p_roberta_ext_denselink_base \
    --max-positions 2048 \
    --encoder-normalize-before \
    --ext-state-num 4 \
    \
    --mask-prob 0.15 \
    --leave-unmasked-prob 0 \
    --random-token-prob 0 \
    \
    --sample-break-mode none \
    --tokens-per-sample 128 \
    \
    --optimizer adam \
    --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay \
    --lr 2e-3 \
    --batch-size 64 \
    --update-freq 32 \
    \
    --warmup-updates 1380 \
    --total-num-update 23000 \
    --max-update 23000 \
    \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    \
    --fp16 \
    --fp16-scale-tolerance 0.1 \
    --fp16-scale-window 50 \
    \
    --keep-interval-updates 100 \
    --save-interval-updates 5000 \
    --validate-interval-updates 5000 \
    --no-epoch-checkpoints \
    --save-interval 99999999 \
    --validate-interval 99999999 \
    --num-workers 8 \
    --seed 1 \
    --log-format simple \
    --log-interval 10 \
    --tensorboard-logdir "${savepath}/tsb" > "${logfile}" 2>&1
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{
  zheng2025exlm,
  title={Ex{LM}: Rethinking the Impact of [MASK] Tokens in Masked Language Models},
  author={Kangjie Zheng and Junwei Yang and Siyue Liang and Bin Feng and Zequn Liu and Wei Ju and Zhiping Xiao and Ming Zhang},
  booktitle={Forty-second International Conference on Machine Learning},
  year={2025},
  url={https://openreview.net/forum?id=IYOPfJwduh}
}
```
