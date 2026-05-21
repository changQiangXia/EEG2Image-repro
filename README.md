# EEG2Image 消融复现项目

本仓库进行 EEG2Image 的消融研究。

当前版本聚焦于 Object subset 上的四组 GAN loss 消融，并对训练、生成、评估流程做了工程化整理。

## 项目定位

本仓库关注以下内容：

- 官方 checkpoint 验证
- 四组消融训练
- 固定 checkpoint 生成 50,000 张图像
- 固定 Inception Score 评估
- 顺序自动化执行


## 消融设置

四组设置如下：

| 实验名 | mode-seeking loss | DiffAugment |
|---|---|---|
| baseline | 关 | 关 |
| mode_only | 开 | 关 |
| diffaug_only | 关 | 开 |
| full | 开 | 开 |

## 本仓库的主要整理内容

- 统一训练入口：`train_gan.py`
- 固定 checkpoint 生成脚本：`generate_from_ckpt.py`
- 固定 Inception Score 评估脚本：`evaluate_is.py`
- 顺序执行脚本：`run_ablation_suite.sh`
- GPU 运行路径整理
- 输出目录隔离
- 消融结果记录与汇总

## 数据与权重来源

大文件资源通过原始项目给出的外部链接获取。

本次复现使用的资源包括：

- `thoughtvizdataset` 中的 Object subset
- `best_ckpt` 中的官方 GAN checkpoint
- `lstm_kmean` 特征提取器 checkpoint
- `inceptionscore` 评估资源

本地路径约定如下：

- 数据集根目录：`data/b2i_data`
- 官方 GAN checkpoint：`experiments/best_ckpt/ckpt-210`
- Triplet 特征提取器 checkpoint：`lstm_kmean/experiments/best_ckpt`
- Inception 评估资源：`tmp/imagenet`

目录说明见：

- [data/README.md](/root/autodl-tmp/data/README.md:1)
- [experiments/README.md](/root/autodl-tmp/experiments/README.md:1)
- [tmp/README.md](/root/autodl-tmp/tmp/README.md:1)
- [lstm_kmean/experiments/README.md](/root/autodl-tmp/lstm_kmean/experiments/README.md:1)

## 环境

测试环境文件位于：

- [anaconda/tf2.8.yml](/root/autodl-tmp/anaconda/tf2.8.yml:1)

主运行环境为 TensorFlow 2.8 GPU。

## 主流程脚本

当前消融主流程使用以下脚本：

- `train_gan.py`
- `generate_from_ckpt.py`
- `evaluate_is.py`
- `run_ablation_suite.sh`

依赖的主模块包括：

- `model.py`
- `utils.py`
- `runtime_utils.py`
- `eval_utils.py`
- `diff_augment.py`
- `losses.py`
- `lstm_kmean/model.py`

旧训练脚本、旧推理脚本和历史评估脚本已退出主流程。

`lstm_kmean/` 目录保留为特征提取器子模块，用于加载 EEG feature extractor。

## 使用方式

### 1. 官方 checkpoint 验证

```bash
CUDA_VISIBLE_DEVICES=0 python generate_from_ckpt.py \
  --data_root data/b2i_data \
  --output_dir experiments/checkpoint_validation/ckpt_210_official \
  --gan_ckpt_path experiments/best_ckpt/ckpt-210 \
  --test_image_count 50000 \
  --dataset_batch_size 64 \
  --generate_batch_size 64

CUDA_VISIBLE_DEVICES=0 python evaluate_is.py \
  --image_dir experiments/checkpoint_validation/ckpt_210_official/images \
  --output_path experiments/checkpoint_validation/ckpt_210_official/inception_score.json \
  --splits 10
```

### 2. 一键顺序运行四组消融

```bash
CUDA_VISIBLE_DEVICES=0 ./run_ablation_suite.sh
```

### 3. 单组手动运行示例

```bash
CUDA_VISIBLE_DEVICES=0 python train_gan.py \
  --data_root data/b2i_data \
  --output_dir experiments/ablations/baseline \
  --epochs 210 \
  --use_diffaug false \
  --use_mode_loss false
```

## 实验设置

- 数据集：Object dataset
- 训练终点：epoch 210
- 生成规模：50,000 张图像
- 评估指标：Inception Score
- 评估脚本：`evaluate_is.py`
- 生成脚本：`generate_from_ckpt.py`

## 实验结果

| 实验名 | Inception Score |
|---|---:|
| baseline | 3.5868 ± 0.4548 |
| mode_only | 4.4098 ± 0.6822 |
| diffaug_only | 6.7917 ± 0.6947 |
| full | 6.9758 ± 0.7264 |

## 结果分析

- 排序为 `baseline < mode_only < diffaug_only < full`
- 这一排序与目标消融趋势一致
- `mode_only` 相比 `baseline` 有稳定提升
- `diffaug_only` 相比 `baseline` 提升幅度最大
- `full` 取得全组最高分
- `DiffAugment` 是主要增益来源
- `mode-seeking loss` 提供附加增益
- `full` 结果接近官方 checkpoint 验证值 `7.4333 ± 0.7612`

## 当前仓库结构

```text
.
├── README.md
├── anaconda/
├── data/
├── experiments/
├── lstm_kmean/
├── train_gan.py
├── generate_from_ckpt.py
├── evaluate_is.py
├── run_ablation_suite.sh
├── model.py
├── utils.py
├── runtime_utils.py
├── eval_utils.py
├── diff_augment.py
└── losses.py
```

## 附录

相关资源链接：

- EEG2Image best checkpoint：
  <https://iitgnacin-my.sharepoint.com/:u:/g/personal/19210048_iitgn_ac_in/EWC0lT5vEN1c206cJ0tdmdQBkVhvCL5TVnNhBI7cWSTKFg?e=jrpnh9>
- Inception Score 评估资源：
  <https://iitgnacin-my.sharepoint.com/:u:/g/personal/19210048_iitgn_ac_in/EfWLlhNk0CxXqgMnsKgt8k8BxSqflp98ACpl9ZLScWSHtA?e=cEfq0R>
- 预处理后的 ThoughtViz EEG 数据：
  <https://iitgnacin-my.sharepoint.com/:u:/g/personal/19210048_iitgn_ac_in/Ea4Sp2UH__ZbRQGZXu9o-6cByJK4E6E4GtxrcVony9_Q8g?e=bVdyIJ>
- 后续工作 EEGStyleGAN-ADA：
  <https://github.com/prajwalsingh/EEGStyleGAN-ADA>

