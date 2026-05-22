# EEG2Image 简化 GAN 对比项目

本分支用于一个更小的对比实验。

目标很直接：保留 EEG2Image 的 EEG 特征提取、条件输入、训练流程、生成流程和 Inception Score 评估流程，只比较两种 GAN 主体。

- `simple_gan`：更简单的传统条件 GAN
- `dcgan`：原项目使用的 DCGAN 版本

两组实验都关闭以下组件：

- `mode-seeking loss`
- `DiffAugment`

这样可以把对比重点放在 GAN 主体本身。

## 与原项目的关系

本分支从四组消融版本继续收缩，形成一个两组对比版本。

- 四组消融稳定版本见 tag `original-ablation`
- 当前分支只保留 `simple_gan` 和 `dcgan` 两组实验语义

## 保留内容

当前版本保留以下主流程模块：

- EEG 特征提取器 `lstm_kmean/model.py`
- GAN 训练入口 `train_gan.py`
- checkpoint 生成入口 `generate_from_ckpt.py`
- Inception Score 评估入口 `evaluate_is.py`
- 顺序执行脚本 `run_ablation_suite.sh`
- 主模型定义 `model.py`

## 两组实验设置

| 实验名 | GAN 主体 | mode-seeking loss | DiffAugment |
|---|---|---|---|
| `simple_gan` | 简化条件 GAN | 关 | 关 |
| `dcgan` | 原 EEG2Image DCGAN | 关 | 关 |

输出目录约定如下：

- `experiments/backbone_compare/simple_gan`
- `experiments/backbone_compare/simple_gan_eval`
- `experiments/backbone_compare/dcgan`
- `experiments/backbone_compare/dcgan_eval`

## 数据与权重来源

大文件资源来自原论文作者提供的公开链接。

本项目使用以下资源：

- ThoughtViz 预处理 EEG 数据中的 Object subset
- 官方 GAN checkpoint
- `lstm_kmean` 特征提取器 checkpoint
- Inception Score 评估资源

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

## 使用方式

### 1. 顺序运行两组实验

```bash
CUDA_VISIBLE_DEVICES=0 ./run_ablation_suite.sh
```

脚本会顺序执行：

1. 训练 `simple_gan`
2. 生成 `simple_gan` 的 50,000 张图像
3. 评估 `simple_gan` 的 Inception Score
4. 训练 `dcgan`
5. 生成 `dcgan` 的 50,000 张图像
6. 评估 `dcgan` 的 Inception Score

### 2. 单组手动训练

`simple_gan` 示例：

```bash
CUDA_VISIBLE_DEVICES=0 python train_gan.py \
  --data_root data/b2i_data \
  --output_dir experiments/backbone_compare/simple_gan \
  --gan_variant simple_gan \
  --epochs 210 \
  --use_diffaug false \
  --use_mode_loss false
```

`dcgan` 示例：

```bash
CUDA_VISIBLE_DEVICES=0 python train_gan.py \
  --data_root data/b2i_data \
  --output_dir experiments/backbone_compare/dcgan \
  --gan_variant dcgan \
  --epochs 210 \
  --use_diffaug false \
  --use_mode_loss false
```

### 3. 从 checkpoint 生成图像

`simple_gan` 示例：

```bash
CUDA_VISIBLE_DEVICES=0 python generate_from_ckpt.py \
  --data_root data/b2i_data \
  --output_dir experiments/backbone_compare/simple_gan_eval \
  --gan_ckpt_path experiments/backbone_compare/simple_gan/ckpt/ckpt-210 \
  --gan_variant simple_gan \
  --test_image_count 50000 \
  --dataset_batch_size 64 \
  --generate_batch_size 64
```

`dcgan` 示例：

```bash
CUDA_VISIBLE_DEVICES=0 python generate_from_ckpt.py \
  --data_root data/b2i_data \
  --output_dir experiments/backbone_compare/dcgan_eval \
  --gan_ckpt_path experiments/backbone_compare/dcgan/ckpt/ckpt-210 \
  --gan_variant dcgan \
  --test_image_count 50000 \
  --dataset_batch_size 64 \
  --generate_batch_size 64
```

### 4. 评估 Inception Score

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_is.py \
  --image_dir experiments/backbone_compare/dcgan_eval/images \
  --output_path experiments/backbone_compare/dcgan_eval/inception_score.json \
  --splits 10
```

## 设计说明

当前对比只改变 GAN 主体。

以下部分保持一致：

- 数据
- EEG 特征提取器
- 条件特征拼接方式
- 训练步逻辑
- 优化器设置
- 训练轮数
- 生成数量
- 评估指标

这样更适合观察简单 GAN 与 DCGAN 的主体差异。

## 附录

原始资源链接：

- EEG2Image best checkpoint：
  <https://iitgnacin-my.sharepoint.com/:u:/g/personal/19210048_iitgn_ac_in/EWC0lT5vEN1c206cJ0tdmdQBkVhvCL5TVnNhBI7cWSTKFg?e=jrpnh9>
- Inception Score 评估资源：
  <https://iitgnacin-my.sharepoint.com/:u:/g/personal/19210048_iitgn_ac_in/EfWLlhNk0CxXqgMnsKgt8k8BxSqflp98ACpl9ZLScWSHtA?e=cEfq0R>
- 预处理后的 ThoughtViz EEG 数据：
  <https://iitgnacin-my.sharepoint.com/:u:/g/personal/19210048_iitgn_ac_in/Ea4Sp2UH__ZbRQGZXu9o-6cByJK4E6E4GtxrcVony9_Q8g?e=bVdyIJ>
- 原始项目 EEG2Image：
  <https://github.com/prajwalsingh/EEG2Image>
- 后续工作 EEGStyleGAN-ADA：
  <https://github.com/prajwalsingh/EEGStyleGAN-ADA>
