# 简短结果报告

## 实验目标

本次实验比较两种 GAN 主体：

- `simple_gan`
- `dcgan`

两组实验共享同一套数据、同一套 EEG 特征提取器、同一套训练与评估流程。

固定设置如下：

- 数据集：ThoughtViz Object subset
- 训练轮数：210 epoch
- 生成数量：50,000 张图像
- 评估指标：Inception Score
- `mode-seeking loss`：关闭
- `DiffAugment`：关闭

## 结果

| 实验 | Inception Score |
|---|---:|
| `simple_gan` | 4.4490 ± 0.6565 |
| `dcgan` | 3.7009 ± 0.3985 |

## 简要分析

本次对比中，`simple_gan` 的结果更高。

绝对差值约为 `0.7481`。

这个结果说明，在当前这套简化设定下，更简单的 GAN 主体表现更好。

一个直接的理解是：当 `mode-seeking loss` 和 `DiffAugment` 都关闭后，较小的模型结构更适合当前任务规模与数据条件。

这个结论只对应本次设置。

如果后续重新加入增强策略、额外损失项或更长训练，排序有可能发生变化。

## 结果文件

- [simple_gan_eval/inception_score.json](/root/autodl-tmp/experiments/backbone_compare/simple_gan_eval/inception_score.json:1)
- [dcgan_eval/inception_score.json](/root/autodl-tmp/experiments/backbone_compare/dcgan_eval/inception_score.json:1)
