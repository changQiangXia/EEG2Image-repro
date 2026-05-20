# EEG2Image 消融实验结果简报

## 1. 实验范围

本次实验围绕两项训练策略展开：

- mode-seeking loss
- DiffAugment

四组设置如下：

| 实验名 | mode-seeking loss | DiffAugment |
|---|---|---|
| baseline | 关 | 关 |
| mode_only | 开 | 关 |
| diffaug_only | 关 | 开 |
| full | 开 | 开 |

## 2. 评估设置

- 数据来源：
  `thoughtvizdataset` 数据包中的 Object subset，整理后路径为 `data/b2i_data`
- 官方权重来源：
  `best_ckpt` 数据包，使用 `experiments/best_ckpt/ckpt-210`
- Inception 评估资源来源：
  `inceptionscore` 数据包，使用 `tmp/imagenet`
- 数据集：Object dataset
- 训练终点：epoch 210
- 生成规模：50,000 张图像
- 指标：Inception Score
- 评估脚本：`evaluate_is.py`
- 生成脚本：`generate_from_ckpt.py`

## 3. 结果

| 实验名 | Inception Score |
|---|---:|
| baseline | 3.5868 ± 0.4548 |
| mode_only | 4.4098 ± 0.6822 |
| diffaug_only | 6.7917 ± 0.6947 |
| full | 6.9758 ± 0.7264 |

## 4. 简要分析

- 排序呈现 `baseline < mode_only < diffaug_only < full`。
- 这一排序与论文预期一致。
- `mode_only` 相比 `baseline` 提升 0.8230，增幅约 22.9%。
- `diffaug_only` 相比 `baseline` 提升 3.2050，增幅约 89.4%。
- `full` 相比 `diffaug_only` 再升 0.1841，增益规模较小，方向清晰。
- 结果说明 `DiffAugment` 是主要增益来源，`mode-seeking loss` 提供附加增益。
- 四组标准差位于 0.45 到 0.73 区间，波动规模处于常见范围。
- 官方 checkpoint 验证值为 7.4333 ± 0.7612。`full` 结果接近这一参考值，说明当前训练流程已经较好复现目标趋势。

## 5. 结论

本次消融实验得到清晰结论：

- `DiffAugment` 对图像质量提升最明显。
- `mode-seeking loss` 单独使用时也有正向作用。
- 两项策略同时开启时取得全组最高分。
- 当前复现实验已经支持论文中的核心判断。
