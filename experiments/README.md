# experiments

此目录用于放置训练输出、评估输出与官方权重。

常用路径包括：

- `experiments/best_ckpt/`
- `experiments/checkpoint_validation/`
- `experiments/ablations/`

运行实验后生成的文件建议保留在本地，仓库中只保留此说明文件。

## LSTM 与 Residual BiLSTM 对照

当前分支的目标是保持官方 LSTM 基线协议，替换 EEG 特征提取器中的时序编码部分，观察生成结果是否提升。

### 对照设置

| 项目 | 官方 LSTM | Residual BiLSTM |
|---|---|---|
| EEG 编码器 | 官方 LSTM | 官方前向 LSTM 冻结，加一个可训练反向 LSTM 残差分支 |
| 特征维度 | 128 | 128 |
| 条件向量 | `feat_l2norm` | `feat_l2norm` |
| GAN | DCGAN | DCGAN |
| GAN 训练轮数 | 10 | 10 |
| mode-seeking loss | 关闭 | 关闭 |
| DiffAugment | 关闭 | 关闭 |
| 生成数量 | 10,000 | 10,000 |
| 评估指标 | Inception Score | Inception Score |

### 使用的 checkpoint

| 用途 | 路径 |
|---|---|
| 官方 LSTM 特征提取器 | `/root/autodl-tmp/lstm_kmean/experiments/best_ckpt/ckpt-2420` |
| 官方 LSTM GAN | `experiments/formal_probe_lstm_featl2_gan_e10/ckpt/ckpt-10` |
| Residual BiLSTM 特征提取器 | `experiments/formal_resbilstm_lstm_frozen_feature_e20_lr1e4_v2/ckpt/ckpt-17` |
| Residual BiLSTM GAN | `experiments/formal_resbilstm_lstm_frozen_v2_featl2_gan_e10_ckptevery/ckpt/ckpt-10` |

### 正式结果

| seed | 官方 LSTM IS | Residual BiLSTM IS | 差值 |
|---:|---:|---:|---:|
| 45 | 3.9728 ± 0.4932 | 4.0022 ± 0.5559 | +0.0294 |
| 123 | 3.9464 ± 0.4150 | 4.0492 ± 0.5539 | +0.1029 |
| 2026 | 3.9475 ± 0.4558 | 4.0300 ± 0.5431 | +0.0824 |
| 平均 | 3.9556 | 4.0271 | +0.0716 |

三组 seed 均显示 Residual BiLSTM 高于官方 LSTM。这个结果说明，在相同 GAN、相同条件向量、相同生成数量和相同评估脚本下，加入冻结官方前向 LSTM 与可训练反向 LSTM 残差分支，可以把特征端的改动传到最终生成结果。

### 复现入口

正式对照可使用仓库根目录的 `run_lstm_vs_resbilstm.sh`。脚本优先读取仓库内 `data/b2i_data` 和 `lstm_kmean/experiments/best_ckpt/ckpt-2420`，当前机器缺少这些文件时会回退到本次实验使用的 `/root/autodl-tmp` 路径。
