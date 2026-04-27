# AquaCLIP-QA

AquaCLIP-QA 是一个无参考水下图像质量评价模型。模型融合 CLIP 图像表征、水下属性 prompt 分数和物理退化特征，用于预测归一化 MOS 质量分数。默认主预测路径采用 concat fusion，reliability 分支用于解释分析。

## 模型结构

![模型结构图](docs/images/模型结构图.png)

## 目录结构

```text
aquaclip/
  model.py                  # 模型结构
  data.py                   # 特征读取与归一化
  train.py                  # 训练与评估入口
  calibration.py            # positive-slope affine calibration
  configs/                  # 配置文件
  docs/                     # 说明文档和图片
  reports/                  # 汇总结果表
  outputs/                  # 训练输出、模型和预测结果
```

## 主结果

统一 in-domain test split，共 3541 张图像。PLCC 和 NRMSE 使用线性映射到 normalized MOS 后计算。

| 方法 | SRCC | PLCC | NRMSE |
|---|---:|---:|---:|
| AquaCLIP-QA | **0.8241** | **0.8347** | **0.1161** |
| LIQE | 0.4419 | 0.4645 | 0.1921 |
| MUSIQ | 0.4257 | 0.4249 | 0.1927 |
| TOPIQ-NR | 0.3802 | 0.3880 | 0.1964 |
| HyperIQA | 0.3509 | 0.3724 | 0.1994 |
| CLIP-IQA classic | 0.3344 | 0.3455 | 0.2021 |
| UCIQE | 0.3308 | 0.3615 | 0.2005 |
| NIQE | 0.2862 | 0.2550 | 0.2093 |
| CLIP-IQA pyiqa | 0.2826 | 0.3024 | 0.2100 |
| BRISQUE | 0.2679 | 0.2809 | 0.2092 |
| MANIQA | 0.2429 | 0.2688 | 0.2135 |
| UIQM | 0.1629 | 0.2236 | 0.2196 |

更完整的逐数据集结果见 `reports/`。

## 跨库标定

| 设置 | SRCC | PLCC | NRMSE |
|---|---:|---:|---:|
| 不标定 | 0.3876 | 0.3875 | 0.2349 |
| + 1% positive affine | 0.3881 | 0.3879 | 0.2024 |
| + 5% positive affine | 0.3874 | 0.3874 | 0.1972 |

## 快速运行

在本仓库根目录运行：

```powershell
python aquaclip\train.py `
  --train uid2021:v0\outputs\uid2021_clip_scores.csv:v0\outputs\uid2021_physics.csv `
  --output-dir aquaclip\outputs\uid2021 `
  --experiment-name aquaclip_final_uid2021 `
  --variant aquaclip-final `
  --device auto
```

常用模型变体：`aquaclip-final`、`concat`、`full`、`regression-only`、`hybrid-fusion`。
