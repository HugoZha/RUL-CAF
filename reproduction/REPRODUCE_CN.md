# 完整复现教程（中文）

这个文件夹把论文《Constant predictions in multi-condition aero-engine RUL
models: causes, early detection, and condition-adaptive network design》的
实验**从头到尾**打包好了：数据、切分、模型、训练、评估、多种子审计，以及
论文表格背后的原始结果文件。

这里的代码不是照着论文重新写的，而是直接从**当初跑出论文数字的脚本**
整理出来的独立版本。用同样的库版本和设备，训练过程可以逐位复现：
我们核对过 FD001 / 种子 7 的第 1 个 epoch，验证 RMSE 都是 77.56，和
原始日志完全一致。

## 先看名词（人话版）

| 说法 | 人话 |
| --- | --- |
| C-MAPSS FD001–FD004 | 4 组发动机退化仿真数据，FD002/FD004 工况更复杂 |
| RUL | 发动机还能再跑多少圈 |
| cap=125 | 把"寿命标签"最多算到 125 圈，再往上都记 125（NASA 惯例） |
| 滑窗 | 拿连续 w 圈的传感器读数，预测窗口最后一圈的剩余寿命 |
| 85/15 划分 | 训练数据里抽 15% 的窗口做验证，种子固定 42，谁也不许变 |
| collapse（坍缩） | 模型摆烂，所有发动机都预测同一个数，RMSE 冲到 43 左右却"看起来收敛了" |
| 五种子审计 | 同一配置换 5 次随机初始化重跑，看它是不是每次都稳定 |
| PHM08 分数 | 比赛用的打分公式：晚预测罚得特别狠，坍缩模型的分数会爆炸 |

---

## 1. 环境准备

```bash
cd reproduction
python3 -m venv .venv && source .venv/bin/activate   # 可选，但推荐
pip install -r requirements.txt
```

验证过的环境：Python 3.9.6、PyTorch 2.8.0、NumPy 2.0.2、pandas 2.3.3、
scikit-learn 1.6.1，macOS + Apple M 系列芯片（MPS）。CPU 也能跑，只是慢
2–3 倍。脚本会自动选 MPS，没有就退回 CPU；也可以用 `--device cpu|mps|cuda`
手动指定。

**时间预期**：单个完整训练在 FD001–FD003 上约 60–130 分钟，FD004 约
70–350 分钟（MPS）。四组 × 五种子的完整审计约需**两天**。跑 1–2 个 epoch
的冒烟测试不到一分钟。

---

## 2. 第一步：拿数据

原始文件约 43 MB，所以放在 GitHub release 里，不塞进 git：

```bash
python3 prepare_data.py --verify
```

它会自动下载 `c-mapss.zip`（来自 `data-v1` release），解压到
`reproduction/data/`，用 `data_manifest.json` 校验 13 个文本文件的 SHA256，
然后生成切分文件。

如果你本地已经有 NASA 原始数据：

```bash
python3 prepare_data.py --data-dir /你的/数据目录 --skip-download --verify
```

数据是 NASA 公开的美国政府作品，引用 Saxena & Goebel (2008)，使用风险自负。

---

## 3. 第二步：数据怎么切的（这是复现的关键）

`prepare_data.py` 会为每个子集生成一个 `splits/FD00X_split.npz`，里面存着
`train_idx`、`val_idx` 和一个说明文件。切分规则（对应论文 3.1 / 6.1 节）：

1. 训练标签 = 该发动机最大圈数 − 当前圈数，超过 **125** 的截断到 125。
2. 特征 = 3 个操作设置 + 方差不为 0 的传感器（FD001/002/003/004 分别是
   17/24/18/24 个）。
3. 用**训练集自己**的均值和极值做 min-max 归一化，验证/测试不能反过来影响。
4. 每台发动机按长度 w 滑窗，步长 1；标签是窗口**最后一圈**的 RUL。
5. 这些窗口按 **85/15** 用 `random_state=42` 随机切成训练/验证。
6. 测试集：每台测试发动机只取**最后一个窗口**，一台出一个预测；行数不够
   一个窗口的发动机跳过（和原始实验完全一致）。

切完的规模：

| 子集 | 窗口 w | 总窗口 | 训练 | 验证 | 测试发动机 | 特征 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FD001 | 40 | 16,731 | 14,221 | 2,510 | 96 | 17 |
| FD002 | 20 | 48,819 | 41,496 | 7,323 | 259 | 24 |
| FD003 | 30 | 21,820 | 18,547 | 3,273 | 100 | 18 |
| FD004 | 60 | 46,558 | 39,574 | 6,984 | 248 | 24 |

索引文件随仓库一起提供，所以换一台机器、换一个 sklearn 版本，切分也
不会漂。想看规则本身，读 `src/data_loader.py` 里的 `build_split_indices()`。

---

## 4. 第三步：冒烟测试（约 1 分钟）

```bash
python3 train.py --dataset 1 --seed 7 --epochs 1 --patience 1 \
    --split-file splits/FD001_split.npz
```

应该看到：

```
[preprocess] FD001: train=14221 val=2510 test=96 | features=17 | window=40 | cap=125
  ep  1 val=77.56 best=77.56 ...
```

输出落在 `runs/FD001/seed_7/`：`result.json`、`history.json`（逐轮的
RMSE、预测标准差、平台期计数）和 `predictions.npy`。
**只跑 1–2 轮会坍缩是正常的**——这正是论文研究的现象，不是程序坏了。

---

## 5. 第四步：跑一次完整训练 + 评估

```bash
python3 train.py --dataset 1 --seed 42 --split-file splits/FD001_split.npz
python3 train.py --dataset 4 --seed 7  --split-file splits/FD004_split.npz
```

配置写在 `configs.py`（核对数字时不要改）：

| 子集 | 窗口 w | Transformer 层数 L | 注意力头 | 隐层 | Dropout |
| --- | ---: | ---: | ---: | ---: | ---: |
| FD001 | 40 | 1 | 4 | 128 | 0.3 |
| FD002 | 20 | 3 | 4 | 128 | 0.3 |
| FD003 | 30 | 1 | 4 | 128 | 0.3 |
| FD004 | 60 | 3 | 4 | 128 | 0.3 |

共用参数：AdamW（学习率 1e-4，权重衰减 1e-4）、余弦退火、MSE 损失、
batch 32、最多 150 轮、早停耐心 20、梯度裁剪 1.0、传感器 dropout 0.1、
差分增强、均值+最大值双池化。参数量：**107.7 万**（单工况）和
**147.4 万**（多工况），和论文里的 1.08M / 1.47M 对得上。

`train.py` 会算 RMSE、MAE、R²、预测标准差、分段误差、PHM08 分数、平均
归一化注意力熵，以及三条 R5 坍缩警报，全部写进 `result.json`。

---

## 6. 第五步：五种子审计（论文表 8、表 9）

```bash
python3 audit.py --datasets 1,2,3,4 --seeds 42,7,123,2026,3407
```

它按顺序跑完 20 个配置，输出 `results/e1_aggregate_reproduced.json`，
并和论文存档数字并排打印。预期结果（存档值）：

| 子集 | 幸存种子 RMSE | 坍缩率 | PHM08（幸存） |
| --- | ---: | ---: | ---: |
| FD001 | 13.66 ± 0.10 | 0/5 | 307 |
| FD002 | 19.43 ± 0.44 | 0/5 | 2 855 |
| FD003 | 16.09 ± 0.52 | 0/5 | 1 050 |
| FD004 | 19.53 ± 0.07 | 6/8 | 1 783–2 448 |

**FD004 坍缩 6/8 不是复现失败**：那片区域本来就是双稳态，6 个种子会收敛
到"永远预测一个常数"（RMSE≈43.7，预测标准差 < 1e-4）。论文 R5 协议的
做法就是用警报抓出来、换种子重跑，只报幸存种子的统计。换 CPU 或库版本
会有微小数值差异，想严格对齐就用锁定的版本并传 `--split-file`。

只查一组或快速试跑：

```bash
python3 audit.py --datasets 1 --seeds 7,123 --epochs 20 --patience 3
```

（epoch 截断会训练不足、被判坍缩，这只用来验证流程通不通。）

---

## 7. 第六步：三条坍缩警报怎么读

写在 `src/metrics.py`，每次训练结束都会算：

- **警报 A**：预测标准差 < 0.01 圈，或测试 RMSE 与"常数预测器 RMSE"
  （等于 `std(y_test)`，论文公式 (1)）相差不到 2%。
- **警报 B**：平均归一化注意力熵 > 0.99（健康模型 0.70–0.93，坍缩模型
  约 1.0000）。
- **警报 C**：训练过程中，验证 RMSE 连续 5 轮贴在 `std(y_val)` 附近、
  同时预测标准差 < 0.1；记录在结果的 `tripwire_C_plateau` 字段。

任意一条触发，或 RMSE > 40，就标记为 `collapsed`。161 次历史实验的坍缩
矩阵在 `results/collapse_matrix.csv` / `collapse_summary.json`，逐轮的
机制探针在 `results/probe_results.json` / `results/e2_escape_analysis.json`。

---

## 8. 文件清单

```
reproduction/
├── README.md                    简要说明
├── REPRODUCE.md / REPRODUCE_CN.md  本教程（英 / 中）
├── requirements.txt             锁定环境
├── configs.py                   CAS-Fuse 配置 + 警报阈值
├── data_manifest.json           13 个 C-MAPSS 文件的 SHA256
├── prepare_data.py              下载 / 校验 / 生成切分
├── splits/FD00X_split.npz       精确的 85/15 窗口索引
├── src/
│   ├── data_loader.py           读取、截断、归一化、滑窗
│   ├── model.py                 TCN + BiGRU + Transformer 模型
│   └── metrics.py               RMSE/MAE/R²/PHM08 + 三条警报
├── train.py                     单次训练 + 评估
├── audit.py                     多种子审计 + 汇总
├── run_all.sh                   一键全流程
└── results/                     论文表格背后的存档结果
```

---

## 9. 常见问题

- **报"No C-MAPSS files"**：先跑 `prepare_data.py`，或者用 `--data-dir` 指到
  你自己的数据目录。
- **下载太慢**：release 资源在 GitHub CDN 上，可以用下载器断点续传，或直接
  把 NASA 原始文件拷进 `data/` 再加 `--skip-download`。
- **数字对不上**：检查锁定版本，尽量用 MPS；CPU 和 MPS 浮点有细微差别。
  另外一定要传仓库自带的 `--split-file`，保证切分一致。
- **FD004 一直坍缩**：预期之中，这正是论文的核心发现。按 R5 协议增加
  种子数重新抽样，不要去改配置。
