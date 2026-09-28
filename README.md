# 论文代码实现

基于多源数据关联匹配的船舶定位与航行态势感知 —— 工程代码实现。

## 项目简介

本工程落地论文三章的核心算法：

| 章节 | 功能 | 核心方法 |
|------|------|----------|
| 第二章 | 轻量多尺度船舶目标检测 | MSL-DETR（IRMB 骨干 + 双级特征融合 + Transformer） |
| 第三章 | 雷达-ARPA 异构特征协同定位 | 双分支编码 + 多目标对比学习对齐 + 联合优化 |
| 第四章 | 多源数据关联匹配态势感知 | 置信度级联关联 + 三源闭环校验 + 残余关联容错 |

三章串成一条完整感知链路：**检测（看见船）→ 定位（知道在哪）→ 关联（认清是哪条船）**。

## 目录结构

```
论文代码实现/
├── README.md                        # 本文件
├── .gitignore                       # 忽略数据/临时文件/本地记忆
├── make_synthetic.py                # 合成数据生成器（雷达+ARPA/AIS+视觉+真值）
├── 全文代码开发计划.md               # 三章算法模块分解 + 接口定义 + 分阶段实施
├── 数据集调研与制作方案.md           # 开源数据源调研 + 制作流程 + 格式映射
├── 学位答辩现场演示系统方案.md       # 现场演示系统架构 + 场景设计 + 稳定性保障
├── 工程代码构造逻辑_讲解稿.md       # 全文版讲解稿（分节 + 速览提纲）
├── 讲解稿_脱稿口语版.md             # 口语逐字稿（脱稿汇报用）
└── data/                            # 数据目录（gitignore，含合成数据）
```

> 后续算法代码将按 `common/`（坐标转换/数据模型/匈牙利匹配）、`ch2/`、`ch3/`、`ch4/` 组织，逐步加入。

## 数据获取

- **WHUT-MSFVessel**（雷达+AIS+视觉三源同步，最对口第四/三章）
  - 百度网盘：https://pan.baidu.com/s/100KGjcVkTfsBWa3NwGPxIA （密码 **WHUT**）
- **FVessel**（AIS+视频+相机内外参，TITS 2023）
  - 百度网盘：https://pan.baidu.com/s/1-VNeZvWqYh7ESLXQxreCDg （码 **MIPC**）
- **SeaShips**（检测，第二章）：https://tianchi.aliyun.com/dataset/150438

> 真实数据未到位前，用合成数据跑通三章闭环。

## 快速开始

```bash
# 生成合成数据（需 numpy）
python make_synthetic.py --scenes 3 --frames 120 --ships 4 --out data/synthetic
```

合成数据目录说明见生成的 `data/synthetic/README.txt`。
