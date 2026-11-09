# OF-GS

基于单目视觉与三维高斯表示的 SLAM 研究工程，面向室外场景中的相机位姿估计、场景建图及可复查的实验记录。

**OF-GS 是 GitHub 仓库名称；当前代码内部名称仍为 TEST2-SLAM，独立运行入口为 `slam.py`。** 终端输出、历史结果和部分文件名保留 TEST2-SLAM，不影响在 `OF-GS/` 目录下运行。

[使用指南](docs/USAGE.md) · [网站部署](docs/DEPLOY.md) · [项目许可证](LICENSE.md)

项目网站采用 `docs/` 静态页面与 GitHub Actions。预期发布地址为 <https://ZhuZhenyu1999.github.io/OF-GS/>；此地址需要在仓库启用 Pages 并成功部署后才可访问，本文不表示部署已经完成。

## 项目内容

系统以 RGB 图像序列为输入，结合几何估计与可微高斯建图：

1. 使用 CLIDD 完成稀疏特征检测、匹配及 GuidedMVS 所需的稠密特征提取。
2. 通过 RANSAC、mini-BA 与 8 个 bootstrap 关键帧建立初始几何和位姿。
3. 在后续关键帧注册后执行共视局部 BA，优化共享路标与关键帧位姿，并同步校正所属高斯。
4. 利用单目深度与 GuidedMVS 引导高斯采样，执行固定姿态的建图优化以及位姿与高斯的联合优化。
5. 保存前端 mini-BA、在线建图两个阶段的轨迹，并输出轨迹误差、留出帧图像质量和运行时间记录。

当前提供 Waymo、KITTI、DL3DV 和 TUM 数据读取入口，支持单序列运行、Waymo/KITTI/DL3DV 批量实验，以及运行中的浏览器可视化。浏览器可视化是 Python SLAM 进程的观察界面；GitHub Pages 网站是静态项目展示，不能执行 CUDA 计算。

本仓库不声明完整系统达到实时性能或最优基准结果。实际效果与运行速度应按数据、硬件、帧选择及评测配置重新验证。

## 目录

```text
OF-GS/
├── slam.py                 # 独立运行入口
├── args.py                 # 命令行参数与配置校验
├── slam_frontend.py        # 前端调度
├── slam_backend.py         # 局部 BA 与后端建图调度
├── slam_outputs.py         # 轨迹、指标和评测输出
├── dataloaders/            # 数据集读取及相机标定
├── poses/                  # 特征、几何估计、mini-BA、局部 BA
├── scene/                  # 深度、关键帧及高斯场景模型
├── visualization/          # 运行中的网页可视化
├── CLIDD-main/             # CLIDD 源码及默认 S64 小权重
├── submodules/             # 深度模型源码与 CUDA 扩展
├── scripts/                # 批量运行、预检与结果工具
├── tests/                  # 回归与 CUDA smoke 测试
└── docs/                   # 项目网站、使用与部署说明
```

仓库附带官方 CLIDD S64 小权重；DA2 大模型权重、数据集、运行结果和本机环境不随源码上传。运行时会使用本地 `models/`、`CLIDD-main/weights/`，并将结果写入指定输出目录。第三方源码按普通目录随仓库提供，无需通过 Git submodule 命令替换本项目定制扩展。

## 快速开始

完整 SLAM 需要 **Linux、NVIDIA GPU、匹配的 CUDA Toolkit / PyTorch / CuPy，以及 C++ 编译工具**。Python 建议使用 3.10 或更高版本；CLIDD 依赖 `triton>=3.2.0`。本项目未提供覆盖所有版本组合的环境锁文件，请先完成[环境安装与预检](docs/USAGE.md#环境与安装)。

```bash
git clone https://github.com/ZhuZhenyu1999/OF-GS.git
cd OF-GS
```

确认以下权重，具体来源见[模型权重](docs/USAGE.md#模型权重)：

| 模型 | 本地位置 |
| --- | --- |
| CLIDD S64（仓库附带） | `CLIDD-main/weights/S64.pth` |
| Depth Anything V2 Large / ViT-L（另行下载） | `models/depth_anything_v2_vitl.pth` |

完成依赖安装、CUDA 扩展编译和数据准备后，在项目根目录运行：

```bash
python slam.py \
  -s /path/to/KITTI/07 \
  --dataset_type kitti --kitti_camera 2 \
  --test_hold 8 --viewer_mode none \
  -m results/kitti_07
```

`/path/to/` 是占位路径，需替换为本机真实数据目录。四种数据集的目录约定、运行命令、批量实验与可视化说明见[使用指南](docs/USAGE.md)。

## 输出与评测边界

- `slam_metrics.json`：PSNR、SSIM、LPIPS、TIME/FPS、ATE 及运行配置等指标。
- `slam_trajectory_metrics.json`：前端与在线建图两阶段的关键帧轨迹指标。
- `slam_poses*.json`、`slam_frame_ate*.csv`：原始/对齐轨迹与逐关键帧误差。
- `slam_render_metrics.csv`：评测帧图像指标；`slam_local_ba.json`：局部 BA 的接受、跳过与拒绝记录。

GT 位姿和数据集深度只用于评测，不参与跟踪或建图。ATE 只统计关键帧，采用 Sim(3) 对齐；没有 GT 时保留原始轨迹，ATE 留空。PSNR/SSIM/LPIPS 仅统计 `--test_hold` 选择的帧；默认 `-1` 不选择图像评测留出帧，示例中的 `8` 选择 frame id 为 8、16、24、… 的帧。

非关键帧且非评测帧通常不估计位姿，不能把输出解读为连续逐帧跟踪轨迹。当前序列结束后直接保存与评测在线建图结果，不再执行额外的全关键帧最终位姿细化。跨方法比较前需统一输入、评测帧、LPIPS 配置、对齐方式和计时范围。

## 许可证与上游组件

请先阅读根目录 [LICENSE.md](LICENSE.md)。其中保留了 on-the-fly-nvs、Hierarchical 3D Gaussians 与原始 3D Gaussian Splatting 相关许可说明，包含非商业研究与评估用途限制；本仓库并非统一采用 MIT 许可证。

本工程使用或基于以下项目与组件，各自的代码和模型仍受其上游许可约束：

- [CLIDD](https://github.com/HITCSC/CLIDD)：特征检测、描述与匹配，源码保留其 MIT 许可证。
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)：单目深度先验。此处使用 Large / ViT-L 权重，上游说明该模型采用 CC-BY-NC-4.0。
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)：高斯表示及相关 CUDA 组件的基础代码；本地渲染扩展包含项目修改。
- [fused-ssim](https://github.com/rahul-goel/fused-ssim)：SSIM 计算组件，保留其本地许可证。
- `simple-knn`：高斯邻域计算 CUDA 扩展，源码保留 Inria / GRAPHDECO 版权及研究用途声明。

组件来源与许可记录见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。保留各目录中的许可证与版权声明；附带的 CLIDD S64 可对照官方发布核验或重新下载，DA2 权重需从官方来源独立获取。不要把某一组件的宽松许可推广为整个工程或所有模型的许可。
