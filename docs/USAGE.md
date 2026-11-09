# OF-GS / TEST2-SLAM 使用指南

GitHub 仓库名为 **OF-GS**，代码内部名称为 **TEST2-SLAM**，入口为 `slam.py`。以下命令均从仓库根目录执行；示例数据路径需要替换为实际路径。

## 环境与安装

完整跟踪与建图依赖 Linux + NVIDIA CUDA。即使部分数据读取、预检或离线工具可在 CPU 上运行，也不代表完整 SLAM 支持 CPU 或 Windows。建议使用独立的 Python 3.10+ 环境，并准备匹配 PyTorch 的 CUDA Toolkit（包含 `nvcc`）与 C++ 编译工具。

```bash
git clone https://github.com/ZhuZhenyu1999/OF-GS.git
cd OF-GS
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel ninja
```

下面沿用项目现有的 CUDA 11.8 安装示例。若使用 CUDA 12 等其他版本，请按实际驱动、Toolkit 与 PyTorch 支持情况替换安装源和 CuPy 包；不要同时安装多个 CuPy CUDA 变体。仓库没有完整环境锁文件，以下命令不代表所有最新版本组合都经过测试。

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
python -m pip install cupy-cuda11x
python -m pip install --no-build-isolation -r requirements.txt
python -m pip install -r CLIDD-main/requirements.txt
```

根 `requirements.txt` 同时安装 Python 依赖和下面三个本地 CUDA 扩展。`--no-build-isolation` 使编译过程使用当前环境中已安装的 PyTorch。请确认源码目录完整，勿以不同接口的上游扩展替代本仓库修改版：

- `submodules/diff-gaussian-rasterization`
- `submodules/fused-ssim`
- `submodules/simple-knn`

迁移机器、切换 PyTorch/CUDA 或修改扩展后，重新编译：

```bash
python -m pip install --no-build-isolation --no-deps --force-reinstall \
  ./submodules/diff-gaussian-rasterization \
  ./submodules/fused-ssim \
  ./submodules/simple-knn
```

CLIDD 需要本地 `CLIDD-main/clidd.py`、`CLIDD-main/model/` 与 `triton>=3.2.0`。默认深度后端需要 `submodules/Depth-Anything-V2/depth_anything_v2/` 源码；网页演示所用的 Gradio 并非运行本项目 SLAM 的必需组件。第三方源码作为普通目录随仓库提供，无需执行 Git submodule 更新，也不要用上游版本覆盖本项目修改过的 CUDA 扩展。

## 模型权重

仓库附带官方 CLIDD S64 小权重；DA2 Large 权重需独立下载。确认下列文件存在：

| 模型 | 本地路径 | 官方来源 |
| --- | --- | --- |
| CLIDD S64（默认，仓库附带） | `CLIDD-main/weights/S64.pth` | [CLIDD v1.0 发布](https://github.com/HITCSC/CLIDD/releases/tag/v1.0)，可用于核验或重新下载 |
| Depth Anything V2 Large / ViT-L（另行下载） | `models/depth_anything_v2_vitl.pth` | [官方 Large 模型](https://huggingface.co/depth-anything/Depth-Anything-V2-Large) |

项目记录的 S64 SHA256 为 `bcd378383a9c0193247181e4d8a1369a9b30342183d314e23fd55b869dd4e6fd`。下载后可运行 `sha256sum CLIDD-main/weights/S64.pth` 检查。其他 CLIDD 型号需准备对应权重，并传入 `--clidd_model`；不同源码位置使用 `--clidd_root`。

常规 SLAM 的 DA2 后端在缺少权重时会尝试从 Hugging Face 下载；离线运行请提前准备。独立 DA2 深度导出工具要求本地已有权重，不自动下载。DA2 Large 模型采用上游声明的 CC-BY-NC-4.0，下载与使用须遵守其条款。

## 安装预检

先确认当前解释器能够看到 GPU 和 CLIDD 文件：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "from poses.clidd_adapter import validate_clidd_install; print(validate_clidd_install(None, 'S64'))"
python scripts/calibration_preflight.py --check_local_ba --check_trajectory_stages
python -m pytest tests/test_renderer_smoke.py -q -rs
```

源码预检不加载 GPU 模型，用于检查配套接口。渲染 smoke 测试依赖可用 CUDA 和已编译扩展；若显示 `skipped`，表示对应 GPU 路径未被验证，不能当作完整安装成功。还需使用一个短序列检查数据读取与实际建图。

## 数据集与单序列运行

以下示例指定 `--test_hold 8` 以产生留出帧渲染指标，并关闭浏览器可视化。`--dataset_type auto` 可自动识别受支持格式，但实验建议显式指定类型。数据根目录包含多个序列时，`slam.py` 只选择一个序列；批量实验请使用后文脚本。

### KITTI

推荐导出结构：

```text
KITTI/07/
├── calib.txt
├── rgb/                 # 例如 000000.png；左彩色相机图像
└── gt/                  # 可选，同名 3×4 或 4×4 位姿文本
```

```bash
python slam.py \
  -s /path/to/KITTI/07 --dataset_type kitti --kitti_camera 2 \
  --test_hold 8 --viewer_mode none -m results/kitti_07
```

默认相机 `2` 使用 `calib.txt` 的 P2 左侧 3×3 内参，输入应为已校正的 pinhole 图像；P2 第四列不作为相机内参或 SLAM 初始位姿。也支持 `image_2/`、`image_02/data/` 等目录。GT 必须对应所选相机，默认解释为 `c2w`；导出为 `w2c` 时添加 `--pose_format w2c`。

### Waymo

本项目读取导出的图像与逐帧标定，不直接读取原始 TFRecord：

```text
waymo/SEQUENCE/FRONT/
├── rgb/                 # PNG/JPEG 图像
├── calib/               # 按相同文件 stem 对应的标定
├── gt/                  # 可选，同名 4×4 位姿
└── depth/               # 可选，仅用于评测相关元数据
```

```bash
python slam.py \
  -s /path/to/waymo/SEQUENCE --dataset_type waymo --camera_name FRONT \
  --waymo_image_model opencv --test_hold 8 --viewer_mode none \
  -m results/waymo_sequence
```

`opencv` 适用于仍保留原始畸变的图像，由读取器按标定去畸变；图像已去畸变且内参与其匹配时改用 `--waymo_image_model pinhole`。不能从 PNG 扩展名推断图像是否已去畸变。标定应给出 `fx/fy/cx/cy`，非零畸变时还需明确正确的图像模型。GT 默认解释为 `c2w`，必要时用 `--pose_format` 显式指定。

### DL3DV

```text
dl3dv/1/
├── rgb/
├── intrinsics.json      # 归一化 fx/fy/cx/cy
└── cameras.json         # 可选，相机记录列表
```

```bash
python slam.py \
  -s /path/to/dl3dv/1 --dataset_type dl3dv \
  --test_hold 8 --viewer_mode none -m results/dl3dv_1
```

内参按 `fx/cx × 图像宽度`、`fy/cy × 图像高度` 转为像素单位；逐帧相机记录可覆盖全局内参。`cameras.json` 按 `image_name` 精确匹配，位姿字段为 `cam_quat`、`cam_trans`。默认约定为 `xyzw` 四元数、`c2w` 位姿、OpenCV 右/下/前相机轴；若导出约定不同，使用 `--dl3dv_quaternion_order wxyz` 或 `--pose_format w2c` 等相应设置，并自行核对坐标约定。缺少相机 GT 的帧不计算 ATE。

### TUM

```text
tum/SEQUENCE/
├── rgb.txt              # 时间戳与 RGB 相对路径
├── rgb/
├── groundtruth.txt      # 可选，时间戳与 tx ty tz qx qy qz qw
├── depth.txt            # 可选
└── depth/               # 可选
```

```bash
python slam.py \
  -s /path/to/tum/SEQUENCE --dataset_type tum \
  --test_hold 8 --viewer_mode none -m results/tum_sequence
```

RGB、深度和 GT 按时间戳关联，默认最大间隔为 0.02 秒，可用 `--tum_max_dt` 调整。默认内参按路径中的 `freiburg2` / `freiburg3` 选择，否则采用 Freiburg 1 预设；应保留真实序列名并核对相机内参。深度与 GT 不参与单目跟踪或建图。

### 先检查短序列

可在上述命令添加 `--max_frames 30`，并改用新的输出目录。短序列必须包含足够的有效图像、匹配和初始化关键帧，限定帧数并不保证初始化成功。不要在不同配置之间复用结果目录以混淆历史指标。

## 批量实验

通用批量入口当前支持 Waymo、KITTI、DL3DV。使用 `custom` profile 和显式路径，避免采用代码中既有机器的默认数据目录：

```bash
python scripts/run_all_slam.py \
  --profile custom --dataset_type waymo --waymo_root /path/to/waymo \
  --cuda_visible_devices 0 --waymo_image_model opencv \
  --viewer_mode none --output_root results/waymo_batch

python scripts/run_all_slam.py \
  --profile custom --dataset_type kitti --kitti_root /path/to/KITTI \
  --cuda_visible_devices 0 --viewer_mode none --output_root results/kitti_batch

python scripts/run_all_slam.py \
  --profile custom --dataset_type dl3dv --dl3dv_root /path/to/dl3dv \
  --cuda_visible_devices 0 --viewer_mode none --output_root results/dl3dv_batch
```

每个序列独立启动 Python 进程，按顺序运行，输出单序列结果与 JSON/CSV/XLSX 汇总。单个序列失败会记录并继续；查看汇总中的失败记录，不要只依据最后一个进程输出判断整批成功。

- `--dry_run`：扫描数据并检查计划命令，不执行训练。
- `--skip_existing`：跳过已有指标的序列，使用前确认其配置与本次实验一致。
- `--summary_only`：读取已有结果重建汇总，不能恢复训练失败时没有写出的指标。
- `--seed 0`：固定相关随机种子；不同环境仍可能存在数值差异。

TUM 使用单序列入口；不要向当前通用批量脚本传入不受支持的 `--dataset_type tum`。

## 运行中的浏览器可视化

将单序列命令的 `--viewer_mode none` 改为下列参数：

```bash
--viewer_mode web --ip 127.0.0.1 --port 6009 --viewer_keep_alive
```

在运行机器的浏览器打开 <http://127.0.0.1:6009>。远程计算时使用自己的 SSH 隧道或受控转发；此服务与 SLAM 在同一 Python 进程内运行，与 GitHub Pages 无关。`--viewer_keep_alive` 保留完成后的画面，按 `Ctrl+C` 退出；批量运行通常不要开启，以免每个序列结束后等待退出。

三维视图显示采样的高斯中心和关键帧轨迹，图像面板调用实际高斯渲染器。当前输入没有估计位姿时，渲染可能使用最近关键帧位姿，应检查各面板帧号。可视化与暂停会增加墙钟时间；比较核心建图速度时应关闭可视化。

## 输出与评测解释

主要输出写入 `-m` 指定目录：

| 文件 | 内容 |
| --- | --- |
| `slam_metrics.json` | 汇总指标、数量、时间与运行配置 |
| `slam_render_metrics.csv` | 留出帧 PSNR、SSIM、LPIPS |
| `slam_poses.json` / `slam_frame_ate.csv` | 在线建图最终轨迹及逐关键帧误差 |
| `slam_trajectory_metrics.json` | 两阶段关键帧 ATE 与 Sim(3) 对齐记录 |
| `slam_poses_{stage}.json` / `slam_frame_ate_{stage}.csv` | 分阶段原始/对齐轨迹及误差 |
| `slam_trajectory_comparison.csv` | 同一关键帧在两阶段的姿态与误差对比 |
| `slam_local_ba.json` | BA 接受、跳过、拒绝原因和优化记录 |
| `metadata.json` / `point_clouds/` | 保存地图的元数据与高斯文件 |

`stage` 为 `frontend_miniba` 或 `online_mapping`。前者记录每帧前端 mini-BA 结束时的位姿副本，但前端使用当时在线地图，不能解释为独立的纯前端实验。后者在在线建图结束后记录最新关键帧位姿，包含后续优化对历史帧的更新，也是最终保存与评测使用的状态。当前不再输出额外的 `final_refinement` 阶段。

评测时必须保留以下边界：

- GT 位姿和数据集深度只用于评测，不参与跟踪或建图。
- ATE/RMSE ATE 仅统计有 GT 的关键帧，各阶段分别执行 Sim(3) 对齐；没有 GT 时 ATE 留空。
- 原始单目地图坐标和未对齐位移不天然具有米制尺度；不能把所有导出的数值直接当作米。
- `--test_hold 8` 选择 frame id 为 8、16、24、… 的图像评测帧；默认 `-1` 不选择留出帧。PSNR/SSIM/LPIPS 不代表对所有输入帧的评测。
- 非关键帧且非评测帧通常不估计位姿，输出不是完整的连续逐帧跟踪轨迹。
- 可视化额外开销不计入核心重建 TIME/FPS，但影响墙钟耗时。比较实验需统一计时口径、硬件、输入与评测设置。
- 保存地图的离线不透明高斯椭球图是几何可视化，不等同于透明度合成的 3DGS RGB 渲染。

## 常见问题与回归测试

- **编译时提示找不到 torch**：先在同一个环境安装 CUDA 匹配的 PyTorch，再使用 `--no-build-isolation` 编译本地扩展。
- **`Installed rasterizer has a stale interface`**：当前源码要求 `GaussianRasterizationSettings` 有 11 项设置；如果已安装模块仍有 13 项，说明加载了旧版本。按“环境与安装”中的命令重新编译扩展，并重新启动 Python 进程。
- **缺少 `distIndex2`、相机梯度或其他 CUDA API**：不要安装通用 upstream 包替代本地定制扩展；重新编译仓库中对应目录。
- **找不到权重**：核对 [模型文件和校验值](../models/README.md)，离线服务器提前下载 DA2 与 LPIPS 所需权重。
- **批量运行找不到数据**：使用 `--profile custom` 并显式传入 `--waymo_root`、`--kitti_root` 或 `--dl3dv_root`。历史 `server/local` profile 包含原实验机器的默认目录，其他机器不应直接照用。

运行源码预检和现有回归测试：

```bash
python scripts/calibration_preflight.py --check_local_ba --check_trajectory_stages
python -m pytest tests -q -rs
```

GitHub 的 `Source and CPU checks` 工作流在 CPU 环境运行源码预检和回归测试。CUDA 测试会跳过；CPU 检查不能替代 Linux CUDA 扩展编译及真实序列的完整建图验证。网页部署使用单独的 Pages 工作流。

## 项目网站

`docs/index.html` 为静态展示页面，部署说明见 [DEPLOY.md](DEPLOY.md)。GitHub Actions 仅上传 `docs/`，不会运行 SLAM、安装模型或上传项目根目录。预期地址为 <https://ZhuZhenyu1999.github.io/OF-GS/>，须在启用 Pages 且部署成功后确认。

## 许可证与组件来源

使用代码前阅读根 [LICENSE.md](../LICENSE.md) 及各组件的许可文件。根许可包含研究、评估和非商业使用限制；不同模型和组件可能另有条件。

上游来源包括 [CLIDD](https://github.com/HITCSC/CLIDD)、[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)、[3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)、[fused-ssim](https://github.com/rahul-goel/fused-ssim)，以及保留 Inria / GRAPHDECO 版权声明的 `simple-knn` 源码。本仓库未为整个工程重新指定统一的 MIT/BSD 许可。

组件来源与许可记录见 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)。CLIDD 源码保留 MIT 许可；该许可不替代整个工程及其他组件的条件。附带的 S64 小权重可对照官方发布核验；DA2 Large 权重独立下载并遵守其 CC-BY-NC-4.0 条款。
