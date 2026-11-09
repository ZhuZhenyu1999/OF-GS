# 第三方代码与模型说明

本仓库保留其基础工程与第三方组件原有的版权和使用限制。根目录 [LICENSE.md](LICENSE.md) 不是给所有第三方组件重新授权，也不把工程变为可自由商用的 MIT 项目。

第三方目录以普通源码文件随仓库提供。部分 CUDA 组件已为当前 SLAM 接口修改；请使用本仓库内版本，不要用 `git submodule update` 或通用上游扩展覆盖。原工作目录没有保留这些副本对应的上游 Git 提交信息，因此下表的来源网址不表示本地副本与上游最新版本完全一致。

| 本地位置 | 上游来源 | 许可记录与用途 |
| --- | --- | --- |
| 根目录核心代码 | [on-the-fly-nvs](https://github.com/graphdeco-inria/on-the-fly-nvs)、[3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) 相关基础工程 | 保留根 [LICENSE.md](LICENSE.md) 中的 Inria 研究/评估用途限制及源码版权头。 |
| `CLIDD-main/` | [HITCSC/CLIDD](https://github.com/HITCSC/CLIDD) | [MIT 许可证](CLIDD-main/LICENSE)。包含 CLIDD 源码及官方 S64 小型权重；权重来源和 SHA256 见 [models/README.md](models/README.md)。 |
| `submodules/Depth-Anything-V2/` | [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) | [Apache-2.0 源码许可证](submodules/Depth-Anything-V2/LICENSE)。默认 ViT-L / Large **权重**使用 CC-BY-NC-4.0，与源码许可证不同，未随仓库上传。 |
| `submodules/diff-gaussian-rasterization/` | [Graphdeco rasterizer](https://github.com/graphdeco-inria/diff-gaussian-rasterization) 及基础工程修改 | [本地许可证](submodules/diff-gaussian-rasterization/LICENSE.md)。保留当前定制的相机梯度、逆深度、高斯标识及优化接口。 |
| `submodules/simple-knn/` | [Inria simple-knn](https://gitlab.inria.fr/bkerbl/simple-knn) 及本地修改 | [补齐的许可证文本](submodules/simple-knn/LICENSE.md)。源码保留 Inria / GRAPHDECO 非商业研究与评估用途声明；当前代码含 `distIndex2` 等定制接口。文本来源核对见下文。 |
| `submodules/fused-ssim/` | [rahul-goel/fused-ssim](https://github.com/rahul-goel/fused-ssim) | [MIT 许可证](submodules/fused-ssim/LICENSE)。 |
| `submodules/diff-gaussian-rasterization/third_party/glm/` | [GLM](https://github.com/g-truc/glm) | 保留 [copying.txt](submodules/diff-gaussian-rasterization/third_party/glm/copying.txt) 中的许可选项。 |
| `dataloaders/read_write_model.py` | [COLMAP](https://github.com/colmap/colmap) | 文件中保留原版权和 BSD 许可条款。 |

## simple-knn 许可文本恢复记录

整理源码时，`simple-knn` 文件的版权头已声明非商业研究/评估用途并引用 `LICENSE.md`，但该文件未在本地副本中保留。此次补齐文本，没有替换 KNN 实现。

- 文本下载自官方 Gaussian-Splatting 项目的 [固定提交 54c035f 的 LICENSE.md](https://raw.githubusercontent.com/graphdeco-inria/gaussian-splatting/54c035f7834b564019656c3e3fcc3646292f727d/LICENSE.md)。
- 与 [simple-knn 公开镜像固定提交 178bc0e 的 LICENSE.md](https://raw.githubusercontent.com/pz0826/simple-knn/178bc0e0be9dc3e514700992f3b915c2543c2266/LICENSE.md) 逐字节核对一致。
- SHA256：`c5ba70a2194af2aefe85dfe3da68608dcb3abd21a3aa53b55aa297c2f0b60eb3`。
- 核对日期：2026-11-09。Inria simple-knn 的原始 raw 文件当时未能直接取得；这条记录仅说明许可文本恢复的来源，不声称已确认本地定制代码的完整上游提交历史。

## 网站材料

`docs/` 中的地图椭球图由项目脚本对保存地图生成。原地图的运行来源未核验，页面已标为几何示意，不用于证明本方法重建质量。相对深度图由 Depth Anything V2 对原始图像单独生成。示例使用 Waymo 场景；数据集的获取与使用仍遵循其官方条款。实验 JSON 保留记录日期、汇总口径与评测限制。

本仓库不附数据集、DA2 大模型权重或第三方组件的商用授权。

## 精简发布范围

公开仓库省略上游演示媒体、CLIDD demo_seq.py、DA2 metric_depth 和演示程序，以及 GLM、fused-ssim 的上游测试/工具；保留运行、安装和 OF-GS 本库回归测试所需文件。对应组件 README 提供上游完整文档链接。
