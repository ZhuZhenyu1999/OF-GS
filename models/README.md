# 模型权重

完整 SLAM 运行需要本地模型文件，大型权重不保存在 Git 历史中。

| 模型 | 路径 | 获取方式 |
| --- | --- | --- |
| CLIDD S64 | `CLIDD-main/weights/S64.pth` | 仓库附带经过 SHA256 校验的小型官方权重；[官方 v1.0 发布](https://github.com/HITCSC/CLIDD/releases/tag/v1.0) |
| Depth Anything V2 ViT-L | `models/depth_anything_v2_vitl.pth` | [官方模型下载](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth) |

在仓库根目录执行：

```bash
mkdir -p models
curl -L --fail --retry 3 \
  https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth \
  -o models/depth_anything_v2_vitl.pth
sha256sum models/depth_anything_v2_vitl.pth
```

本项目已使用的 DA2 ViT-L 文件 SHA256：

```text
a7ea19fa0ed99244e67b624c72b8580b7e9553043245905be58796a608eb9345
```

CLIDD S64 文件 SHA256：

```text
bcd378383a9c0193247181e4d8a1369a9b30342183d314e23fd55b869dd4e6fd
```

DA2 ViT-L 权重采用 **CC-BY-NC-4.0**，与其 Apache-2.0 源码许可不同。请保留并遵循上游使用条件。

主 SLAM 的 DA2 后端可以尝试下载缺失权重；独立 `scripts/export_da2_depth.py` 要求文件已存在。离线服务器应先完成下载。LPIPS 首次运行也可能下载其感知网络权重。

可选 DA3 后端不属于默认安装，需要另行安装 Depth Anything 3 并准备相应权重。
