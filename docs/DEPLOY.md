# 部署 TEST2-SLAM 项目网站

网站位于 `docs/`，由静态 HTML、CSS、JavaScript 和项目图片组成，无需 Node.js 依赖或构建步骤。GitHub Pages 仅提供项目展示、图像浏览和命令示例，不能运行 Python、CUDA 或 SLAM 后端。

## 1. 准备 GitHub 仓库

项目仓库为 [ZhuZhenyu1999/OF-GS](https://github.com/ZhuZhenyu1999/OF-GS)，部署工作流支持 `main` 和 `master` 分支。网站源码及工作流已包含在项目目录内；克隆仓库即可取得，无需替换 `.git` 元数据。

需要保留的文件为：

- 整个 `docs/` 目录（包括图片和 `.nojekyll`）。
- `.github/workflows/pages.yml`。

如果项目源代码尚未发布，先检查仓库内容和大小再上传。网站部署只会上传 `docs/`；无需为了展示网站上传数据集、模型权重、实验缓存或其他本地文档。

## 2. 设置仓库入口

`docs/site-config.js` 的 `repositoryUrl` 已设为 `https://github.com/ZhuZhenyu1999/OF-GS`。以后迁移仓库时修改此值即可。

站点资源使用相对路径，支持 GitHub Pages 的项目子路径，无需写死域名。

## 3. 使用 GitHub Actions 发布（推荐）

1. 将网站文件及工作流提交并推送到实际仓库的 `main` 或 `master` 分支。
2. 打开仓库 **Settings → Pages → Build and deployment → Source**，选择 **GitHub Actions**。
3. 打开 **Actions → Deploy project website to GitHub Pages → Run workflow**，选择实际发布分支并运行。若推送时尚未启用 Pages，配置完成后重新运行即可。
4. 等待工作流成功，在部署记录或 **Settings → Pages** 中打开 GitHub 返回的网站地址。

默认项目网址将是 `https://zhuzhenyu1999.github.io/OF-GS/`。它是部署完成后的预期地址，本地生成文件不代表该地址已经上线。

如果单独迁移网站，把 `docs` 与 `.github/workflows/pages.yml` 保持相同目录结构放到目标仓库；不要把 ZIP 文件本身当成网站上传。使用分支发布方式时按下方 `/docs` 说明配置。

使用 Git 的示例（在工程目录之外的新目录中执行，并确保已登录 GitHub）：

```bash
git clone https://github.com/ZhuZhenyu1999/OF-GS.git
cd OF-GS
# 此时将发布包中的 docs 和 .github 两个文件夹复制到当前目录
git add docs .github/workflows/pages.yml
git commit -m "Add project website"
git push -u origin main
```

之后，推送到 `main` / `master` 且修改 `docs/` 或此工作流时，会自动更新网站。其他分支名需要同步修改工作流的 `branches`。仓库的 Pages 环境若设有额外保护规则，请按仓库已有规则完成部署。

工作流仅授予读取仓库、发布 Pages 和获取部署身份所需的权限，不使用自建令牌，也不上传项目根目录。

## 4. 使用分支目录发布（备用）

若不使用 Actions，可在 **Settings → Pages → Source** 中选择 **Deploy from a branch**，选择实际发布分支和 **/docs** 目录，然后保存。此方式与 Actions 发布二选一；选择分支发布时，在 GitHub Actions 页面禁用上述发布工作流，避免两种发布流程混用。

## 5. 本地预览

在项目根目录运行：

```bash
python -m http.server 8000 --bind 127.0.0.1 --directory docs
```

浏览器打开 `http://127.0.0.1:8000`。按 `Ctrl+C` 停止服务。也可以直接打开 `docs/index.html`；涉及浏览器安全上下文的功能（例如剪贴板）建议通过本地 HTTP 预览验证。

## 内容与媒体

页面的功能说明依据项目代码与 README；修改算法后，请同步检查页面说明。展示图片属于静态项目材料，页面本身不会触发 GPU 计算或实时重建。对外发布前，应确认选用材料适合公开，保留项目许可证及所用第三方组件的署名与许可要求。
