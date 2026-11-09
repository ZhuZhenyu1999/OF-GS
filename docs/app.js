(() => {
  "use strict";
  const $ = (selector) => document.querySelector(selector);
  const steps = [
    { kicker: "FEATURES TO GEOMETRY", title: "为三维地图建立几何起点。", description: "使用 CLIDD 特征建立跨帧对应，通过 RANSAC / PnP 与 mini-BA 估计初始位姿。8 个 bootstrap 关键帧共同构成初始化窗口，随后进入高斯建图。", input: "输入：单目 RGB 图像序列", output: "输出：关键帧位姿与稀疏几何" },
    { kicker: "COVISIBILITY & LOCAL BA", title: "用共视关系，约束局部几何。", description: "在新关键帧注册后、新高斯插入前执行共视局部 BA。默认每 5 个关键帧触发一次，窗口最多 10 帧，包含最近 5 个有效训练关键帧与 2 个更早的固定锚帧；联合优化可变位姿与共享路标。", input: "约束：重投影、视差与窗口连通性", output: "输出：更新位姿、路标与局部地图" },
    { kicker: "DEPTH-GUIDED GAUSSIANS", title: "让深度先验，落到场景表面。", description: "将单目深度与稀疏几何对齐，结合 CLIDD 多帧稠密特征进行 GuidedMVS 深度估计。Scharr 响应与邻域抑制构建像素采样概率，再根据深度与位姿直接生成三维高斯。", input: "输入：单目深度、稀疏几何与多帧特征", output: "输出：新增三维高斯" },
    { kicker: "JOINT POSE & MAP OPTIMIZATION", title: "在建图中，持续协调位姿与地图。", description: "初始化建图后，每个新增关键帧触发 25 次固定姿态建图，再执行 25 次联合优化。高斯成熟度门控控制位姿梯度参与，结合图像、单目逆深度与稀疏几何损失；序列结束后保存在线建图结果。", input: "过程：固定姿态 → 位姿与高斯联合更新", output: "输出：在线地图与两阶段轨迹" }
  ];
  const views = {
    overview: { src: "assets/scene-comparison.png", alt: "Waymo 第 94 帧街道原图与保存地图的不透明高斯椭球视图并排展示", label: "RGB IMAGE → GAUSSIAN ELLIPSOIDS", description: "由项目脚本绘制的 1σ 不透明几何表面，非 3DGS 合成渲染。", note: "图示来自已保存的地图文件；地图运行来源未核验，不作为本方法的重建质量结论。原图与椭球视图不保证像素级对齐。" },
    detail: { src: "assets/scene-detail.png", alt: "原图与高斯椭球视图中建筑、车辆和路面的局部细节对比", label: "LOCAL DETAIL / ELLIPSOID SURFACES", description: "以裁剪视角重新绘制局部高斯几何，观察椭球的形状与分布。", note: "局部图展示保存的最终地图，不代表处理该帧时的历史状态。地图运行来源未核验；不透明椭球表面与透明度合成的 Gaussian RGB 渲染不同。" },
    depth: { src: "assets/relative-depth.png", alt: "Depth Anything V2 相对逆深度伪彩图，暖色表示近处，冷色表示远处", label: "DEPTH ANYTHING V2 / RELATIVE INVERSE DEPTH", description: "对原始 PNG 独立导出的相对逆深度；暖色近、冷色远。", note: "深度图按单张图像的最小值与最大值映射为 TURBO 伪彩色，没有米制单位，未与 SLAM 或 GT 尺度对齐，也未经过 SLAM 的去畸变处理。" }
  };
  const commands = {
    kitti: { comment: "# KITTI 07 · 左彩色相机 P2", lines: ["python slam.py", "  -s /path/to/KITTI/07", "  --dataset_type kitti", "  --kitti_camera 2", "  --test_hold 8", "  --viewer_mode none", "  -m results/kitti_07"], note: "输入为已校正的 pinhole 图像；每个序列读取自己的 calib.txt。" },
    waymo: { comment: "# Waymo 153495 · FRONT 相机", lines: ["python slam.py", "  -s /path/to/waymo/153495", "  --dataset_type waymo --camera_name FRONT", "  --waymo_image_model opencv", "  --test_hold 8 --viewer_mode none", "  -m results/waymo_153495"], note: "opencv 适用于带原始畸变的 PNG；图像已去畸变且内参匹配时，改用 pinhole。" },
    dl3dv: { comment: "# DL3DV · 单序列运行", lines: ["python slam.py", "  -s /path/to/dl3dv/1", "  --dataset_type dl3dv", "  --test_hold 8", "  --viewer_mode none", "  -m results/dl3dv_1"], note: "序列目录包含 rgb/、intrinsics.json 与 cameras.json。请核对导出的坐标轴与四元数约定。" },
    tum: { comment: "# TUM · 单序列运行", lines: ["python slam.py", "  -s /path/to/tum/SEQUENCE", "  --dataset_type tum", "  --test_hold 8", "  --viewer_mode none", "  -m results/tum_sequence"], note: "GT pose 与数据集 depth 仅用于评测，不参与跟踪或建图优化。" }
  };

  // All tabs support click, arrow keys, Home and End with a single tab stop.
  function wireTabs(selector, onSelect) {
    const tabs = [...document.querySelectorAll(selector)];
    const vertical = tabs[0]?.parentElement.getAttribute("aria-orientation") === "vertical";
    const select = (tab, focus = false) => {
      tabs.forEach((item) => {
        const selected = item === tab;
        item.setAttribute("aria-selected", String(selected));
        item.tabIndex = selected ? 0 : -1;
        item.classList.toggle("active", selected);
      });
      const panel = document.getElementById(tab.getAttribute("aria-controls"));
      panel.setAttribute("aria-labelledby", tab.id);
      onSelect(tab);
      if (focus) tab.focus();
    };
    tabs.forEach((tab, index) => {
      tab.addEventListener("click", () => select(tab));
      tab.addEventListener("keydown", (event) => {
        let target;
        if (event.key === "Home") target = 0;
        else if (event.key === "End") target = tabs.length - 1;
        else if (event.key === (vertical ? "ArrowDown" : "ArrowRight")) target = (index + 1) % tabs.length;
        else if (event.key === (vertical ? "ArrowUp" : "ArrowLeft")) target = (index - 1 + tabs.length) % tabs.length;
        else return;
        event.preventDefault();
        select(tabs[target], true);
      });
    });
  }

  wireTabs("[data-step]", (tab) => {
    const index = Number(tab.dataset.step);
    const step = steps[index];
    $("#step-kicker").textContent = step.kicker;
    $("#step-count").textContent = `0${index + 1} / 04`;
    $("#step-title").textContent = step.title;
    $("#step-description").textContent = step.description;
    const output = document.createElement("span");
    output.textContent = step.output;
    $("#step-detail").replaceChildren(document.createTextNode(step.input), output);
  });

  let currentView = views.overview;
  wireTabs("[data-view]", (tab) => {
    currentView = views[tab.dataset.view];
    $("#scene-image").src = currentView.src;
    $("#scene-image").alt = currentView.alt;
    $("#scene-label").textContent = currentView.label;
    $("#scene-description").textContent = currentView.description;
    $("#scene-note").textContent = currentView.note;
  });

  let copyResetTimer;
  let copyRevision = 0;
  wireTabs("[data-dataset]", (tab) => {
    const command = commands[tab.dataset.dataset];
    $("#run-command").textContent = command.lines.join(" \\\n");
    $("#command-comment").textContent = command.comment;
    $("#command-note").textContent = command.note;
    clearTimeout(copyResetTimer);
    copyRevision += 1;
    $("#copy-status").textContent = "在项目根目录执行";
  });

  $("#copy-command").addEventListener("click", async () => {
    const text = $("#run-command").textContent;
    const revision = ++copyRevision;
    let copied = false;
    try {
      await navigator.clipboard.writeText(text);
      copied = true;
    } catch {
      // Support local file previews and browsers that deny the Clipboard API.
      const textarea = document.createElement("textarea");
      textarea.value = text;
      textarea.style.position = "fixed";
      textarea.style.opacity = "0";
      document.body.append(textarea);
      textarea.select();
      try { copied = document.execCommand("copy"); } catch { copied = false; }
      textarea.remove();
      $("#copy-command").focus();
    }
    if (revision !== copyRevision) return;
    $("#copy-status").textContent = copied ? "已复制，可粘贴到终端" : "请选中上方命令，手动复制";
    clearTimeout(copyResetTimer);
    copyResetTimer = setTimeout(() => { $("#copy-status").textContent = "在项目根目录执行"; }, 4500);
  });

  const dialog = $("#image-dialog");
  $("#expand-image").addEventListener("click", () => {
    $("#dialog-image").src = currentView.src;
    $("#dialog-image").alt = currentView.alt;
    $("#dialog-caption").textContent = `${currentView.description} ${currentView.note}`;
    dialog.showModal();
    document.body.classList.add("dialog-open");
  });
  $("#close-dialog").addEventListener("click", () => dialog.close());
  dialog.addEventListener("close", () => document.body.classList.remove("dialog-open"));
  dialog.addEventListener("click", (event) => {
    if (event.target !== dialog) return;
    const bounds = dialog.getBoundingClientRect();
    if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) dialog.close();
  });

  const repositoryUrl = window.SITE_CONFIG?.repositoryUrl?.trim();
  if (repositoryUrl) {
    try {
      const url = new URL(repositoryUrl);
      if (url.protocol === "https:" && url.hostname === "github.com" && /^\/[\w.-]+\/[\w.-]+\/?$/.test(url.pathname)) {
        document.querySelectorAll("[data-repository-link]").forEach((link) => {
          link.href = url.href;
          link.target = "_blank";
          link.rel = "noopener";
        });
        document.querySelectorAll("[data-repository-label]").forEach((label) => { label.textContent = "GitHub"; });
      }
    } catch { /* Keep the working quick-start link when the configuration is invalid. */ }
  }

  if ("IntersectionObserver" in window) {
    const links = [...document.querySelectorAll(".site-header nav a")];
    const observer = new IntersectionObserver((entries) => {
      const visible = entries.filter((entry) => entry.isIntersecting);
      if (!visible.length) return;
      const current = visible[0].target.id;
      links.forEach((link) => {
        const active = link.hash === `#${current}`;
        link.classList.toggle("active", active);
        if (active) link.setAttribute("aria-current", "location");
        else link.removeAttribute("aria-current");
      });
    }, { rootMargin: "-15% 0px -55% 0px", threshold: 0 });
    links.forEach((link) => { const section = document.querySelector(link.hash); if (section) observer.observe(section); });
  }
})();
