# Latent Space

大模型、强化学习与分布式训练知识库。内容使用 Obsidian 维护，由 Quartz 5 生成静态站点并部署到 GitHub Pages。

- 在线地址：<https://gkcy.github.io/LatentSpace/>
- 笔记目录：`content/`
- 站点配置：`quartz.config.yaml`

## 本地预览

需要 Node.js 22+ 和 npm 10.9.2+。

```bash
npm ci
npm run plugins
npm run dev
```

浏览器打开 <http://localhost:8080/>。编辑笔记时，直接把 `content/` 作为 Obsidian vault 打开。

生产构建：

```bash
npm run build
```

生成结果位于 `public/`，该目录不提交到 Git。

## 发布

推送到 `main` 后，`.github/workflows/deploy.yml` 会构建并发布站点。首次使用需要在 GitHub 仓库的 **Settings → Pages** 中把 Source 设为 **GitHub Actions**。

## Quartz 来源

仓库内的 Quartz 5 引擎来自 `jackyzha0/quartz` 的 `v5` 分支，导入基线为提交 `075afd3f712da0088a07f5284a7b3aba37dd61b6`。Quartz 自身的 MIT 许可见 `LICENSE-QUARTZ.txt`。

本仓库采用精简 vendor 方式，没有合并 Quartz 上游 Git 历史。升级时应在独立分支中受控同步上游 `v5` 文件并完整构建验证，不要直接假设 `npx quartz upgrade` 会无冲突完成。
