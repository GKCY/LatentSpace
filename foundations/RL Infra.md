TODO：
- [ ] foundations/RL/Trainning-Inference Mismatch 训练推理不一致
- [ ] foundations/RL/Weight Refit

训练框架的常见性能优化手段，来自https://zhuanlan.zhihu.com/p/2045509863221064141
DualPipeV 和 EP 的计算-通信 overlap、`torch.compile`、延迟计算 weight gradient、融合的 SwiGLU kernel、expert dispatch 去重，再加上跨 micro-batch 复用 FP8 权重等等
