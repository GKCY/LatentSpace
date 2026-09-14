---
title: 强化学习
description: 强化学习算法、Agentic RL、训练系统与工程实践。
---

# 强化学习

强化学习算法、Agentic RL、训练系统与工程实践。

- [[foundations/RL/recipe分析/t1-terminal-agent-rl|T1 论文分析：长程终端 Agent 的 PPO 训练 Recipe]]：分析断言计数奖励、critic、TITO、路由重放与长尾调度，并核查实验收益和复现边界。
- [[foundations/RL/recipe分析/miles-v0.1-production-post-training|Miles v0.1 论文分析：长程 Agentic RL 的系统设计与训练 Recipe]]：结合 GLM-5.2 的 64 卡案例，分析异步调度、轨迹一致性、训推校正、低精度、优化器卸载与权重同步。
- [[paper_reading/skill-rl-from-reuse-to-internalization|从 Skill 增强到 Skill 内化：Agent RL 如何把经验变成能力]]：串联十五篇工作，讨论技能如何参与探索、信用分配、参数内化与部署后的持续学习。
- [[foundations/RL/系统/权重同步/TensorHub权重同步机制|TensorHub 论文解读：从权重传输到版本分发服务]]：从权重分发讨论 GPU 等待成本、版本一致性、跨机房新鲜度和增量同步的组合。
- [[foundations/RL/系统/训练稳定性/精度/训推一致性/IcePop：用双侧掩码校正稳定MoE强化学习|IcePop 详解：用双侧掩码校正稳定 MoE 强化学习]]：拆解概率校正、双侧掩码与 PPO clipping，讨论实现语义、实验依据和稳定性的代价。
