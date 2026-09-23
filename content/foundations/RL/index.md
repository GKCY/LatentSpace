---
title: 强化学习
description: 强化学习算法、Agentic RL、训练系统与工程实践。
---

# 强化学习

强化学习算法、Agentic RL、训练系统与工程实践。

- [[foundations/RL/算法/强化学习的数学基础：从策略梯度到PPO与GRPO|强化学习的数学基础：从策略梯度到 PPO 与 GRPO]]：从期望与对数概率推导策略梯度，串起 baseline、GAE、重要性采样、PPO 裁剪和 GRPO 组内优势。
- [[foundations/RL/recipe分析/deepseek-v4.1-post-training|DeepSeek-V4.1 后训练解读：任务合成、异步 RL 与推理预算]]：从任务与环境扩展解释 Agent 训练流程，分析异步调度、effort 奖励、多教师 OPD 和协作训练的收益与边界。
- [[foundations/RL/系统/训练稳定性/精度/训推一致性/训推一致性诊断：从概率偏差到根因定位|训推一致性诊断：从概率偏差到根因定位]]：用同权重、同前缀实验发现概率差异，再定位数据链路、缓存、数值算子和 MoE 路由，并验证它对训练的影响。
- [[foundations/RL/recipe分析/t1-terminal-agent-rl|T1 论文分析：长程终端 Agent 的 PPO 训练 Recipe]]：分析断言计数奖励、critic、TITO、路由重放与长尾调度，并核查实验收益和复现边界。
- [[foundations/RL/recipe分析/miles-v0.1-production-post-training|从 Miles 看长程 Agentic RL：轨迹、异步与训练闭环]]：沿一次终端任务理解真实轨迹、同题分组、异步调度与概率校正，再看 GLM-5.2 的 64 卡配置与实验证据。
- [[foundations/RL/系统/异步/Partial Rollout|Partial Rollout：异步 LLM RL 中的中断、续写与长尾调度]]：拆解权重同步中断、长尾回收和 Agent 分段三种语义，分析 token provenance、KV cache 与 off-policy 校正。
- [[paper_reading/skill-rl-from-reuse-to-internalization|从 Skill 增强到 Skill 内化：Agent RL 如何把经验变成能力]]：串联十五篇工作，讨论技能如何参与探索、信用分配、参数内化与部署后的持续学习。
- [[foundations/RL/系统/权重同步/TensorHub权重同步机制|TensorHub 论文解读：从权重传输到版本分发服务]]：从权重分发讨论 GPU 等待成本、版本一致性、跨机房新鲜度和增量同步的组合。
- [[foundations/RL/系统/训练稳定性/精度/训推一致性/IcePop：用双侧掩码校正稳定MoE强化学习|IcePop 详解：用双侧掩码校正稳定 MoE 强化学习]]：拆解概率校正、双侧掩码与 PPO clipping，讨论实现语义、实验依据和稳定性的代价。
