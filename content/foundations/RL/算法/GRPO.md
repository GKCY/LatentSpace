# GRPO: Group Relative Policy Optimization

## 概述

GRPO（Group Relative Policy Optimization）由 DeepSeek 在 2024 年的 [DeepSeekMath](https://arxiv.org/abs/2402.03300v3) 论文中提出，并首先用于训练 DeepSeekMath-RL 7B；之后，[DeepSeek-R1 v2](https://arxiv.org/abs/2501.12948v2) 和 DeepSeek-R1-Zero 也采用了 GRPO。

GRPO 的核心作用不是替代奖励函数，而是替代 PPO 中用于估计优势的 value/critic model：对于同一个 prompt，先采样一组响应并分别获得奖励，再用组内奖励统计量构造相对优势。

> 核心直觉：**同一 prompt 下多个响应的奖励可以互相提供 baseline，因此不必额外训练一个 value model；但每个响应仍然需要由规则、环境或 reward model 打分。**

本文以 DeepSeekMath 论文中的原始、token-level、outcome-supervised GRPO 为主，并单独说明 DeepSeek-R1 和现代框架中的不同口径。

---

## 1. 从 PPO 到 GRPO

### 1.1 LLM PPO 中的 value model

PPO 是一类 clipped policy-gradient 方法。典型的 LLM PPO/RLHF 实现通常使用 learned value function，并通过 return 或 GAE（Generalized Advantage Estimation）估计每个 token 的优势：

$$
A_t \approx \operatorname{GAE}(r_{\geq t}, V_\psi).
$$

value model 在工程上可能是独立模型，也可能与 policy 共享部分参数；因此不能笼统地说 PPO 必然需要一个完全独立的 Critic，也不能据此断言参数量、显存和计算量恰好翻倍。不过，在大模型训练中，一个通常与 policy 规模相近的 value model 确实会引入显著的参数、优化器状态、激活和计算开销。

### 1.2 GRPO 的取舍

GRPO 不再学习 $V_\psi$，而是使用同一 prompt 下 $G$ 个 rollout 的奖励构造相对优势。这样可以省去 value model，但代价是每个 prompt 必须生成多个响应。

需要区分三个概念：

- **reward source**：规则、环境、验证器或 reward model，负责给响应打分；
- **group baseline**：同组奖励的统计量，用于把绝对奖励转换为相对优势；
- **reference policy**：用于可选的 KL 正则，防止 policy 偏离参考策略过远。

GRPO 只用 group baseline 替代了 value/critic，并没有自动消除 reward source 或 reference policy。

---

## 2. 原始 GRPO 的核心定义

### 2.1 组采样与相对优势

给定 prompt $q$，先从 rollout policy（记作旧策略 $\pi_{\theta_{\mathrm{old}}}$）采样 $G$ 个响应：

$$
\{o_1,o_2,\ldots,o_G\}\sim\pi_{\theta_{\mathrm{old}}}(\cdot\mid q).
$$

通过 reward source 得到对应奖励 $\{r_1,r_2,\ldots,r_G\}$。DeepSeekMath 原文只写了 $\operatorname{std}$，没有规定有限样本标准差的 correction。为使公式与后文代码一致，本文采用总体标准差约定：

$$
\mu_q=\frac{1}{G}\sum_{j=1}^G r_j,
\qquad
\sigma_q=\sqrt{\frac{1}{G}\sum_{j=1}^G(r_j-\mu_q)^2}.
$$

outcome-supervised GRPO 使用归一化终局奖励作为该响应所有 completion token 的优势：

$$
\hat A_{i,t}=\widetilde r_i
=\frac{r_i-\mu_q}{\sigma_q},
\qquad t=1,\ldots,|o_i|.
$$

实现中需要单独处理 $\sigma_q=0$。下面的伪代码使用阈值 $\delta_{\mathrm{std}}$：当 $\sigma_q\leq\delta_{\mathrm{std}}$ 时令整组优势为 0，否则仍除以实际的 $\sigma_q$。因此，对 $\sigma_q>\delta_{\mathrm{std}}$ 的组，优势均值为 0、总体标准差为 1（忽略浮点误差）。

### 2.2 Token-level 策略比与裁剪目标

原始 DeepSeekMath GRPO 对每个 completion token 计算 importance ratio：

$$
\rho_{i,t}(\theta)=
\frac{\pi_\theta(o_{i,t}\mid q,o_{i,<t})}
{\pi_{\theta_{\mathrm{old}}}(o_{i,t}\mid q,o_{i,<t})}.
$$

对应的 clipped surrogate 为：

$$
J_{\mathrm{clip}}(\theta)=
\mathbb E\left[
\frac{1}{G}\sum_{i=1}^G\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}
\min\left(
\rho_{i,t}\hat A_{i,t},
\operatorname{clip}(\rho_{i,t},1-\epsilon_{\mathrm{clip}},1+\epsilon_{\mathrm{clip}})\hat A_{i,t}
\right)
\right].
$$

clipping 会抑制一部分让 surrogate objective 变化过大的更新，但它并不保证训练后的实际策略比始终落在裁剪区间内。

### 2.3 KL 正则

原始 GRPO formulation 在目标中直接加入 policy 到 reference policy 的 KL 惩罚。论文使用以下逐 token 采样估计量：

$$
d_{i,t}=\log\pi_{\mathrm{ref}}(o_{i,t}\mid q,o_{i,<t})
-\log\pi_\theta(o_{i,t}\mid q,o_{i,<t}),
$$

$$
\widehat D_{\mathrm{KL},i,t}
=\exp(d_{i,t})-d_{i,t}-1.
$$

当 token 从当前 $\pi_\theta$ 采样时，该估计量的期望等于 $D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\mathrm{ref}})$，且单样本值非负。GRPO rollout 实际来自 $\pi_{\theta_{\mathrm{old}}}$：第一次计算 loss 时通常有 $\pi_\theta=\pi_{\theta_{\mathrm{old}}}$；如果同一批 rollout 被复用多个 inner epochs，二者随后不再相等，此时该项应理解为原始目标使用的非负 KL proxy，而不是当前策略 KL 的无偏估计。

完整目标是：

$$
J_{\mathrm{GRPO}}(\theta)=
\mathbb E\left[
\frac{1}{G}\sum_{i=1}^G\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}
\left(
\min\left(
\rho_{i,t}\hat A_{i,t},
\operatorname{clip}(\rho_{i,t},1-\epsilon_{\mathrm{clip}},1+\epsilon_{\mathrm{clip}})\hat A_{i,t}
\right)
-\beta\widehat D_{\mathrm{KL},i,t}
\right)
\right].
$$

$J_{\mathrm{GRPO}}$ 是需要**最大化**的目标。若使用常见的梯度下降优化器，应最小化：

$$
\mathcal L_{\mathrm{GRPO}}=-J_{\mathrm{GRPO}}.
$$

$\pi_{\mathrm{ref}}$ 通常是与 policy 同源的冻结 checkpoint，例如初始 SFT policy；某些训练流程会周期性更新它。它不是 GPT-4 之类的外部教师模型。

---

## 3. 算法流程

```text
初始化 policy πθ
初始化 reference policy πref（若启用 KL）

for training step = 1, 2, ...:
    从数据集中采样一批 prompts

    # rollout policy 在本轮优化期间保持固定
    πold ← πθ

    for each prompt q:
        从 πold(·|q) 采样 G 个 responses
        通过规则、环境、验证器或 reward model 得到 G 个 rewards
        用组内 reward mean/std 计算每个 response 的 advantage
        将 outcome advantage 广播到该 response 的 completion tokens

    计算每个 completion token 的：
        current log-prob、old log-prob
        token-level importance ratio
        clipped policy objective
        reference log-prob 和 sampled KL（若启用）

    J = 按 token、response、group 和 batch 聚合后的 GRPO objective
    loss = -J
    θ ← θ - η · ∇θ loss

    可选：按预设频率更新 πref
```

若一次 rollout 后进行多个 inner optimization epochs，必须固定 $\pi_{\theta_{\mathrm{old}}}$，或至少缓存并固定 rollout 时的 old token log-prob。还需注意：随着当前 policy 偏离 rollout policy，在旧策略样本上计算的 sampled KL 不再是当前策略 KL 的无偏估计。

---

## 4. PyTorch 伪代码

下面实现采用原始 DeepSeekMath 的 token-level、按 response 平均的 loss aggregation，并将 KL 写成可选项。它省略了分布式训练、padding 对齐、截断样本处理和数值溢出保护，但张量维度、符号和 KL 方向与公式一致。

```python
import torch


def grpo_loss(
    log_probs,             # (B, G, T): current policy token log-probs
    old_log_probs,         # (B, G, T): rollout policy token log-probs
    rewards,               # (B, G): one scalar reward per response
    completion_mask,       # (B, G, T): 1 for valid completion tokens
    ref_log_probs=None,    # (B, G, T): reference policy token log-probs
    clip_epsilon=0.2,
    kl_beta=0.0,
    std_epsilon=1e-8,
):
    """Return a scalar loss to minimize."""
    if kl_beta < 0.0:
        raise ValueError("kl_beta must be non-negative")
    if kl_beta != 0.0 and ref_log_probs is None:
        raise ValueError("ref_log_probs is required when KL is enabled")

    rewards = rewards.detach()
    old_log_probs = old_log_probs.detach()
    valid_token = completion_mask.bool()
    if not valid_token.any(dim=-1).all().item():
        raise ValueError("every response must contain at least one token")
    mask = valid_token.to(log_probs.dtype)

    # 1. Group-relative outcome advantage, normalized per prompt.
    reward_mean = rewards.mean(dim=1, keepdim=True)
    reward_std = rewards.std(dim=1, keepdim=True, correction=0)
    safe_std = reward_std.clamp_min(std_epsilon)

    advantages = (rewards - reward_mean) / safe_std
    advantages = torch.where(
        reward_std > std_epsilon,
        advantages,
        torch.zeros_like(advantages),
    )
    advantages = advantages.unsqueeze(-1)  # (B, G, 1), broadcast over T

    # 2. Token-level clipped policy objective.
    log_ratio = (log_probs - old_log_probs).masked_fill(~valid_token, 0.0)
    ratio = torch.exp(log_ratio)
    unclipped = ratio * advantages
    clipped = torch.clamp(
        ratio,
        1.0 - clip_epsilon,
        1.0 + clip_epsilon,
    ) * advantages
    token_objective = torch.minimum(unclipped, clipped)

    # 3. Optional sampled policy-to-reference KL proxy: exp(d) - d - 1.
    if kl_beta != 0.0:
        d = (ref_log_probs.detach() - log_probs).masked_fill(
            ~valid_token,
            0.0,
        )
        sampled_kl = torch.exp(d) - d - 1.0
        token_objective = token_objective - kl_beta * sampled_kl

    # 4. Original GRPO: average tokens within each response, then B and G.
    token_count = mask.sum(dim=-1)
    response_objective = (token_objective * mask).sum(dim=-1) / token_count
    objective = response_objective.mean()

    return -objective
```

这段代码中的 `clip_epsilon` 是 PPO/GRPO 裁剪超参数；`std_epsilon` 则是奖励标准差的硬阈值，小于等于该值的组会被置为零优势。二者不是同一个 $\epsilon$。代码选择 `correction=0` 以匹配本文的总体标准差公式，其他实现也可能使用样本标准差。

默认 `kl_beta=0.0` 只是为了让示例在没有 reference log-prob 时也能调用，并不是 DeepSeekMath-RL 的复现配置；DeepSeekMath-RL 报告的 KL 系数是 $0.04$。同样，`clip_epsilon=0.2` 只是常见实现的示例值，并非 DeepSeekMath 论文披露的固定裁剪参数。

---

## 5. 与典型 LLM PPO 的对比

| 特性 | 典型 LLM PPO | 原始 outcome-supervised GRPO |
|---|---|---|
| 优势估计 | 通常基于 learned value function 和 GAE | 同 prompt 的组内归一化奖励 |
| Critic/value model | 通常需要；可独立或共享部分参数 | 不需要 learned value model |
| Reward source | 规则、环境或 reward model | 同样需要规则、环境或 reward model |
| Reference policy | 常用于 KL 正则 | 原始 formulation 也可使用 |
| Rollout 要求 | 不要求每个 prompt 固定采样一组响应 | 每个 prompt 需要 $G$ 个响应 |
| 计算权衡 | 增加 value model 的训练成本 | 省去 value model，但增加组采样成本 |
| Token 级信用分配 | GAE 可给不同 token 不同优势 | 终局奖励版本给同一响应所有 token 相同优势 |

因此，GRPO 的主要优势是省去 value model，而不是无条件获得“两倍效率”或“更高样本效率”。实际吞吐、显存和样本效率取决于 group size、生成长度、KL/reference、reward model、参数共享和分布式实现。

---

## 6. 不同论文与实现口径

### 6.1 DeepSeekMath 与 DeepSeek-R1 的目标粒度

DeepSeekMath 原论文使用逐 token ratio，并在每个 response 内按 token 平均。DeepSeek-R1 v2 的展示公式使用整条 response 的似然比：

$$
\rho_i^{\mathrm{R1}}
=\frac{\pi_\theta(o_i\mid q)}{\pi_{\theta_{\mathrm{old}}}(o_i\mid q)}
=\exp\left(\sum_{t=1}^{|o_i|}\log\rho_{i,t}\right).
$$

现代框架也可能提供名为 sequence-level 的模式，但定义未必相同。例如当前 TRL 的 sequence-level importance ratio 是有效 token ratio 的几何平均：

$$
\rho_i^{\mathrm{TRL\text{-}seq}}
=\exp\left(\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\log\rho_{i,t}\right),
$$

它不等于 R1 v2 中未经长度归一化的整序列似然比。

因此，阅读或实现 GRPO 时必须说明：

- importance ratio 是 token-level 还是 sequence-level；
- token loss 如何在 response、group 和 batch 中归一化；
- rollout 后会复用多少次数据；
- reference policy 是否启用、何时更新。

这些选择会改变梯度尺度、长度偏差和训练稳定性，不能仅用“GRPO”三个字推断。

### 6.2 KL 不是所有现代 GRPO 的必选项

DeepSeekMath 的原始目标包含 KL 项，DeepSeek-R1 也报告了非零 KL 系数。但现代实现可能将 $\beta$ 设为 0。例如截至 2026 年 8 月，[TRL GRPOTrainer 文档](https://huggingface.co/docs/trl/grpo_trainer) 的主要默认值是：

- `beta=0.0`：不加载 reference model；
- `importance_sampling_level="token"`；
- `loss_type="dapo"`：按全局有效 completion token 数聚合，而不是原始 GRPO 的逐 response 平均；
- `num_iterations=1`；
- `scale_rewards="group"`。

只有选择对应的 `loss_type="grpo"`，才采用原始的 $G^{-1}\sum_i|o_i|^{-1}\sum_t$ 聚合。默认值会随版本变化，使用时应核对对应版本文档。

所以更准确的说法是：**KL 是原始 GRPO formulation 和部分训练配置中的正则项，不是所有 GRPO 实现都必须启用的组成部分。**

### 6.3 Outcome supervision 与 process supervision

本文前面的优势公式属于 outcome supervision：一个终局奖励被广播给响应内所有 token。它不能自动判断长推理链中究竟哪一步正确或错误，因此不提供细粒度的 step-level credit assignment。

DeepSeekMath 还讨论了 process-supervised GRPO：由 process reward model 在若干推理步骤结束处给出奖励，归一化后令某个 token 的优势等于其后续步骤奖励之和。它需要额外的过程奖励信号，不等同于“在组内应用 GAE”。

### 6.4 Reward source 与 iterative GRPO

GRPO 并不限定奖励必须来自人类偏好模型。DeepSeekMath-RL 使用 outcome/process reward model；DeepSeek-R1-Zero 的推理训练使用正确性和格式等规则奖励，并明确没有在推理任务中使用 neural outcome/process reward model。完整 DeepSeek-R1 的推理数据继续使用规则奖励，而通用 helpfulness/safety 数据使用 reward model。因此，GRPO 可以用于 RLHF、RLAIF 或 RLVR，不能将它们混为同一种训练范式。

DeepSeekMath 还提出了 iterative GRPO 训练流程：根据 policy 的新采样结果构造 reward-model 数据，使用包含 10% 历史样本的 replay 机制继续训练 reward model，并在外层迭代中刷新 reference policy。这是围绕核心 GRPO objective 的训练流程扩展，不是组相对优势公式本身的必要组成。

### 6.5 多奖励信号

正确性、格式、简洁性、安全性等奖励可以组合使用，但必须明确各奖励的权重以及“先组合再归一化”还是“分别归一化再组合”。这两种顺序一般不等价，并可能改变训练信号。

---

## 7. 适用场景与局限

GRPO 较适合：

1. **具有可靠奖励或验证器的任务**：数学、代码、结构化推理等；
2. **同一 prompt 可以生成多个候选响应的 LLM RL**；
3. **value model 成为主要训练瓶颈的场景**；
4. **RLVR、RLHF 或 RLAIF 等不同 reward source 的在线 RL 后训练**。

主要局限包括：

- 每个 prompt 需要多个 rollout，生成成本可能很高；
- 如果同组奖励完全相同，归一化优势为 0，该 prompt 不产生 reward-advantage 梯度信号；若启用了 KL 或其他辅助损失，这些项仍可能产生梯度；
- outcome reward 会被广播给整条响应，长 CoT 内部的细粒度信用分配仍然困难；
- 训练效果依赖 reward source 的可靠性，仍可能发生 reward hacking；
- 原始的按响应长度归一化和组内标准差缩放可能引入长度或题目难度相关偏差；[DAPO](https://arxiv.org/abs/2503.14476) 和 [Dr. GRPO](https://arxiv.org/abs/2503.20783) 等工作因此提出了不同的 loss aggregation 与 reward-scaling 方案。

---

## 8. 一句话总结

> **GRPO 用同一 prompt 下多个 rollout 的奖励统计量构造相对优势，从而省去 learned value/critic model；原始 DeepSeekMath 版本把该优势应用到每个 completion token，通过 token-level clipped surrogate 更新 policy，并可选择加入到 reference policy 的 KL 正则。**

它降低的是 value model 带来的训练负担，并不消除 reward source，也不天然保证更高样本效率或更细粒度的推理信用分配。

---

## 9. 参考资料

- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300v3)
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948v2)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)
- [Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/abs/2503.20783)
- [Hugging Face TRL: GRPO Trainer](https://huggingface.co/docs/trl/grpo_trainer)
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)
