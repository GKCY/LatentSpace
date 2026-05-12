# GRPO: Group Relative Policy Optimization

## 概述

GRPO（Group Relative Policy Optimization）是 DeepSeek 于 2024 年提出的强化学习算法，首次应用于 DeepSeek-R1 系列模型的训练。与传统 PPO 相比，GRPO 通过组内相对收益替代价值网络，大幅降低了训练开销，尤其适合大语言模型的强化学习微调。

> 核心洞察：**同一 prompt 下的多个采样响应天然可以互相作为 baseline，无需单独学习一个价值网络。**

---

## 1. 背景：为什么需要 GRPO？

### PPO 的局限性

PPO（Proximal Policy Optimization）是 RLHF pipeline 中的主流算法，但其存在以下问题：

- **需要 Critic 网络**：PPO 需要一个与 Actor 独立的价值网络 V(s) 来估计状态价值，以计算优势 A(s,a) = r - V(s)。
- **计算开销大**：双网络结构意味着两倍的参数、两倍的内存和两倍的计算量。
- **价值估计偏差**：Critic 的训练本身不稳定，偏差会传播到 Actor 更新。

### GRPO 的解决思路

DeepSeek 意识到：**对于大语言模型，同一个 prompt 下的多采样响应天然具有可比性**。每个响应的"好坏"不需要一个学出来的价值函数来评判——组内其他响应就是最好的参照物。

---

## 2. 核心算法

### 2.1 组内归一化优势函数（Group-relative Advantage）

给定一个 prompt $q$，从当前策略 $\pi_\theta$ 采样 $G$ 个响应 $\{a_1, a_2, \ldots, a_G\}$，每个响应得到一个奖励 $\{r_1, r_2, \ldots, r_G\}$。

定义组内归一化优势：

$$\hat{A}_i = \frac{r_i - \mu_G}{\sigma_G + \epsilon}$$

其中：

- $\mu_G = \frac{1}{G}\sum_{j=1}^{G} r_j$ — 组内奖励均值（作为 baseline）
- $\sigma_G = \sqrt{\frac{1}{G}\sum_{j=1}^{G}(r_j - \mu_G)^2}$ — 组内标准差
- $\epsilon$ — 很小常数（通常 $10^{-8}$），防止除零

**直观理解**：这个公式把组内奖励变成均值为 0、标准差为 1 的分布。每个响应的优势就是它"高出组内平均多少个标准差"。

### 2.2 策略梯度目标

得到优势后，GRPO 采用标准策略梯度的加权形式：

$$\mathcal{L}(\theta) = \mathbb{E}_{q \sim p(Q), \{a_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(·|q)} \left[ \frac{1}{G}\sum_{i=1}^{G} \hat{A}_i \cdot \nabla_\theta \log \pi_\theta(a_i | q) \right]$$

其中 $\pi_{\theta_{\text{old}}}$ 是采样时的旧策略。

### 2.3 裁剪的策略比目标（Clipped Surrogate Objective）

为防止更新步幅过大，GRPO 沿用 PPO 的裁剪机制：

$$\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E} \left[ \frac{1}{G}\sum_{i=1}^{G} \min\left( \frac{\pi_\theta(a_i|q)}{\pi_{\theta_{\text{old}}}(a_i|q)} \cdot \hat{A}_i,\; \text{clip}\left(\frac{\pi_\theta(a_i|q)}{\pi_{\theta_{\text{old}}}(a_i|q)}, 1-\epsilon, 1+\epsilon\right) \cdot \hat{A}_i \right) \right]$$

其中 $\epsilon$ 通常取 $0.2$。

**clip 函数的作用**：当策略比超出 $[1-\epsilon, 1+\epsilon]$ 区间时，对其进行裁剪，防止单步更新过大。

### 2.4 KL 散度正则项

为防止策略偏离原始模型太远，通常加入 KL 正则项：

$$\mathcal{L}_{\text{GRPO}} = \mathcal{L}^{\text{CLIP}}(\theta) - \beta \cdot \mathbb{E}_{q, a} \left[ \text{KL}(\pi_\theta(·|q) \| \pi_{\text{ref}}(·|q)) \right]$$

其中：

- $\beta$ 是 KL 系数（控制正则强度）
- $\pi_{\text{ref}}$ 是 SFT 阶段的参考模型（如 GPT-4 等基座模型）

---

## 3. 算法流程

```
初始化策略网络 πθ 和参考网络 πref
for iteration = 1, 2, ... do
    for each prompt q do
        # Step 1: 组内采样
        从 πθ(·|q) 采样 G 个响应: a1, a2, ..., aG

        # Step 2: 计算奖励
        对每个响应获取奖励: r1, r2, ..., rG

        # Step 3: 计算组内归一化优势
        μG = mean(r1...rG)
        σG = std(r1...rG)
        Âi = (ri - μG) / (σG + ε)

        # Step 4: 计算裁剪损失
        LCLIP = (1/G) * Σ min(ratio_i * Âi, clip(ratio_i, 1-ε, 1+ε) * Âi)

        # Step 5: 加 KL 正则
        L = LCLIP - β * KL(πθ || πref)

        # Step 6: 更新策略
        θ = θ - η * ∇θ L
    end for
end for
```

---

## 4. 与 PPO 的对比

| 特性 | PPO | GRPO |
|------|-----|------|
| **优势函数** | 需要 Critic 网络 V(s) | 组内归一化，无需 Critic |
| **Baseline** | 学出来的价值函数 | 同组采样的均值 |
| **网络结构** | Actor + Critic 双网络 | 仅 Actor 单网络 |
| **计算成本** | 高（双倍参数） | 低（节省约一半显存） |
| **适用场景** | 通用 RL | 大语言模型（尤其推理任务） |
| **样本效率** | 较低（每个 prompt 采样少） | 较高（组内采样共享） |

---

## 5. 代码实现（Python/PyTorch 伪代码）

```python
import torch
import torch.nn.functional as F

def grpo_loss(log_probs, old_log_probs, rewards, epsilon=0.2, kl_coeff=0.01, ref_log_probs=None):
    """
    Args:
        log_probs:        当前策略的对数概率, shape (G,)
        old_log_probs:    旧策略的对数概率, shape (G,)
        rewards:          组内 G 个奖励, shape (G,)
        epsilon:           裁剪参数
        kl_coeff:          KL 正则系数
        ref_log_probs:     参考模型的对数概率, shape (G,)
    Returns:
        loss: GRPO 损失
    """
    G = len(rewards)

    # Step 1: 组内归一化优势
    mean_reward = rewards.mean()
    std_reward = rewards.std() + 1e-8
    advantages = (rewards - mean_reward) / std_reward  # shape (G,)

    # Step 2: 策略比
    ratio = torch.exp(log_probs - old_log_probs)  # shape (G,)

    # Step 3: PPO 裁剪损失
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    clip_loss = -torch.min(surr1, surr2).mean()

    # Step 4: KL 正则项
    if ref_log_probs is not None:
        kl_loss = (torch.exp(ref_log_probs) * (ref_log_probs - log_probs)).sum(dim=-1).mean()
        total_loss = clip_loss + kl_coeff * kl_loss
    else:
        total_loss = clip_loss

    return total_loss
```

---

## 6. GRPO 的变种与改进

### 6.1 带 baseline 调整的 GRPO

有些实现会在优势计算前对奖励进行平移和缩放：

$$\hat{A}_i = \frac{r_i - \text{median}(r_1...r_G)}{\text{mad}(r_1...r_G) + \epsilon}$$

使用中位数和绝对中位差（MAD）替代均值和标准差，增强对离群值的鲁棒性。

### 6.2 多信号组合

在复杂任务中，奖励可能来自多个信号（如正确性 + 格式 + 简洁性），可以先将多维奖励加权求和，再代入 GRPO 流程。

### 6.3 优势函数的其他选择

- **_reward clipping_**：将奖励限制在 $[-H, H]$ 范围内，其中 $H$ 是 episode 最大长度
- **GAE（Generalized Advantage Estimation）**：也可以在组内应用 GAE，但通常 GRPO 的简单归一化已经足够

---

## 7. 适用场景

GRPO 特别适合以下场景：

1. **大语言模型的 RLHF 微调**：如 DeepSeek-R1、DeepSeekMath 等
2. **推理任务强化学习**：如数学证明、代码生成、逻辑推理
3. **多步推理的信用分配**：长思维链（Chain-of-Thought）场景下的奖励分配
4. **资源受限的训练环境**：无法负担双网络计算开销的场景

---

## 8. 参考资料

- DeepSeek-R1 论文（待补充具体 arXiv 链接）
- PPO 原始论文：Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
- OpenRLHF / TRL 框架中的 GRPO 实现

---

## 9. 一句话总结

> **GRPO 的本质：用同 prompt 下的多采样响应的组内均值作为 baseline，计算每个响应超出组内平均多少个标准差作为优势，然后对这个优势做策略梯度 + PPO 裁剪 + KL 正则。** 这让它无需价值网络即可实现稳定的策略更新，大幅降低了大模型 RL 训练的计算成本。
