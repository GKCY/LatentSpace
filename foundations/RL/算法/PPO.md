# PPO：给策略更新装上一条“安全带”

> PPO（Proximal Policy Optimization，近端策略优化）是一类**同策略（on-policy）策略梯度算法**。它用旧策略采样，用 Advantage（优势）判断动作应该被鼓励还是抑制，再通过概率比和裁剪目标，限制一批数据上的更新不要过于激进。

如果只记住一句话：

> **PPO 的裁剪机制不是把新策略硬锁在旧策略附近，而是让“过度更新”无法继续从目标函数中获利。**

本文聚焦最常用的**裁剪版 PPO（PPO-Clip）**。原论文还提出了惩罚版 PPO（PPO-Penalty），但如今大家口中的 PPO，通常默认指前者。

下文仍以中文叙述为主，但 Advantage、Critic、loss 等常见术语直接保留英文；其他英文检索词只在首次出现时标注。公式符号、代码变量名和论文标题保留原样，便于与论文和实现对照。

---

## 1. PPO 到底在解决什么问题？

先想象一个最朴素的策略梯度训练过程：

1. 用当前策略和环境交互，采集一批轨迹；
2. 找出回报高于预期的动作，提高它们的概率；
3. 找出回报低于预期的动作，降低它们的概率；
4. 更新模型，再重新采样。

逻辑没有问题，麻烦出在更新幅度上。

假设旧策略在某个状态下，以 20% 的概率选择动作 A。一次轨迹采样恰好发现 A 的回报不错，梯度便会提高 A 的概率。如果学习率太大，或者在同一批数据上重复训练太多轮，这个概率可能直接从 20% 冲到 90%。

问题是：**这批数据描述的是旧策略生活的世界，不是概率已经变成 90% 后的新世界。**

策略一旦变化，后续访问到的状态、采到的动作、整条轨迹的分布都会跟着变化。旧数据对新策略越来越缺乏代表性，而我们却还在用它继续更新。这就是策略梯度容易“迈出一步，整个数据分布都变了”的原因。

TRPO（Trust Region Policy Optimization）的思路是：给新旧策略之间的 KL 散度加一个硬约束，在一个可信区域内寻找最优更新。但 TRPO 需要较复杂的二阶近似与约束优化。

PPO 试图保留同一个核心思想：

```text
尽量利用这一批采样轨迹多更新几次，
但别让新策略离生成数据的旧策略太远。
```

它把复杂的约束优化换成了一个容易用 Adam 优化器训练的裁剪目标，因此实现简单、效果稳定、工程上也更容易扩展。

---

## 2. 从策略梯度走到 PPO

### 2.1 Advantage：这个动作比“正常发挥”好多少？

策略梯度的核心形式是：

$$
\nabla_\theta J(\theta)
=
\mathbb{E}
\left[
\nabla_\theta \log \pi_\theta(a_t \mid s_t)
A_t
\right]
$$

其中：

- $\pi_\theta(a_t \mid s_t)$：策略在状态 $s_t$ 选择动作 $a_t$ 的概率；
- $A_t$：动作 $a_t$ 相对当前策略平均水平的 Advantage；
- $A_t > 0$：这个动作比预期好，应该提高概率；
- $A_t < 0$：这个动作比预期差，应该降低概率。

为什么不直接使用回报，而要减去一个基准值？

假设某个游戏关卡天然容易，随便操作都能拿 100 分。动作 A 拿了 102 分，虽然绝对分数很高，但它只比正常水平好一点。另一个困难关卡平均只能拿 0 分，动作 B 拿了 5 分，反而可能是更值得学习的动作。

Advantage 关心的不是“得了多少分”，而是：

$$
A^\pi(s_t,a_t)
=
Q^\pi(s_t,a_t)-V^\pi(s_t)
$$

也就是，**选择这个动作以后，比在该状态下正常发挥好多少。**

### 2.2 数据来自旧策略

一次轨迹采样结束后，我们会固定生成这批数据的策略，记作：

$$
\pi_{\theta_{\mathrm{old}}}
$$

接下来进行若干个参数更新步骤时，当前策略 $\pi_\theta$ 会不断变化，但旧策略的概率必须保持不变。

为了衡量某个动作的概率发生了多大变化，PPO 引入重要性比率：

$$
\rho_t(\theta)
=
\frac{
\pi_\theta(a_t \mid s_t)
}{
\pi_{\theta_{\mathrm{old}}}(a_t \mid s_t)
}
$$

实际计算时使用对数概率，避免直接除小概率导致数值下溢：

$$
\rho_t(\theta)
=
\exp
\left(
\log \pi_\theta(a_t \mid s_t)
-
\log \pi_{\theta_{\mathrm{old}}}(a_t \mid s_t)
\right)
$$

它的含义很直观：

| $\rho_t$ | 含义 |
|---:|---|
| $1.0$ | 新旧策略对该动作的概率相同 |
| $1.2$ | 新策略选择该动作的概率提高了 20% |
| $0.7$ | 新策略选择该动作的概率降到了原来的 70% |

有了这个比率，一个自然的代理目标是：

$$
L^{\mathrm{PG}}(\theta)
=
\mathbb{E}_t
\left[
\rho_t(\theta)\hat A_t
\right]
$$

这里的 $\hat A_t$ 是 $A^{\pi_{\mathrm{old}}}(s_t,a_t)$ 的样本估计。在本轮采样轨迹的所有参数更新步骤中，它都保持固定，不会随着当前策略更新而重新计算。

刚开始更新时，$\rho_t \approx 1$，这个目标和普通策略梯度一致。但如果在同一批数据上优化很多轮，$\rho_t$ 会逐渐远离 1，模型就可能为了吃透一批旧数据而走得太远。

需要注意：这不是把 PPO 变成了可以任意复用历史经验的异策略（off-policy）算法。它仍然使用旧策略下采到的状态分布，只在有限范围内修正动作概率的变化。因此，PPO 依然是**同策略**算法，旧采样轨迹只能复用有限个训练轮次。

---

## 3. PPO 裁剪：不给过度更新继续加分

裁剪版 PPO 的核心目标是：

$$
L^{\mathrm{CLIP}}(\theta)
=
\mathbb{E}_t
\left[
\min
\left(
\rho_t(\theta)\hat A_t,\;
\operatorname{clip}
\left(
\rho_t(\theta),1-\epsilon,1+\epsilon
\right)
\hat A_t
\right)
\right]
$$

这是一个需要**最大化**的目标。代码里通常最小化它的相反数。

$\epsilon$ 控制“不再继续奖励”的区间，常见值是 0.1～0.2。若 $\epsilon=0.2$，裁剪区间就是 $[0.8,1.2]$。

这个公式里的 `min` 表示取较小值，看起来有些绕，拆成 Advantage 为正、Advantage 为负两种情况就清楚了。

### 3.1 当 $\hat A_t > 0$：这是一个好动作

如果动作比预期好，优化器希望提高它的概率，也就是让 $\rho_t$ 变大。

但是，当 $\rho_t$ 超过 $1+\epsilon$ 后，裁剪分支会成为更小、也更保守的目标。继续提高这个动作的概率，不再带来额外收益。

$$
\hat A_t > 0
\quad\Rightarrow\quad
L_t
=
\min(\rho_t,1+\epsilon)\hat A_t
$$

直觉上：

> 好动作可以多选一点，但没必要只凭这一批数据就把它捧上天。

### 3.2 当 $\hat A_t < 0$：这是一个坏动作

如果动作比预期差，优化器希望降低它的概率，也就是让 $\rho_t$ 变小。

但当 $\rho_t$ 低于 $1-\epsilon$ 后，继续降低概率也不再获得额外收益。

$$
\hat A_t < 0
\quad\Rightarrow\quad
L_t
=
\max(\rho_t,1-\epsilon)\hat A_t
$$

这里出现 `max`（取较大值），是因为乘数 $\hat A_t$ 为负，符号会翻转。

直觉上：

> 坏动作应该少选一点，但也别因为一次差评就把它彻底判死刑。

### 3.3 一个可以手算的例子

令 $\epsilon=0.2$。

| Advantage | $\rho_t$ | 未裁剪项 $\rho_t\hat A_t$ | PPO 采用的值 | 解释 |
|---:|---:|---:|---:|---|
| $+2$ | $1.0$ | $2.0$ | $2.0$ | 概率还没变化 |
| $+2$ | $1.2$ | $2.4$ | $2.4$ | 好动作被适度增强 |
| $+2$ | $1.4$ | $2.8$ | $2.4$ | 超过上界，不再奖励 |
| $-2$ | $1.0$ | $-2.0$ | $-2.0$ | 概率还没变化 |
| $-2$ | $0.8$ | $-1.6$ | $-1.6$ | 坏动作被适度抑制 |
| $-2$ | $0.6$ | $-1.2$ | $-1.6$ | 低于下界，不再奖励 |

还有一个常被忽略的细节：**如果概率朝错误方向变化，PPO 仍会继续惩罚。**

- 好动作的概率从 1.0 倍降到 0.6 倍，目标不会帮它裁剪成 0.8 倍；
- 坏动作的概率从 1.0 倍升到 1.4 倍，目标也不会帮它裁剪成 1.2 倍。

所以，不能简单地说“比率越界以后梯度就一定为零”。只有当更新方向已经正确、但幅度过大时，裁剪才会让继续前进失去收益。

### 3.4 裁剪不是硬约束

PPO 裁剪的是代理目标，不是：

- 模型参数；
- 梯度；
- 奖励；
- 新策略的真实动作概率。

神经网络的参数是共享的。更新一个状态上的动作概率，也可能间接改变其他状态上的分布。即使某个已采样动作的目标进入平坦区，其他样本的梯度仍可能把它推得更远。

因此，裁剪版 PPO 只能表达：

> 我不再为过度更新提供额外激励。

它不能保证：

$$
D_{\mathrm{KL}}
\left(
\pi_{\theta_{\mathrm{old}}}
\Vert
\pi_\theta
\right)
\leq \delta
$$

工程实现仍然会监控旧策略到当前策略的 KL 散度；如果 KL 散度增长过快，可以提前结束当前采样轨迹上的优化轮次。

---

## 4. Advantage 从哪里来：Critic 与 GAE

PPO 的裁剪目标告诉我们“怎么限制更新”，但还没有回答“每一步的 $\hat A_t$ 怎么算”。

最常见的 PPO 实现采用 Actor-Critic 架构：

- **Actor（策略模型）** $\pi_\theta(a_t\mid s_t)$：决定动作；
- **Critic（价值模型）** $V_\phi(s_t)$：预测从状态 $s_t$ 开始，未来大约还能获得多少回报。

Critic 可以是独立网络，也可以和 Actor 共享一部分参数。PPO 本身并不要求它必须是一个完全独立的模型。

### 4.1 时序差分误差

为避免同时用 $r_t$ 表示奖励和概率比，本文把环境奖励记作 $u_t$。

下面的 $V_{\phi_{\mathrm{old}}}$ 表示采样轨迹生成时，由 Critic 算出并缓存的价值估计，不是额外常驻的另一个模型。

单步时序差分误差为：

$$
\delta_t
=
u_t
+
\gamma b_t V_{\phi_{\mathrm{old}}}(s_{t+1})
-
V_{\phi_{\mathrm{old}}}(s_t)
$$

其中：

- $\gamma$：折扣因子；
- $b_t$：价值续接标记，决定是否借助下一状态的价值估计补全后续回报；
- 真正终止时 $b_t=0$；
- 普通中间状态，以及仅因时间上限或采样分段而停止时，通常有 $b_t=1$。

如果一个回合只是因为时间上限而被截断，环境并没有真正结束，通常仍应该用 $V(s_{t+1})$ 补全尚未观察到的后续回报。这里必须使用环境重置前最后一个真实观测对应的价值估计，而不是自动重置后新回合的初始状态。把“终止”和“截断”混为一谈，会系统性低估边界状态的回报。

### 4.2 GAE：在偏差与方差之间调节

GAE（Generalized Advantage Estimation，广义优势估计）把未来的时序差分误差做指数加权：

$$
\hat A_t
=
\delta_t
+
\gamma\lambda c_t\hat A_{t+1}
$$

这里的 $c_t$ 是 Advantage 递推标记，表示 GAE 能否继续递推到轨迹里的下一个位置：

- 普通中间状态时 $c_t=1$；
- 真正终止、时间截断或采样分段边界时 $c_t=0$。

为什么要把 $b_t$ 与 $c_t$ 分开？如果时间上限只是采样过程的外部限制，任务本身尚未终止，那么计算 $\delta_t$ 时仍应使用截断处最后观测到的下一状态价值，近似尚未采到的后续回报，因此 $b_t=1$。可是，采样缓冲区中的下一条记录可能已经是环境重置后新回合的开头，GAE 不能把下一回合的 Advantage 接到上一回合，因此 $c_t=0$。

对位置 $t=1,\ldots,T$，递推式也可以展开为：

$$
\hat A_t
=
\sum_{l=0}^{T-t}
(\gamma\lambda)^l
\left(
\prod_{j=0}^{l-1}c_{t+j}
\right)
\delta_{t+l}
$$

其中 $l=0$ 时的空乘积定义为 1。

$\lambda$ 控制偏差与方差：

- $\lambda=0$：接近一步时序差分估计，方差较低，但更依赖 Critic 的准确性；
- $\lambda\to1$：在完整回合、边界不额外使用下一状态价值时，更接近蒙特卡洛回报，通常方差更大；
- 实践中常取接近 1 的值，例如 0.95，但它不是固定答案。

Critic 的回归目标可以写成：

$$
\hat G_t
=
\hat A_t
+
V_{\phi_{\mathrm{old}}}(s_t)
$$

轨迹采样结束后，旧对数概率、旧价值估计、Advantage 和回报目标都应该固定下来，不随着后续参数更新步骤重新计算或反向传播。

许多实现还会对当前数据批次的 Advantage 做标准化：

$$
\tilde A_t
\leftarrow
\frac{\hat A_t-\mu_A}{\sigma_A+\varepsilon}
$$

正确的顺序是：先用**未标准化**的 GAE Advantage 构造 $\hat G_t$，再复制一份 Advantage，仅为 Policy loss 做标准化。标准化后的 Advantage 不应用来重新计算 Critic 的回归目标。

Advantage 标准化通常有助于优化稳定，但属于实现选择，不是 PPO 的定义。

---

## 5. 完整 PPO 不只有裁剪目标

一个常见的 PPO 总 loss 由三部分组成。

### 5.1 Policy loss

理论上最大化：

$$
L^{\mathrm{CLIP}}(\theta)
$$

代码里通常最小化：

$$
L_{\mathrm{policy}}
=
-L^{\mathrm{CLIP}}(\theta)
$$

### 5.2 Value loss

Critic 回归到固定的回报目标：

$$
L_V(\phi)
=
\frac{1}{2}
\mathbb{E}_t
\left[
\left(
V_\phi(s_t)-\hat G_t
\right)^2
\right]
$$

一些实现还会裁剪新旧价值估计的变化，再取裁剪与未裁剪 loss 中较大的一个。**价值裁剪是常见变体，不是 PPO 裁剪目标的核心定义。**

### 5.3 熵奖励

策略熵为：

$$
\mathcal H(\pi_\theta)
=
-
\mathbb{E}_{a\sim\pi_\theta}
\left[
\log\pi_\theta(a\mid s)
\right]
$$

熵奖励鼓励策略保留一定随机性，避免过早坍缩到单一动作。

如果统一写成“最小化 loss”，常见形式是：

$$
L_{\mathrm{total}}
=
-L^{\mathrm{CLIP}}
+
c_V L_V
-
c_H\mathcal H(\pi_\theta)
$$

其中 $c_V$、$c_H$ 分别控制 Value loss 和熵奖励的权重。

需要强调：熵控制的是策略自身的不确定性，它和“让策略靠近某个参考模型”不是一回事。

---

## 6. 一轮 PPO 训练到底发生了什么？

完整数据流可以写成：

```text
当前策略
  ↓ 固定为本轮旧策略
采集一批轨迹
  ↓
保存动作 / 奖励 / 终止标记 / 旧对数概率 / 旧价值估计
  ↓
反向计算 GAE Advantage 与回报目标
  ↓
把采样轨迹打乱并切成小批次
  ↓
在同一批采样轨迹上更新若干轮
  ├─ 重新计算当前策略的对数概率
  ├─ 用新旧对数概率之差得到概率比
  ├─ 计算裁剪后的 Policy loss
  ├─ 计算 Value loss 与熵
  └─ 若旧策略到当前策略的 KL 散度过大则提前停止
  ↓
丢弃旧采样轨迹，用更新后的策略重新采样
```

对应伪代码如下：

```python
# 初始化 Actor πθ 和 Critic Vφ

for iteration in range(num_iterations):
    # 1. 本轮轨迹采样使用的策略快照
    old_policy = snapshot(πθ)

    # 2. 同策略采样
    rollout = collect_trajectories(old_policy)
    rollout.old_logprobs = old_policy.log_prob(rollout.actions)
    rollout.old_values = Vφ(rollout.states)
    # 边界处使用环境重置前的真实下一观测
    rollout.next_values = Vφ(rollout.bootstrap_observations)

    # 3. 固定训练目标
    rollout.raw_advantages = compute_gae(
        rewards=rollout.rewards,
        values=rollout.old_values,
        next_values=rollout.next_values,
        terminated=rollout.terminated,
        truncated=rollout.truncated,
        segment_ends=rollout.segment_ends,
    )
    rollout.returns = rollout.raw_advantages + rollout.old_values
    rollout.advantages = normalize_for_policy(
        rollout.raw_advantages
    )

    # 4. 有限次数地复用这批采样轨迹
    for epoch in range(update_epochs):
        for minibatch in shuffle_and_split(rollout):
            new_logprobs, entropy = πθ.evaluate(
                minibatch.states,
                minibatch.actions,
            )
            new_values = Vφ(minibatch.states)

            ratio = exp(new_logprobs - minibatch.old_logprobs)
            policy_loss = ppo_clip_loss(ratio, minibatch.advantages)
            value_loss = mse(new_values, minibatch.returns)

            loss = policy_loss + c_v * value_loss - c_h * entropy
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(parameters)
            optimizer.step()

        if approximate_kl_old_to_current(
            rollout.old_logprobs,
            πθ,
            rollout,
        ) > target_kl:
            break
```

这里最重要的时间关系是：

```text
旧策略 / 旧对数概率 / 旧价值估计 / Advantage / 回报目标
在一批采样轨迹的所有更新轮次中都保持不变。
```

---

## 7. PyTorch 风格的核心 loss

下面是一个省略网络前向计算、数据采样和分布式细节的核心实现。它同时兼容普通轨迹和带填充位置的序列数据。

```python
import torch


def masked_mean(x, mask):
    mask = mask.to(x.dtype)
    return (x * mask).sum() / mask.sum().clamp_min(1.0)


def ppo_loss(
    new_logprobs,
    old_logprobs,
    advantages,
    new_values,
    returns,
    entropy,
    action_mask,
    clip_eps=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
):
    # 轨迹采样后缓存的量都应视为固定训练目标
    old_logprobs = old_logprobs.detach()
    advantages = advantages.detach()
    returns = returns.detach()

    log_ratio = new_logprobs - old_logprobs
    ratio = torch.exp(log_ratio)

    # 最大化两个代理目标中的较小值
    # 等价于最小化下面两个负 loss 中的较大值
    loss_unclipped = -ratio * advantages
    loss_clipped = (
        -torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        * advantages
    )
    policy_loss = masked_mean(
        torch.maximum(loss_unclipped, loss_clipped),
        action_mask,
    )

    value_loss = 0.5 * masked_mean(
        (new_values - returns).square(),
        action_mask,
    )
    entropy_mean = masked_mean(entropy, action_mask)

    total_loss = (
        policy_loss
        + value_coef * value_loss
        - entropy_coef * entropy_mean
    )

    with torch.no_grad():
        # 一个常用、近似非负的采样 KL 估计量
        approx_kl = masked_mean(
            (ratio - 1.0) - log_ratio,
            action_mask,
        )
        clip_fraction = masked_mean(
            ((ratio - 1.0).abs() > clip_eps).float(),
            action_mask,
        )

    metrics = {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy_mean,
        "approx_kl_old": approx_kl,
        "clip_fraction": clip_fraction,
    }
    return total_loss, metrics
```

真实实现还需要处理：

- GAE 的终止与截断语义；
- Advantage 标准化；
- 价值裁剪；
- 混合精度；
- 梯度累积；
- 多卡上考虑掩码的全局均值；
- 循环策略的序列边界；
- 采样端与训练器之间的策略滞后。

---

## 8. PPO 如何用于大语言模型与 RLHF？

RLHF 指人类反馈强化学习（Reinforcement Learning from Human Feedback）。在机器人任务中，PPO 每一步选择一个控制动作；在大语言模型中，每一步选择的是下一个词元（token）。

设提示词（prompt）为 $x$，已经生成的回答前缀为 $y_{<t}$：

$$
s_t=(x,y_{<t}),
\qquad
a_t=y_t
$$

$$
\pi_\theta(a_t\mid s_t)
=
\pi_\theta(y_t\mid x,y_{<t})
$$

对应关系如下：

| 经典强化学习 | LLM / RLHF |
|---|---|
| 初始状态 | 提示词 |
| 状态 $s_t$ | 提示词 + 已生成词元前缀 |
| 动作 $a_t$ | 下一个词元 |
| 一个回合 | 一整段回答 |
| 环境转移 | 把新词元拼到上下文末尾 |
| 采样停止 | 序列结束标记（EOS）等真正终止，或长度上限等截断 |
| 奖励 | 奖励模型、规则验证器或任务反馈 |

LLM 的状态转移几乎是确定性的，但动作空间是整个词表，轨迹可能很长，而且主要奖励通常要等完整回答生成后才出现。

长度上限究竟应视为“任务真的结束”，还是“尚未结束但被截断”，取决于训练目标。若是后者，GAE 在边界处仍应使用下一状态的价值估计，近似尚未生成部分的回报。

### 8.1 经典 PPO-RLHF 中的四个逻辑模型

| 模型 | 作用 | 通常是否更新 |
|---|---|---|
| **策略模型** | 生成回答，也是最终要训练的模型 | 更新 |
| **Critic** | 预测每个生成前缀未来采用塑形奖励后得到的累计回报 | 更新 |
| **奖励模型（Reward Model）** | 给完整的提示词—回答打分 | 冻结 |
| **参考模型（Reference Model）** | 提供监督微调（SFT）后的行为锚点，限制长期漂移 | 冻结 |

此外还有一个很容易与参考模型混淆的概念：**旧策略**。

- 旧策略：本轮轨迹采样时策略模型的快照，是 PPO 概率比的分母；
- 参考策略：通常是强化学习开始前冻结的监督微调模型，用于长期行为约束；
- 当前策略：正在进行参数更新的策略模型。

旧策略不一定需要常驻为“第五个完整模型”。工程上经常只缓存生成词元对应的旧对数概率 `old_logprobs`。

### 8.2 LLM 中通常按词元计算 PPO 概率比

对于生成的第 $t$ 个词元：

$$
\rho_t^{\mathrm{PPO}}
=
\exp
\left[
\log\pi_\theta(y_t\mid x,y_{<t})
-
\log\pi_{\mathrm{old}}(y_t\mid x,y_{<t})
\right]
$$

不要把它轻易替换成整条序列概率比：

$$
\rho_{\mathrm{sequence}}
=
\prod_{t=1}^{T}
\rho_t
$$

长序列上的概率连乘方差极大，也容易上溢或下溢。PPO-RLHF 通常把回答中的每个词元看作一步动作，只在有效词元上计算 loss。

这意味着所有统计都必须使用正确的回答掩码：

- 提示词中的词元不参与 Policy loss；
- 填充位置不参与 loss 和均值；
- 序列结束标记之后的位置不参与计算；
- Advantage 标准化也只能统计有效的回答词元。

### 8.3 一条回答的最终奖励如何变成逐词元 Advantage？

奖励模型往往只对完整回答输出一个标量：

$$
R_{\mathrm{RM}}=r_\psi(x,y_{1:T})
$$

常见做法是先加入逐词元的参考模型偏离项。对旧策略实际采到的词元，可以构造：

$$
k_t^{\mathrm{ref}}
=
\log\pi_{\mathrm{old}}(y_t\mid x,y_{<t})
-
\log\pi_{\mathrm{ref}}(y_t\mid x,y_{<t})
$$

$$
\tilde u_t
=
-\beta k_t^{\mathrm{ref}}
+
\mathbb{1}[t=T]R_{\mathrm{RM}}
$$

也就是：

- 每个词元都可能得到一个由参考模型对数概率比构造的 KL 塑形奖励项，它在期望上起惩罚作用；
- 奖励模型给出的最终分数只加到最后一个有效词元；
- 再由 Critic 和 GAE 把未来回报向前传播，得到各词元的 Advantage。

单个已采样词元的 $k_t^{\mathrm{ref}}$ 可能为负。它只是 KL 散度的蒙特卡洛采样项；对旧策略的动作分布取期望后，才对应非负的真实 KL 散度。

更具体地，在给定词元前缀 $h_t=(x,y_{<t})$ 时：

$$
\mathbb{E}_{y_t\sim\pi_{\mathrm{old}}(\cdot\mid h_t)}
\left[
k_t^{\mathrm{ref}}
\right]
=
D_{\mathrm{KL}}
\left(
\pi_{\mathrm{old}}(\cdot\mid h_t)
\Vert
\pi_{\mathrm{ref}}(\cdot\mid h_t)
\right)
$$

因此，上面的固定奖励塑形衡量的是**采样时的旧策略到参考模型**的偏离，不是经过若干次内部 PPO 更新后，当前策略到参考模型的精确 KL 散度。

如果目标是未折扣的整段回答奖励，PPO-RLHF 常取 $\gamma=1$。若使用 $\gamma<1$，奖励模型的最终得分向前传播时会随词元距离折扣，从而引入额外的长度相关权重；这应当是有意选择的目标变化。

有些实现不把相对参考模型的 KL 散度放进逐词元奖励，而是在更新策略模型时加入可微的 KL loss。两种位置的优化行为不完全相同，阅读代码时应先确认采用的是哪一种；如果两种方式同时存在，还要确认是否有意进行双重惩罚。

### 8.4 PPO 裁剪与参考模型 KL 是两道不同的护栏

| 机制 | 比较对象 | 时间尺度 | 作用 |
|---|---|---|---|
| **PPO 裁剪** | 当前策略与本轮旧策略 | 一轮 PPO 更新中的若干训练轮次 | 限制单轮更新过猛 |
| **参考模型 KL** | 采样策略（奖励塑形方案）或当前策略（可微 KL loss 方案）与固定的监督微调参考模型 | 整个强化学习训练过程 | 限制长期行为漂移 |
| **熵奖励** | 策略自身 | 每次更新 | 保持一定探索性 |

它们解决的是三个不同问题，不能统称为同一个“KL 约束”。

### 8.5 一轮 PPO-RLHF 数据流

```text
提示词批次
  ↓
旧策略生成回答，并缓存旧对数概率
  ↓
参考模型计算相同回答词元的对数概率
  ↓
奖励模型或验证器给完整回答打分
  ↓
Critic 计算每个生成前缀的旧价值估计
  ↓
组合逐词元 KL 惩罚与最终奖励
  ↓
计算 GAE Advantage 与回报目标
  ↓
进行若干轮 PPO 更新
  ├─ 更新 Critic
  └─ 用裁剪目标更新策略模型
  ↓
刷新采样策略，进入下一轮
```

一条训练样本常见字段包括。下面保留英文，是因为它们通常直接对应代码中的字段名：

```text
prompt_ids
response_ids
attention_mask
response_mask
old_logprobs
ref_logprobs
old_values
reward_model_score
token_rewards
advantages
returns
```

如果希望继续沿着工程代码阅读，可以参考：

- [[verl-ray-ppo-trainer|verl：读懂 ray_trainer.py 里的 PPO 训练主线]]

---

## 9. 训练时应该看哪些指标？

单看 Policy loss `policy_loss` 几乎无法判断策略是否真的变好了。更有用的是同时观察以下指标：

| 指标 | 在回答什么问题？ |
|---|---|
| 回合回报 / 奖励模型得分 | 策略拿到的原始奖励是否上升？ |
| 目标回报 | 扣除 KL 塑形项后，优化目标是否上升？ |
| 近似 KL `approx_kl_old` | 从本轮旧策略到当前策略的近似 KL 散度有多大？ |
| 参考模型 KL | 策略长期偏离监督微调参考模型多远？ |
| 裁剪比例 | 有多少有效动作的概率比已超出裁剪区间？ |
| 熵 | 策略是否过早变得确定？ |
| Value loss | Critic 的回归误差多大？ |
| 解释方差 | Critic 是否真的解释了回报变化？ |
| Advantage 均值 / 标准差 | Advantage 估计和归一化是否健康？ |
| 梯度范数 | 更新是否出现尖峰？ |

裁剪比例可以写成：

$$
\operatorname{clipfrac}
=
\Pr
\left(
\left|\rho_t-1\right|>\epsilon
\right)
$$

Critic 的解释方差常写成：

$$
\operatorname{EV}
=
1-
\frac{
\operatorname{Var}(\hat G_t-V_\phi(s_t))
}{
\operatorname{Var}(\hat G_t)
}
$$

一些典型信号：

- 旧策略到当前策略的 `approx_kl_old` 与裁剪比例突然升高：单轮策略更新可能过激；
- 解释方差长期接近 0 或为负：Critic 没有提供有效的基线值；
- 奖励模型得分上升，但参考模型 KL 和回答长度同时暴涨：可能存在奖励投机，也就是模型在钻奖励规则的漏洞，或者存在长度偏置；
- 序列结束标记出现率下降、长度截断率上升：模型可能在利用长度获取奖励；
- Advantage 方差极大：检查奖励尺度、Critic、终止掩码与归一化逻辑。

在 LLM 训练中，相对参考模型的 KL 散度最好同时记录：

- 每个有效词元的均值；
- 每条回答的总和。

否则，回答长度变化可能掩盖真实趋势。

---

## 10. PPO 与 GRPO 是什么关系？

[[GRPO]] 可以理解为沿用了 PPO 的概率比与裁剪思想，但换掉了 Advantage 的来源。

| 维度 | PPO | GRPO |
|---|---|---|
| 更新护栏 | 概率比 + 裁剪 | 概率比 + 裁剪 |
| Advantage 来源 | 通常由 Critic + GAE 得到 | 同一提示词下的组内奖励相对值 |
| Critic | 通常需要 | 通常不需要 |
| 采样组织 | 普通同策略轨迹采样 | 每个提示词采样一组回答 |
| 信用分配 | 可得到词元级或步骤级 Advantage | 常把结果级 Advantage 分配给回答中的各个词元 |
| 主要代价 | 训练与维护 Critic | 每个提示词需要更多采样 |

所以：

> GRPO 不是把 PPO 全部推翻，而是保留“别走太远”的更新方式，用组内比较替代了 PPO 中常见的 Critic 估计。

理解 PPO 的概率比、裁剪、旧策略和参考策略后，再读 GRPO 会轻松很多。

---

## 11. 常见误解

### 误解一：裁剪会保证新策略概率落在 $[0.8,1.2]$

不会。它只裁剪代理目标中的概率比收益，不是硬概率约束。

### 误解二：只写 $\operatorname{clip}(\rho_t)\hat A_t$ 就够了

不够。外层取较小值的 `min` 不能省略，否则概率朝错误方向变化时也可能被裁掉惩罚，改变原目标的含义。

### 误解三：PPO 有重要性比率，所以是异策略算法

不是。PPO 仍依赖最新策略附近的数据，只能有限次复用当前采样轨迹，不能像 DQN 或 SAC 那样长期从经验回放缓冲区取任意旧数据。

### 误解四：Critic 就是奖励模型

不是。奖励模型评价一条样本得到了什么奖励；Critic 预测从某个状态或词元前缀出发，未来期望还能得到多少经过奖励塑形后的累计回报。

### 误解五：旧策略就是参考策略

不是。旧策略是本轮轨迹采样使用的短期快照，参考策略是 RLHF 中长期冻结的行为锚点。

### 误解六：Policy loss 下降了，策略一定变好了

不一定。Policy loss 只衡量当前采样轨迹上的代理目标，而且每轮数据分布都会变化。真正的表现要看累计回报、验证指标、KL 散度、熵和 Critic 质量。

### 误解七：时间上限和真正终止都应该把下一状态价值设为 0

不一定。真正终止时，通常不再用下一状态价值补算后续回报；单纯因时间上限截断时，通常仍应使用下一状态的价值估计。

---

## 12. 参考资料

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — Schulman et al., 2017
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) — Schulman et al., 2015
- [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) — Schulman et al., 2015
- [OpenAI Spinning Up: Proximal Policy Optimization](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — Ouyang et al., 2022

---

## 13. 一句话总结

> **PPO 先用旧策略按同策略方式采集轨迹，用 Critic 和 GAE 判断每一步动作比预期好还是差，再依据新旧策略的概率比更新策略；裁剪的作用，是让正确方向上的过度更新不再获得额外收益。它是一条“安全带”，不是一堵绝对不会越过的墙。**
