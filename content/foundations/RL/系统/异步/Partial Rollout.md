---
title: Partial Rollout：异步 LLM RL 中的中断、续写与长尾调度
description: 拆解 partial rollout 的三种语义：权重同步中断、长尾回收与 Agent 分段，分析 token provenance、KV cache、环境状态和 off-policy 校正。
date: 2026-09-20
tags:
  - RL
  - Agentic-RL
  - RL-Systems
---

# Partial Rollout：异步 LLM RL 中的中断、续写与长尾调度

在同步 RL 里，一条 rollout 通常要从 prompt 一直生成到 EOS、超出最大长度或环境给出终止信号，才能交给 reward 和 trainer。这个假设在短回答上没有问题；到了长 CoT、工具调用和 coding agent，它会把一次很慢的轨迹变成整批任务的同步屏障。

**Partial rollout 的核心是：在轨迹尚未完成时保留或持久化足够的状态，让系统先处理别的工作，之后从这个状态继续生成。** 状态可以暂存在 inference engine 的内存中，也可以写入 continuation buffer；它不是一段可读文本，而是可以被训练和推理共同解释的 token 前缀、旧策略概率、模型版本、loss mask，以及必要时的环境状态。

这个名字在资料中有三种相近但不相同的用法：

| 用法             | 触发器                      | 主要目标                                | 典型代表                                                                                                                                                            |
| ---------------- | --------------------------- | --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **权重同步型**   | trainer 发布新权重          | 避免等待在途请求，缩短同步造成的 bubble | [verl fully async](https://github.com/verl-project/verl/blob/main/docs/advance/fully_async.md)、[SkyRL](https://docs.skyrl.ai/docs/tutorials/fully_async)、OpenRLHF |
| **长尾调度型**   | 本轮已收齐目标样本          | 截停长尾请求，回收未完成前缀            | [Kimi k1.5](https://arxiv.org/html/2501.12599)、[APRIL](https://arxiv.org/html/2509.18521)、[CoPRIS](https://arxiv.org/html/2511.05589)                             |
| **Agent 分段型** | token 或 tool-call 预算用尽 | 把超长、多工具轨迹变成可调度片段        | [AgentMath](https://arxiv.org/html/2512.20745)                                                                                                                      |

三者都可能让一条最终轨迹跨越多个 iteration 或 policy version，但优化对象不同：第一种减少 weight sync 的等待，第二种减少 batch 长尾，第三种限制单个 Agent 请求占用调度器的时间。它们可以叠加，也不能把一个实现中的“partial”直接当成另一个实现的语义。

本文资料查阅于 **2026-09-20**。框架接口变化很快，文中的能力表和参数名以链接中的文档或源码为准；论文中的吞吐数字是各自实验设置下的结果，不是统一硬件 benchmark。

## 一、同步 rollout 为什么会被一条长轨迹拖住

设一个 batch 中有 $B$ 条请求，第 $i$ 条请求的生成和环境交互耗时为 $T_i$，训练耗时为 $T_{\mathrm{train}}$。同步循环的单步时间近似为：

$$
T_{\mathrm{sync}} \approx \max_{1\le i\le B} T_i + T_{\mathrm{train}} + T_{\mathrm{reward}}.
$$

大多数请求已经结束时，推理 GPU 仍要为最后几条长请求保留 KV cache；trainer 也只能等待整批数据。工具调用会进一步拉长尾部：模型生成一次工具调用后，GPU 可能在等待代码沙箱、终端或远程服务，而下一条可以继续 decode 的请求没有被及时调度。

异步架构把生成与训练放入两个并行循环，中间用 buffer 连接：

```text
rollout workers ──→ completed / partial buffer ──→ trainer workers
       ↑                                              │
       └────────────── weight update ──────────────────┘
```

这消除了大部分阶段间空闲，却引入两个新的状态：已经完成但可能过期的 **stale rollout**，以及还没有完成、仍占用 KV 或环境资源的 **partial rollout**。前者是数据年龄问题，后者是请求生命周期问题。二者经常同时出现，但处理位置不同：stale 在样本进入 loss 前处理，partial 在 rollout scheduler 和 inference engine 中处理。

## 二、Partial rollout 究竟保存什么

可以把一条可恢复的请求抽象成：

```text
TrajectoryState = (
  task_id,
  token_prefix,
  segment_logprobs,
  segment_model_versions,
  loss_mask,
  environment_state,
  continuation_metadata
)
```

其中：

- `token_prefix` 是模型已经消费或生成的 token ID，不能只保存 decode 后的字符串；
- `segment_logprobs` 是这些 token 在生成时的行为概率，后续 PPO、GRPO 或重要性采样需要它；
- `segment_model_versions` 记录每个片段由哪个 policy version 生成。一条续写轨迹可能是 $v_{17}\rightarrow v_{18}\rightarrow v_{19}$；
- `loss_mask` 区分模型动作、工具观察、系统拼接和补齐边界。通常只对 policy 生成的 token 做 loss；
- `environment_state` 让工具调用、进程、文件系统和 session 可以继续。纯 token prefix 不能自动恢复外部副作用；
- `continuation_metadata` 包括任务组 ID、请求 ID、停止原因、剩余预算、KV-cache 版本和重试次数等。

因此，partial rollout 不是“把输出截短后当成一条训练样本”。在尚未拿到最终 reward 前，未完成 prefix 一般不能作为一个完整的 GRPO response 参加组内归一化，也不能因为没有完成就写成零分。它应留在 pending pool，直到完成、超时、不可恢复或被明确丢弃。

这也解释了为什么 [TITO](../轨迹管理/什么是TITO.md) 是 partial rollout 的基础约束：下一段生成必须接在真实 token 历史后面，不能把前缀 decode 成文本、解析成 JSON，再用 chat template 重新 tokenize。否则续写 token 的条件上下文已经变了，保存的 logprob 也不再对应这条轨迹。

## 三、三条实现路线

### 1. 权重同步到来：abort、保存并续写

在 fully async RL 中，rollout 可能正在生成，trainer 已经完成一次 optimizer step。若等所有在途请求完成再换权重，更新频率会被最长请求限制；若直接把新参数写进正在 decode 的模型，又可能在一次 forward 内看到新旧混合参数。

最常见的流程是：

```text
正在用 W_k 生成
      │
      │ trainer 发布 W_(k+1)
      ▼
abort / pause，保存 prefix、logprob、版本与环境句柄
      │
      ▼
同步并安装 W_(k+1)，处理 KV cache
      │
      ▼
用保存的 prefix 重新 prefill，继续生成 suffix
```

[verl V1 文档](https://verl.readthedocs.io/en/latest/advance/v1_async_trainer.html)把这四步写得很明确：中断前的 token 和 logprob 保留，未完成轨迹经负载均衡器重试，新请求使用更新后的模型，保留前缀的 KV cache 重新构建。当前 fully async 文档还把 partial sample 的数量、轨迹跨越的版本跨度等作为监控指标。

vLLM 的 [Async RL pause/resume API](https://docs.vllm.ai/en/latest/training/async_rl/)展示了同一个设计空间：

| pause 模式 | 在途请求                  | 是否产生 partial                                       |
| ---------- | ------------------------- | ------------------------------------------------------ |
| `abort`    | 立即终止并返回已生成结果  | 是                                                     |
| `wait`     | 等请求自然完成后再暂停    | 否，代价是长尾等待                                     |
| `keep`     | 冻结请求，`resume` 后继续 | API 不返回 partial；上层可把冻结状态序列化为可恢复对象 |

`keep` 不是自动的正确性保证。若保留旧 KV cache，新权重继续算出的 token 可能依赖旧权重产生的 hidden state；若 `clear_cache=True`，则要对保存的 prefix 重新 prefill。后者增加计算，却更容易明确“新 suffix 使用哪一套权重和 cache”。[vLLM 文档](https://docs.vllm.ai/en/latest/training/async_rl/)明确区分了这两种语义。

还有一种更细的实现是 in-flight 更新：在 token decode 循环之间接收新权重，sequence 本身不被 abort。[PipelineRL](https://arxiv.org/abs/2509.19128)采用了这种思路；它减少了重算，但同一条 sequence 会跨 policy version。这里的更新粒度取决于引擎如何安装一次 optimizer step 的权重，不能直接等同于“每个 token 都原子换权重”。再弱一层的 per-parameter streaming 甚至可能在相邻参数安装期间形成混合模型，不能与完整 step 的权重切换混为一谈。

### 2. 本轮样本已收齐：超额请求、截停和回收

第二条路线不由权重同步触发，而由“本轮已经收齐足够多的样本”触发。令训练步需要 $N$ 条完成样本，调度器先发起 $N'>N$ 个请求：

```text
发起 N' 个请求
  ├─ 已完成 N 条 → 送 reward / trainer
  └─ 仍在生成   → abort，保存 prefix，放入 continuation buffer

下一步：优先恢复 continuation buffer，再补发新任务
```

[Kimi k1.5](https://arxiv.org/html/2501.12599)把它用于长 CoT RL：固定每轮输出 token budget，超限轨迹保存到 replay buffer，下一轮继续；早期 segment 可以复用，当前轮新 segment 再做 on-policy 计算。Kimi K2 的技术报告也明确说，长尾未完成任务会暂停并在下一次 RL iteration 恢复。

[APRIL](https://arxiv.org/html/2509.18521)把这个想法系统化为 Active Partial Rollouts：over-provision、达到目标数后 early termination、buffer continuation、下一轮优先恢复。它仍然可以运行在同步的 rollout→train 外框里，所以不必引入一个完整的异步 replay pipeline。

APRIL 的 v3 论文在 Qwen3-4B/8B、GRPO/DAPO 等设置下报告平均 **22.5%** rollout throughput 提升，表中单项最高 **49.5%**；平均最终准确率提升约 **2.1 个百分点（按论文口径）**。这些数字对应论文给出的单节点实验，不能直接泛化成所有模型。论文还观察到，后续 batch 中约 40% 的 token 来自之前未完成的轨迹，最多观察到一条轨迹跨约 5 个 policy version，因此它已经是轻度 off-policy 方案，而不是免费的调度优化。

[CoPRIS](https://arxiv.org/html/2511.05589)进一步固定并发数，避免“超额请求过多”带来的 KV 重算和 off-policy 堆积，同时把不同阶段缓存的 logprob 沿 token 维拼接，再做 Cross-stage Importance Sampling。论文在 veRL 上报告 1.58–1.94× 的端到端加速；这是论文自己的模型、数据和并发设置，适合用来理解设计取舍，不应当视为通用收益承诺。

### 3. Agent 轨迹按预算切段

第三条路线把完整 Agent 轨迹主动写成多个 segment：

$$
\tau = \tau^{(1)} \oplus \tau^{(2)} \oplus \cdots \oplus \tau^{(K)}.
$$

每一段可以同时受 `max_new_tokens`、tool-call 数、环境等待时间或全局任务预算限制。段结束后：

- 生成 EOS，轨迹完成并计算 reward；
- 达到 segment budget，保存 prefix 和环境状态，下轮继续；
- 达到全局长度或 tool-call 上限，按任务定义终止或判为失败；
- 工具调用进入等待时，释放 inference engine 给其他 ready request。

[AgentMath](https://arxiv.org/html/2512.20745)把超长数学 Agent 轨迹按 token 和 tool-call 预算切分，并使用 prefix-aware load balancing：prefix 越长，prefill 代价越高，调度权重就越大；同一 session 尽量回到持有其 KV cache 的 engine。论文将这种 Agentic Partial Rollout 与 request-level asynchronous scheduling、工具沙箱服务结合，报告端到端 4–5× 的系统收益；其中 partial rollout 本身的贡献不能脱离其他两项优化单独解读。

分段型 partial 和权重同步型 partial 可以叠加。例如一条 trajectory 正在生成第 3 段时遇到新权重，它既可能因为 `weight_update` 被 abort，又可能因为到达 segment token budget 被保存。实现时要把两个原因写入 metadata，否则无法解释一段 prefix 为什么重算、为什么被 mask 或为什么跨越了多个版本。

## 四、partial、stale 和 terminal 不是一回事

三种状态建议在协议层分开：

| 状态              | 是否有最终 reward    | 是否应进入当前训练 batch      | 后续动作                  |
| ----------------- | -------------------- | ----------------------------- | ------------------------- |
| `complete`        | 有                   | 可以，按 token mask 训练      | 结束                      |
| `partial`         | 通常没有             | 通常不作为完整 response 训练  | 保存并 resume，或明确丢弃 |
| `stale_complete`  | 有                   | 经过版本阈值、IS 或 mask 决定 | 训练、等待或丢弃          |
| `terminal_failed` | 有失败原因或超时结果 | 依算法决定是否作为负样本      | 结束，不自动 resume       |

“未完成”不等于“失败”。工具超时后是否重试、环境是否可快照、最大 token 是否是硬终止，都应由任务协议决定。对有状态环境，保存 token prefix 不足以恢复：一个 shell 进程、容器文件系统、数据库事务或浏览器 session 可能已经改变。若环境不能 checkpoint 或重放，就必须把该请求标为不可恢复，而不是假装它可以从文本继续。

同样，`partial` 也不等于 `stale`。一条 partial 轨迹在第 $k$ 步开始、在第 $k+1$ 步完成，它的 prefix 可能来自旧版本，suffix 来自新版本；一条 stale_complete 则可能在一次请求内完整生成，只是 trainer 已经前进了若干版本。前者需要 segment 级 provenance 和恢复协议，后者主要需要 admission control 或 loss correction。

## 五、真正的训练问题：一条轨迹可能没有单一行为策略

在严格 on-policy 训练中，token $x_t$ 是行为策略 $\mu$ 在上下文 $x_{<t}$ 下采样的：

$$
\mu_t = \mu(x_t\mid x_{<t}).
$$

普通 PPO 的 token ratio 写作：

$$
\rho_t = \frac{\pi_\theta(x_t\mid x_{<t})}{\pi_{\mathrm{old}}(x_t\mid x_{<t})}.
$$

partial rollout 让 `old` 可能是一个分段函数：

$$
\mu_t = \pi_{v_j}(x_t\mid x_{<t}),
\quad t\in\text{segment }j.
$$

所以不能只给整条 trajectory 写一个结束时的 `model_version`，再假定所有 token 都来自这个版本。至少应保存：

```text
token_ids                  # 完整 token 序列
old_logprobs               # 每个 action token 的行为 logprob
segment_start / end        # 版本切换边界
model_version              # 每段或每 token 的行为版本
loss_mask                  # 哪些 token 可训练
group_id / reward          # GRPO 组关系与最终奖励
```

如果只保存字符串，或者只保存最终版本号，训练器就无法区分：

```text
old prefix (v17) | new suffix (v18)
```

与：

```text
complete response generated entirely by v18
```

这会污染 PPO ratio、KL、entropy 和 reward attribution。[OpenRLHF 的 async + partial 文档](https://openrlhf.readthedocs.io/en/latest/async_training.html)因此明确提醒：一条 sample 可能含有旧、新权重生成的 token，需要用少量 on-policy 性换吞吐；建议先验证同步训练的收敛，再调异步和 partial。

[DORA 的分析](https://arxiv.org/html/2604.26256)把这个问题写成 trajectory-level mixing 与 step/segment-level mixing 的区别：前者可以用整条轨迹的 behavior ratio，后者的行为策略沿 token 边界变化；若仍用单一 policy 的 advantage 解释整条混合轨迹，会产生 advantage-substitution bias。工程上的 `model_version`、`old_logprobs` 和 segment mask，正是为了让这种校正有可计算的边界。

### 前缀 token 和新 suffix 的 loss mask

续写时有两种常见的训练口径：

1. **整条 response 都训练**：保留旧 prefix 的 old logprob，并对每个 token 做 token-level importance correction；实现完整，但校正方差更大。
2. **只训练当前轮 suffix**：旧 prefix 只作为条件上下文，`loss_mask=0`；更接近当前 policy 的 on-policy 更新，但会丢掉前缀的梯度信号。

Slime 暴露了 `--mask-offpolicy-in-partial-rollout` 这一类选择；[verl fully async 文档](https://github.com/verl-project/verl/blob/main/docs/advance/fully_async.md)则把 rollout logprob、partial span 和 stale threshold 放在同一条数据链上。两种口径都可以成立，关键是 metadata、mask 和概率定义必须一致，不能把旧 prefix 当新模型刚采样的 action，却又不做校正。

### GRPO group 不能被调度顺序打散

GRPO 的 advantage 通常在同一个 prompt 的 $G$ 条 response 之间计算：

$$
A_i = \frac{R_i-\operatorname{mean}(R_{1:G})}
{\operatorname{std}(R_{1:G})+\varepsilon}.
$$

调度器可以按单条请求释放槽位，但训练器仍需保留 `group_id`。一条 response 被 partial 中断时，已完成的同组 response 可以暂存，但不能因为它们先完成就和别的 prompt 混成一个 group；未完成 response 最终回收后，才有完整的组内 reward 语义。

## 六、KV cache：省掉重算还是保住语义

权重更新后的续写有三个典型选择：

| 方案               | 做法                                                 | 代价 / 风险                                  |
| ------------------ | ---------------------------------------------------- | -------------------------------------------- |
| **re-prefill**     | 用新权重对保存的 token prefix 重新计算 KV，再 decode | 计算量随 prefix 变长；语义最容易解释         |
| **保留旧 KV**      | 直接用新权重接着旧 cache decode                      | 新旧 hidden state 混合，行为分布难以精确描述 |
| **drain 后再同步** | 不中断，在途请求完成后才换权重                       | 不浪费 token，但等待最长请求                 |

如果每次中断都对长度 $\ell$ 的 prefix 重新 prefill，标准 attention 的一次 prefill 近似有 $O(\ell^2)$ 的注意力代价。若一条总长度为 $L$ 的轨迹每 $K$ 个 token 中断一次，粗略累加为：

$$
\sum_{j=1}^{L/K}(jK)^2 \approx \frac{L^3}{3K}.
$$

这是展示趋势的上界式，不是端到端 wall-clock 定律；实际常数会受 paged KV、chunked prefill、batch 合并、prefix cache 命中和模型结构影响。它说明了为什么 partial rollout 不能只看“少等了多少同步时间”，还要看恢复带来的 prefill 和 cache 管理成本。

工程上常见的折中是：权重更新频率不宜高到每几个 token 都触发；优先在 tool boundary、segment boundary 或较大的 token budget 上切断；长 prefix 需要 sticky routing，把 continuation 送回持有缓存的 engine；对过于陈旧或重算代价过高的请求直接丢弃，而不是无限续写。

## 七、一个可审计的 scheduler 状态机

建议把请求生命周期写成显式状态机，而不是在 HTTP 重试逻辑里隐式完成：

```text
NEW
  │ dispatch
  ▼
RUNNING(v_k)
  ├─ EOS / terminal reward ───────────────→ COMPLETE
  ├─ token/tool budget reached ───────────→ PARTIAL_BUFFER
  ├─ weight update / abort ───────────────→ PARTIAL_BUFFER
  ├─ environment timeout / unrecoverable ─→ TERMINAL_FAILED
  └─ stale threshold exceeded ────────────→ DROPPED or WAITING

PARTIAL_BUFFER
  ├─ environment restored + version allowed → PREFILL(prefix, v_new)
  ├─ version too old                       → DROPPED
  └─ retry limit / state lost              → TERMINAL_FAILED

PREFILL(prefix, v_new) → RUNNING(v_new)
```

每次状态转移都应记录事件时间、policy version、prefix token 数、累计生成 token、KV 行为和原因。这样才能回答“吞吐变快后，训练到底吃了什么数据”：是减少了长尾等待，还是把更多旧 prefix 推进了训练；是缓存复用带来的收益，还是把 prefill 计算转移到了别的阶段。

伪代码可以写成：

```python
while not stop:
    req = choose_from(partial_buffer, new_tasks)
    state = load_state(req)

    segment = generate(
        prefix_ids=state.token_ids,
        max_new_tokens=segment_budget,
        max_tool_calls=tool_budget,
    )
    state.append(
        token_ids=segment.token_ids,
        old_logprobs=segment.logprobs,
        model_version=segment.model_version,
        loss_mask=segment.loss_mask,
    )

    if segment.terminal:
        completed_queue.put(finalize(state))
    elif segment.interruptible:
        partial_buffer.put(checkpoint(state))
    else:
        failed_queue.put(mark_unrecoverable(state))
```

真实实现还要处理并发的 weight update、工具调用的幂等性、重复提交和 checkpoint crash；上面的代码只表达数据契约，不代表可以直接替换框架实现。

## 八、主流框架的选择差异

以下表格综合 [Hugging Face 对 16 个异步 RL 库的比较](https://huggingface.co/blog/async-rl-training-landscape) 和各项目当前文档。它的用途是定位设计空间，不是替代具体 commit 的代码审计。

| 框架 / 方案                 | partial 触发与恢复                                                            | 版本 / stale 处理                                                 | 主要代价                                        |
| --------------------------- | ----------------------------------------------------------------------------- | ----------------------------------------------------------------- | ----------------------------------------------- |
| **verl V1 / fully async**   | abort 后保存 token、logprob，更新后从 prefix 重试；支持 multi-turn            | `max_off_policy_threshold`，可 drop 或 wait；记录 trajectory span | 重建 KV 和状态机复杂，轨迹可跨多个版本          |
| **SkyRL**                   | pause 时 abort in-flight `/chat/completions`，保存 partial，resume 时重新送入 | bounded staleness，容量门控                                       | 生成器要处理 abort/retry，prefix prefill 成本高 |
| **OpenRLHF**                | vLLM pause/resume 与 async queue 结合；一条 sample 可能混合新旧权重           | queue size 控制 lag，文档建议先从同步基线验证                     | 对 on-policy 的放松需要额外校正和监控           |
| **Slime**                   | dynamic sampling 中未完成 sample 回收进 data buffer，下一轮继续               | 可按 staleness 排序；可 mask partial 中旧 token                   | 轨迹拼接、版本边界和 loss mask 需要一起维护     |
| **AReaL**                   | 文档/综述中常见为 pause 后继续；具体版本可能 abort 或 drain，需查源码         | bounded queue 与 decoupled loss 等机制                            | 不能只看综述标签，更新语义随版本变化            |
| **APRIL / CoPRIS**          | 超额请求，收齐目标后截停并回收；下一轮优先续写                                | APRIL 依赖轻度 off-policy；CoPRIS 增加并发控制和 IS               | 长尾收益与旧 prefix 比例、KV 重算互相制约       |
| **PipelineRL**              | in-flight / step-level 更新，sequence 隐式跨版本继续                          | 记录行为版本并限制 lag                                            | 不丢 token，但整条 sequence 不再单版本          |
| **AgentMath**               | token/tool budget 切段，未完成轨迹从 unfinished pool 继续                     | 需要 segment provenance 和 prefix-aware routing                   | 环境、KV、工具状态都要可恢复                    |
| **batch boundary baseline** | 等所有请求完成后再同步，不支持 partial                                        | 几乎没有跨版本问题                                                | 长尾造成最大 bubble，吞吐最低但最易验证         |

综述中的 interrupt 表曾把 partial 归纳为“abort + prefix”“explicit save/resume”“soft pause”等档位；这些分类有助于比较，但不能忽略具体实现。比如同一个项目可能同时提供 drain 和 in-flight 两种配置，或者在 rollout server 里 abort、在上层 scheduler 里丢弃结果。真正需要核对的是：谁保存 prefix、谁拥有 old logprob、KV 何时失效、trainer 看见的 version span 是什么。

## 九、应该监控哪些量

只看 tokens/s 不足以判断 partial rollout 是否真的有效。建议至少记录：

| 指标                                     | 它回答的问题                         |
| ---------------------------------------- | ------------------------------------ |
| `partial_rate`、`partial_count`          | 有多少请求没有一次完成               |
| `partial_span` / `max_partial_span`      | 一条轨迹跨了多少 policy version      |
| `prefix_tokens`、`resume_prefill_tokens` | 续写前重算了多少上下文               |
| `buffer_age`、`oldest_model_version`     | pending 或 completed 样本有多旧      |
| `stale_drop_rate`、`wait_rate`           | 多少数据被丢弃或阻塞等待             |
| `offpolicy_token_ratio`                  | 训练 batch 中有多少 token 来自旧版本 |
| `group_completion_delay`                 | GRPO group 是否被少数长轨迹拖住      |
| `env_restore_failure`、`retry_count`     | partial 在真实环境里是否可恢复       |
| `KV_reuse_hit_rate`、`KV_clear_count`    | cache 复用是否抵消了 prefill 代价    |
| `reward_by_age`、`reward_by_span`        | 旧数据是否产生了系统性质量偏差       |

实验对比应至少同时报告：端到端 step time、有效训练样本吞吐、生成 token 吞吐、GPU idle ratio、样本丢弃率、最终 reward/accuracy，以及不同 prefix 长度下的恢复成本。只报告 rollout throughput，可能把训练等待、reward 服务和 IS 重算的代价藏起来。

## 十、什么时候适合使用

partial rollout 最适合下面几类场景：

- rollout 长度有明显长尾，少数请求经常接近最大 token 或 tool-call 上限；
- 训练和推理分离，权重同步等待已经成为可测量的 step-time 占比；
- inference engine 能返回真实 token IDs 和 old logprobs，并支持可靠的 abort/resume；
- Agent 环境可以快照、重放或以幂等方式恢复；
- 算法能够接受轻度 off-policy，或者已经实现 token-level IS、mask、版本阈值等控制。

以下情况应先保持同步或只使用 drain：

- 任务短且长度分布窄，partial 的 buffer、prefill 和调试成本大于节省的等待；
- 环境有不可逆副作用，无法恢复进程、文件系统或外部事务；
- 训练严格依赖单一行为策略，且没有 old logprob 或 version span；
- 需要可复现到 token 级，但恢复时会改变随机数、采样参数或 chat template；
- 目前还没有同步 baseline，无法判断异步后的性能变化来自算法还是系统。

一个稳妥的落地顺序是：先用同步模式验证 reward、loss mask、TITO 和 group 统计；再启用不跨权重版本的长尾切段；然后引入 weight-sync abort/resume；最后才扩大 buffer、并发和 stale threshold。每一步都保留可回退的 `partial=False` 路径。

## 十一、关键结论

Partial rollout 解决的不是一个单独的“生成太慢”问题，而是把一条长轨迹从阻塞式函数调用改成可暂停、可恢复、可审计的状态机。它的收益来自更细的调度边界，代价来自更复杂的数据来源和状态一致性。

最容易被忽略的三点是：

1. **prefix 是 token 数据，不是 transcript 文本。** 续写和训练都必须保留真实 token 与 logprob；
2. **partial 和 stale 要分开记。** 一个描述请求是否完成，一个描述样本相对 trainer 是否过期；
3. **环境状态和 KV cache 都属于轨迹状态。** 只保存模型输出而不保存可恢复的工具/环境上下文，不能构成真正的 partial rollout。

如果只需要选择一个起点，短且有长尾的数学 RL 可以先尝试 APRIL/CoPRIS 这一类 batch-level 回收；长时间、多轮、频繁同步的 Agent RL 则应从 verl/SkyRL 的 prefix resume 和 token-level provenance 开始。两条路线都需要用同步 baseline 和版本跨度指标验证收益，不能只看生成端的瞬时吞吐。

## 参考资料

- [Keep the Tokens Flowing: Lessons from 16 Open-Source RL Libraries](https://huggingface.co/blog/async-rl-training-landscape) —— 异步 RL 库的 buffer、权重同步、staleness 和 partial rollout 对比。
- [V1 Async Trainer / Partial Rollout and Staleness](https://verl.readthedocs.io/en/latest/advance/v1_async_trainer.html) —— verl V1 的 abort、prefix resume、KV 重建和版本跨度指标。
- [Fully Async Training](https://github.com/verl-project/verl/blob/main/docs/advance/fully_async.md) —— partial rollout 流水线、staleness threshold 与监控项。
- [Async Reinforcement Learning](https://docs.vllm.ai/en/latest/training/async_rl/) —— vLLM `pause_generation`、`resume_generation` 和 KV cache 语义。
- [Async Training & Partial Rollout](https://openrlhf.readthedocs.io/en/latest/async_training.html) —— OpenRLHF 对 queue、权重混合和 on-policy 取舍的说明。
- [PipelineRL](https://arxiv.org/abs/2509.19128) —— 每个 optimizer step 的 in-flight 权重更新与 sequence 跨版本续写。
- [DORA: A Scalable Asynchronous Reinforcement Learning System for Language Model Training](https://arxiv.org/html/2604.26256) —— 区分 trajectory-level 与 step/segment-level mixing，说明混合行为策略下的 advantage 偏差。
- [Kimi k1.5: Scaling Reinforcement Learning with LLMs](https://arxiv.org/html/2501.12599) —— 固定 token budget、replay buffer 和跨 iteration 的长 CoT segment。
- [Kimi K2: Open Agentic Intelligence](https://arxiv.org/html/2507.20534) —— 长尾 Agent 轨迹暂停并在下一轮恢复的工程描述。
- [APRIL: Active Partial Rollouts in Reinforcement Learning to tame long-tail generation](https://arxiv.org/html/2509.18521) —— over-provision、early termination、continuation buffer 和实验分析。
- [CoPRIS: Efficient and Stable Reinforcement Learning via Concurrency-Controlled Partial Rollout with Importance Sampling](https://arxiv.org/html/2511.05589) —— 并发控制和 Cross-stage Importance Sampling。
- [AgentMath: Empowering Mathematical Reasoning for Large Language Models via Tool-Augmented Agent](https://arxiv.org/html/2512.20745) —— request-level async、Agentic Partial Rollout 和 prefix-aware routing。
- [[什么是TITO]] —— 多轮 Agent 轨迹的 token prefix 不变量。
- [[Slime轨迹管理]] —— Message Tree、TurnRecord 和 token-level trajectory 拼接。
- [[主要RL框架重同步方案]] —— 权重传输、interrupt model 与 partial rollout 的框架快照。
