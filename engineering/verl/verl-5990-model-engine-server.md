# verl #5990: 用 Model Engine Server 移出 old_log_prob 关键路径

> PR: [verl-project/verl#5990](https://github.com/verl-project/verl/pull/5990)  
> 标题: `[fully_async] feat: standalone log prob server (Model Engine Server) support`  
> 状态: open，截至 2026-05-13  
> 核心问题: fully async 训练里的 `old_log_prob` 计算太重，并且卡在训练关键路径上。

---

## 1. 这条 PR 到底在做什么？

#5990 引入了一个独立的 `Model Engine Server`，主要为 fully async 训练计算 `old_log_prob`。

它不是一个新的 RL 算法，也不是让 logprob forward 本身变得更便宜。它的核心是系统结构调整：

> 把 `old_log_prob` 从 actor 训练引擎里拆出去，放到额外 GPU 上用独立服务并发计算。

旧方案里，`old_log_prob` 由 actor training engine 自己重算。fully async 场景下，因为训练和 rollout 使用的参数版本可能不同，trainer 经常需要保存当前参数、恢复旧参数、重算 logprob，再把当前参数恢复回来。

新方案里，rollout 生成 token 后，会把 `prompt + response` 发给独立的 `Model Engine Server`。这个 server 用额外 GPU 计算 `engine_server_logprobs` 和 `entropy`，然后把结果随 batch 带回训练阶段。训练阶段看到这些字段后，直接使用，不再自己重算。

当前 PR head 里，入口已经从 PR 描述里的概念性 `enable_standalone` 收敛成了 `model_engine_server.enable` 这套配置：

- `enable_old_mode=True` 会启动 old policy logprob server，并把它纳入 actor 到 server 的权重同步。
- `enable_ref_mode=True` 也有代码路径，可以启动 ref policy logprob server；不过示例脚本目前只打开了 old mode。
- `batch_size` 和 `timeout` 控制 server 侧异步攒批。

---

## 2. 为什么 old_log_prob 会成为瓶颈？

在 PPO/GRPO 这类算法里，`old_log_prob` 是策略更新的锚点。策略损失里通常会用到：

$$
\text{ratio} = \exp(\log \pi_\theta(a|s) - \log \pi_{\text{old}}(a|s))
$$

其中 `old_log_prob` 对应的就是 $\log \pi_{\text{old}}(a|s)$。

同步训练里，`old` 通常比较清楚：采样时的策略就是旧策略。但 fully async 里，rollout、参数同步和训练更新被流水化了，同一个 batch 可能对应一个较早的参数版本。为了得到正确的旧策略 logprob，系统需要重新用旧参数跑一次 forward。

当前 fully async 的旧路径大致是：

```text
rollout 生成样本
  ↓
reward / balance
  ↓
进入训练阶段
  ↓
_compute_old_log_prob
  ↓
如果 local_trigger_step == 1:
    save current model params to CPU
    actor engine compute old_log_prob
否则:
    save current model params to CPU
    restore old version params
    actor engine compute old_log_prob
    restore current version params
    clear temporary CPU copy
  ↓
ref_log_prob / value / actor update
```

这个路径的问题不只是多做了一次 forward，而是它卡在训练关键路径里，并且夹带了很多非计算开销：

- 保存当前 actor 参数到 CPU。
- 恢复旧版本参数。
- 用 actor engine 重算 `old_log_prob`。
- 再恢复当前版本参数。
- 清理临时 CPU copy 和 GPU cache。

所以 `old_log_prob` 的总耗时更接近：

```text
old_log_prob 总耗时
= 权重保存/恢复时间
+ 旧策略 forward 时间
+ 同步等待和缓存扰动时间
```

这会直接拉长一次训练 update 的 wall-clock 时间。

---

## 3. #5990 的新路径

#5990 把 `old_log_prob` 前移到 rollout 侧，用独立的 `Model Engine Server` 计算。当前实现把这条路径挂在 `FullyAsyncLLMServerClient.generate` 之后：每次 rollout server 生成一段新 token，client 会把这段新 token 对应的上下文发给 `ModelEngineServerManager.compute_log_probs`。

新时序大致是：

```text
rollout 生成 token
  ↓
FullyAsyncLLMServerClient 把 context + 本轮新 response 发给 ModelEngineServerManager
  ↓
ModelEngineServerManager 分发给 old/ref Model Engine Server
  ↓
Model Engine Server 异步攒批，ModelEngineWorker 做 forward-only infer
  ↓
返回 engine_server_logprobs / engine_server_entropys
可选返回 ref_logprobs / ref_entropys
  ↓
AgentLoop postprocess 做 padding，batch 带着 old_log_prob 进入训练队列
  ↓
训练阶段 _compute_old_log_prob 直接读取结果
  ↓
actor update
```

训练阶段原来很重的这一段：

```text
save/restore weights + compute old_log_prob
```

被替换成：

```text
pop engine_server_logprobs from batch
```

这就是 PR 测试中端到端加速的主要来源。

验证阶段会跳过这段 server logprob 计算，因为 `generate` 里会读取 `validate` 标记，避免把验证推理也挂到 old logprob server 上。

---

## 4. 新增的几个核心组件

当前实现可以拆成几个角色：

| 类 | 角色 |
|---|---|
| `ModelEngineServerManager` | 管理 old/ref 两类 `ModelEngineReplica`，对上提供统一的 `compute_log_probs` |
| `ModelEngineReplica` | 类似 rollout replica，负责资源分配、生命周期和权重同步 |
| `ModelEngineWorker` | 每张 GPU 上的 Ray actor，继承 checkpoint engine worker 逻辑 |
| `ModelEngineServerAdapter` | 把 `TrainingWorker` 包成 forward-only inference adapter |
| `ModelEngineServer` | 对外服务的 Ray actor，负责异步 batching、排队、pause/resume |

这个设计复用了 verl 已有的 `RolloutReplica`、`BaseRollout`、`CheckpointEngineWorker` 架构。它不是另起一套完整 serving 系统，而是把“旧策略 logprob 推理”伪装成一种特殊 rollout/inference replica。

目前 PR 声明只支持 Megatron backend，并依赖 mbridge 侧的配套改动。

---

## 5. 实际配置入口

示例脚本 `dapo_7b_math_megatron_16_16_8.sh` 现在给出的关键配置是：

```shell
model_engine_server.enable=True
model_engine_server.enable_old_mode=True
model_engine_server.nnodes="${NNODES_LOG_PROB}"
model_engine_server.n_gpus_per_node="${NGPUS_PER_NODE}"
model_engine_server.use_dynamic_bsz=${use_dynamic_bsz}
model_engine_server.megatron.pipeline_model_parallel_size=1
model_engine_server.megatron.tensor_model_parallel_size=1
model_engine_server.batch_size=64
model_engine_server.timeout=5
```

对应的默认配置文件还保留了 `enable_ref_mode=False`，说明这套 `ModelEngineServer` 已经被设计成不只服务 old policy logprob；只是 #5990 的主路径和性能数字仍然围绕 old logprob 展开。

另外要注意，`fully_async_ppo_megatron_trainer.yaml` 的基础配置里仍然写着：

```yaml
actor_rollout_ref.rollout.calculate_log_probs: True
actor_rollout_ref.actor.use_rollout_log_probs: True
algorithm.rollout_correction.bypass_mode: True
```

示例脚本里又覆盖成 `algorithm.rollout_correction.bypass_mode=False`。这意味着默认配置、实验脚本和最终推荐用法还没有完全收敛，blog 里最好把它解释成“PR 当前代码路径/实验脚本的配置组合”，不要写成已经稳定沉淀的用户 API。

---

## 6. 权重一致性问题

独立 logprob server 最大的正确性风险是：权重更新期间，不能让同一个 batch 的 logprob 混用新旧权重。

因此 `ModelEngineServer` 需要一个类似 drain 的协议：

```text
sleep:
    停止接收新请求
    等待正在执行的 batch 完成
    把队列中已有请求用当前权重 flush 掉
    允许外部同步新权重

wake_up:
    新权重加载完成
    重新打开 serving gate
```

这个协议的作用是保证：

- 进入 drain 前的请求使用旧权重。
- 新权重加载完成后的请求使用新权重。
- 不在一次权重切换中间夹杂不明确版本的 logprob。

这部分比“单纯开一个推理服务”更关键，因为 fully async 的核心难点本来就是多版本参数的一致性。

---

## 7. 测试收益主要来自哪里？

PR 描述里给出的结论是：

- 使用 `Model Engine Server` 后，`timing_s/gen` 会略微增加。
- 但可以有效消除 `timing_s/old_log_prob`。
- 在原始方案 `16 GPUs training + 16 GPUs rollout` 基础上，新增 `8 GPUs` 做 logprob server，端到端时间大约提升 `1.64x`。
- 作者还提到考虑新增资源后的收益约 `1.09x`。

这里要区分两个概念。

第一，`1.64x` 是 wall-clock 加速。它的主要来源不是 logprob forward 变快，而是：

> 原来串在训练关键路径里的 `old_log_prob` 被搬走了。

具体收益来自：

- 训练阶段基本不再等待 `timing_s/old_log_prob`。
- actor training engine 不再反复保存/恢复旧版本权重。
- 参数搬运、缓存清理和同步等待减少。
- logprob forward 被放到额外 GPU 上，和 rollout/训练流水线并发推进。

第二，`timing_s/gen` 变慢是合理的。因为 rollout 生成完 response 后，还要等待 logprob server 计算这段 response 的 logprob。只要这部分增加小于训练阶段减少的 `old_log_prob` 时间，端到端就会变快。

因此可以把收益理解成：

```text
新方案收益
= 训练关键路径缩短
- rollout 侧等待 logprob server 的新增开销
- 额外 GPU 资源成本
```

---

## 8. 什么是资源归一化收益？

资源归一化收益是在问：

> 你跑得更快，是不是只是因为用了更多 GPU？

最朴素的口径是按 GPU-time 归一化：

```text
资源归一化收益
= 基线 GPU 数 × 基线耗时 / (新方案 GPU 数 × 新方案耗时)
```

等价于：

```text
资源归一化收益
= 纯时间加速比 / GPU 数量放大倍数
```

假设基线是：

```text
16 training GPU + 16 rollout GPU = 32 GPU
```

新方案是：

```text
16 training GPU + 16 rollout GPU + 8 logprob GPU = 40 GPU
```

GPU 数量放大倍数是：

```text
40 / 32 = 1.25
```

如果端到端 wall-clock 加速是 `1.64x`，按最简单的 GPU-time 口径：

```text
1.64 / 1.25 = 1.31x
```

这表示同样任务消耗的总 GPU 时间也下降了。

但 PR 文本里写的资源归一化收益是约 `1.09x`，比这个简单计算更保守。仅凭公开描述无法精确复原 `1.09x` 的计算方式，可能作者还纳入了其他资源成本、有效利用率、实验平均值，或者使用了不同的归一化口径。

这里重要的不是执着于 `1.31x` 还是 `1.09x`，而是理解它们在回答不同问题：

| 指标 | 回答的问题 |
|---|---|
| wall-clock speedup | 训练多久跑完？ |
| resource-normalized speedup | 算上多用的卡，整体资源效率有没有变好？ |

---

## 9. 额外 GPU 的空泡问题

这个方案的明显弱点是：额外用于 logprob server 的 GPU 平均利用率未必高。

原因很直接：

- 示例路径只负责 `old_log_prob` forward；即便打开 ref mode，本质上也还是 logprob inference。
- 请求流可能是 bursty 的。
- batch 可能攒不满。
- 权重同步期间需要 pause/drain。
- 训练或 rollout 其他阶段慢时，logprob GPU 会等待。

所以这不是一个“提高 GPU 利用率”的优化，而是一个“用额外 GPU 买训练关键路径缩短”的优化。

更准确地说：

> #5990 优化的是 wall-clock 和系统解耦，不是 logprob GPU 自身的利用率。

什么时候它比较值：

- `old_log_prob` 是当前训练主瓶颈。
- GPU 资源相对充足，实验周转时间更重要。
- rollout 请求流足够连续，logprob server 能被喂饱。
- fully async 的 staleness 或权重切换成本已经明显影响训练效率。

什么时候它可能不值：

- 集群 GPU 很紧。
- batch 小、并发低、请求稀疏。
- `old_log_prob` 本来占比不高。
- 主要目标是最大化 `tokens/GPU-hour`，而不是缩短单次实验时间。

---

## 10. 可能的后续优化方向

如果沿着这个方向继续做，关键不是再加更多 logprob GPU，而是提高这组 GPU 的复用率和弹性。

可能的演进方向：

- 当前代码已有 `enable_ref_mode` 的雏形，但 old/ref 是否应该是两个 manager、两组 replica，还是更统一的资源池，还在 review 讨论里。
- 根据请求流量动态调整 `batch_size` 和 `timeout`。
- 让 logprob server 支持 autoscaling，而不是固定常驻 `8` 张 GPU。
- 与 rollout server 或其他 inference worker 做资源池复用。
- 更精细地记录 GPU 利用率、队列等待时间、batch fill ratio、drain 时间，判断真实瓶颈在哪里。

尤其需要关注三个指标：

| 指标 | 含义 |
|---|---|
| `timing_s/old_log_prob` 减少量 | 训练关键路径真正缩短了多少 |
| `timing_s/gen` 增加量 | rollout 侧被 logprob server 拖慢了多少 |
| logprob GPU utilization / batch fill ratio | 额外资源是否被有效使用 |

如果 `timing_s/old_log_prob` 减少很多，但 logprob GPU 利用率很低，这个方案仍然可能适合抢时间；如果资源归一化收益接近 1 或低于 1，就说明它主要是在堆卡换时间。

---

## 11. 截至 2026-05-13 的状态

这篇分析可以继续保留主判断，但需要把“实现状态”写得更谨慎：

- PR 仍是 open，且没有直接更新 `docs/` 或 blog。
- 5 月 12 日的 review 还在追 `device api`、old/ref manager 结构、FSDP 实现等问题。
- 当前示例只展示 Megatron + mbridge + old mode，不能写成 FSDP 或通用 backend 已支持。
- `ModelEngineServerManager` 已经有 old/ref 双实例的代码路径，但 review 里仍在质疑 old/ref manager 的组织方式。
- mbridge 侧仍依赖配套 PR，相关文档可能需要单独补。

所以 #5990 更适合被描述成“fully_async 里把 old logprob 从训练关键路径拆出去的系统方案和进行中的实现”，而不是已经完全落地的稳定 feature。

---

## 12. 一句话总结

> #5990 的本质是把 fully async 训练中昂贵且串行的 `old_log_prob` 计算，从 actor 训练引擎里拆出来，放到独立 GPU 服务上并发执行。它的收益主要来自缩短训练关键路径和减少权重 save/restore，而不是让 logprob forward 本身更便宜。当前 PR 已经加入 `ModelEngineServerManager` 和 old/ref mode 的配置雏形，但示例和性能数字仍主要围绕 Megatron old logprob server；代价是额外 GPU 可能有明显空泡，因此它适合 GPU 相对充足、wall-clock 更重要的训练场景。
