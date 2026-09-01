# Training–Inference Mismatch

## 1. 原因及应对方法
### 1.1 Dynamic batching 导致数值路径不一致

参考：[Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)

先从一个看起来有点反直觉的现象开始：如果把 LLM 推理的 `temperature` 设成 0，模型理论上应该每一步都选择概率最高的 token，也就是 greedy decoding。既然没有采样随机性，同一个 prompt 多跑几次，输出是不是就应该完全一样？

直觉上是这样，但真实的线上推理服务并不一定如此。Thinking Machines 的这篇文章指出，问题的关键往往不在采样器，而在推理服务的 dynamic batching：你的请求会和多少其他请求一起被处理、处在 batch 里的哪个位置、prefill 和 decode 阶段如何被切分，这些都可能随着服务端负载变化而变化。

这件事为什么会影响输出？因为 `RMSNorm`、`MatMul`、`Attention` 等算子内部都有 reduction，而浮点加法不满足结合律。换句话说，同一组数换一个相加顺序，最后得到的浮点结果可能会有细微差异。dynamic batching 改变 batch size、batch position 或 sequence slicing 时，底层 kernel 可能选择不同的 reduction order、tile 策略、Split-K 或 Split-KV 策略，于是 logits/logprobs 会出现微小差别。

这些差别通常很小，但在 LLM 解码里，小差别并不总是无害的。如果两个候选 token 的概率非常接近，一点点 logit 抖动就可能让 greedy decoding 选择另一个 token。一旦某一步 token 不同，后面的上下文也变了，最终 completion 就会一路分叉。

放到 RL 训练里，这个问题会变得更麻烦。RLHF/RLVR 这类流程通常包含一个 sampler 和一个 trainer：sampler 负责生成 rollout，trainer 再根据这些 rollout 重新计算 logprob、advantage 和 loss。如果 sampler 推理时的数值路径和 trainer 计算 logprob 时的数值路径不一致，那么两边看到的 policy 就不是严格同一个 policy。这样一来，名义上的 on-policy RL 实际上会混入一部分 off-policy mismatch。

Thinking Machines 提出的 batch-invariant kernels，就是在解决这个根因：让单个样本的数值结果不依赖 batch size、batch position 或服务端调度方式。这样同一个请求无论和谁一起被 batch、prefill/decode 怎么被切分，都尽量走相同的数值计算路径，从而消除 sampler 和 trainer 之间由推理系统引入的 logprob mismatch。

但这个方案有明确代价：性能会退化。为了保持 batch-invariant，推理系统不能总是根据当前 batch shape 选择最快的 kernel 配置，例如不能随意切换 tile size、Split-K、Split-KV，或者在小 batch 下换用更合适的 tensor core 指令。Thinking Machines 在 vLLM 上的实验里，默认配置处理一组请求需要约 26 秒；启用未充分优化的 deterministic vLLM 后变成约 55 秒；改进 attention kernel 后降到约 42 秒。也就是说，确定性推理解决了 sampler/trainer mismatch，但会牺牲吞吐和延迟。放到 RL 训练里，这个 trade-off 需要明确摆出来：是接受一定的 off-policy mismatch，还是用推理性能换更严格的 on-policy。

一个可以继续看的工程参考是 [VeXact](https://github.com/verl-project/vexact/)。它不是通用 serving engine，而是 verl/VeOmni 生态里的 zero-mismatch rollout engine，目标是让 rollout 阶段和 FSDP actor 重新计算出来的 logprobs bitwise 对齐。更准确地说，它的作用是一个 TIM-free diagnostic baseline：先构造一个没有 Training-Inference Mismatch 的基线，再用它判断训练不稳定到底来自算法本身，还是来自 sampler/trainer 的系统级 logprob mismatch。对应论文是 [Diagnosing Training-Inference Mismatch in LLM Reinforcement Learning](https://arxiv.org/abs/2605.14220)。

### 1.2 MoE 模型中的 Expert 路由不一致

参考：[Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers](https://arxiv.org/html/2510.11370)

先补一点 MoE 的基本结构。标准 Transformer block 通常包含 Attention 和 FFN；Dense 模型中，每一层只有一套 FFN 参数，经过这一层的所有 token 都使用它。MoE（Mixture of Experts，混合专家）通常把这套 FFN 替换成 $M$ 套相互独立的 FFN，每一套称为一个 expert。Expert 不是一整个模型，Attention、Embedding、LM Head 等部分通常仍由所有 token 共享；expert 也不一定由人工预先指定成“数学专家”或“代码专家”，它们的分工通常是在训练中与 router 一起学出来的。

每个 MoE 层还有一个很小的 gating network，通常称为 router。对于进入该层的某个 token，其隐藏状态为 $\mathbf{x}$，router 先计算 $M$ 个 expert score：$\mathbf{s}=\mathbf{x}\mathbf{W}_r$，再选出分数最高的 $K$ 个 expert，记作 $\mathcal{I}=\operatorname{TopK}(\mathbf{s},K)$。被选中 expert 的输出按照 gating weight 加权求和，可以抽象成：

$$
\mathbf{y}=\sum_{i\in\mathcal{I}}g_i\mathcal{E}_i(\mathbf{x}).
$$

其中，$\mathcal{E}_i$ 表示第 $i$ 个 expert 的 FFN，$g_i$ 表示它在这次混合中的权重。例如，一个 MoE 层共有 8 个 expert，但 `Top-K = 2`，某个 token 可能只经过 expert 2 和 expert 7，最后把两者的输出加权合并。下一个 token，或者同一个 token 到了下一个 MoE 层，都可能选择完全不同的 expert。因此“路由”不是给整条序列永久指定一个 expert，而是**每个 token 在每个 MoE 层各做一次选择**。一些模型还带有所有 token 都会经过的 shared expert，或者使用 sigmoid、分组 `Top-K` 等不同打分规则，但后面讨论的关键仍然是：每层都存在一个离散的 expert 选择集合。

MoE 的好处是，模型可以拥有 $M$ 个 expert 带来的大参数容量，但一次前向只激活其中 $K$ 个，实际计算量主要由 $K$ 而不是 $M$ 决定。分布式运行时还可以把不同 expert 放在不同 GPU 上，通过 Expert Parallel 把 token 派发给对应设备。代价是系统必须额外处理路由、跨卡派发和负载均衡，而本节关心的正是 rollout 与 trainer 是否为同一个 token 选中了同一组 expert。

如果说 1.1 讨论的是微小数值误差从哪里来，那么 MoE 的 `Top-K` 会在此基础上再增加一个离散的“误差放大器”。Dense 层的输入发生小扰动时，输出通常也先发生连续的小扰动；MoE 的 `Top-K` 却是不连续的，只要第 $K$ 名和第 $K+1$ 名的 score 很接近，一点点数值变化就可能让 expert 集合发生翻转。

这不是说 MoE router 本身一定带随机性。即使 rollout 和 trainer 加载同一个 checkpoint、teacher-force 同一串 token，两侧使用的模型实现、fused kernel、计算精度、reduction 顺序、dynamic batching、sequence packing，以及 TP、CP 等并行路径也可能不同，从而让 router 的输入或 score 出现细微差异。Expert Parallel 中逻辑 expert 的物理放置本身不应该改变路由，但 dispatch、capacity/drop 规则和逻辑 ID 到物理槽位的映射如果在两套执行栈中语义不同，也会进一步造成差异。

一旦 `Top-K` 翻转，变化就不再只是末位浮点误差：token 会进入另一组 FFN 参数，新的隐藏状态继续传到后续层，最终可能形成明显而且重尾的 logprob mismatch。即使两侧最后生成或重放的是同一个 token，这个 token 的概率也可能已经不同，所以“completion 一样”不能作为训推一致的充分证据。

放到 RL 训练里，rollout 侧真正的行为策略可能让 token 经过 expert A/B，而 trainer 重算 old logprob 或执行 policy update 时却让它经过 expert C/D。这样一来，importance ratio 在参数尚未更新前就可能偏离 1，名义上的 on-policy 更新混入了执行系统额外引入的 off-policy 偏差；更直接的问题是，生成行为所经过的 expert 和承担这次 policy gradient 的 expert 不再是同一组。PPO/GRPO 的 clipping 主要约束策略更新本身，并不能从根因上修复这种跨引擎计算路径差异。

R3 论文在 Qwen3-30B-A3B、SGLang rollout 和 Megatron trainer 这一组实验中观察到：约 10% 的 token×MoE-layer 路由位置在两侧选择了不同 expert，94% 的 token 至少在一个 MoE 层出现过 expert 集合差异。这里的 94% 不表示 94% 的 token 生成错误，只表示它们至少经历过一次不同的离散计算路径；这些数字也依赖具体模型和执行栈，不能直接当成所有 MoE 系统的固定比例。

针对这个问题，论文提出 **Rollout Routing Replay（R3）**：rollout 生成轨迹时，记录每个 token、每个 MoE 层实际执行的 `Top-K` 全局逻辑 expert ID；trainer 在 old-policy logprob recompute、current-policy update，以及 activation checkpointing 触发的 forward recompute 中，重放同一组 expert mask。这样，同一条轨迹在生成和训练时至少会经过相同的离散专家路径。

R3 重放的是 expert ID/mask，而不是 rollout 侧完整的 router logits 或 gating weight。Trainer 仍然计算自己的 router score，并在固定的 expert 集合内计算混合权重，因此被选中 gate 的梯度仍可以回传，router 参数也没有被冻结。这个区别很重要：R3 是把训练目标约束在 rollout 实际走过的路径上，而不是复制 rollout 的完整前向结果。

这个方案也有工程代价。路由 trace 的体积随 token 数、MoE 层数和 `Top-K` 线性增长，并且必须与真实 token 位置严格对齐；prompt、response、工具 observation、sequence packing 和并行切片都不能错位。多轮任务如果复用 prefix KV cache，还要把对应 prefix 的路由一起缓存和恢复，否则 cache hit 跳过的那部分 forward 将缺少可重放的路由信息。

最后，R3 只能消除 MoE 特有的离散路由差异，不能自动对齐两侧的 gating weight、attention/FFN kernel、量化精度、sampling processor，也不能解决异步训练中的权重陈旧。在原论文的同一组实验里，R3 将 chosen-token 上的估计 KL 从 $1.535\times10^{-3}$ 降到 $7.5\times10^{-4}$，接近其 Dense 对照的 $6.4\times10^{-4}$，但并没有降到 0。因此更准确的说法是：R3 显著缩小了 MoE routing 引入的 Training-Inference Mismatch，而不是提供 bitwise 的 zero-mismatch。

更完整的算法推导、R2/R3 区别、数据契约、缓存与并行实现、验证指标和适用边界，可以继续看同目录的 [R3 技术详解](<./R3：用Rollout Routing Replay解决MoE Agentic RL的训推不一致.md>)。

### 1.3 用 TIS 对训推分布偏差做算法校正

如果暂时无法让 rollout 与 trainer 的数值路径完全一致，可以把 rollout 保存的真实行为概率记作 $\mu$，把训练后端在同一旧权重下重算的概率记作 $q$，使用重要性比率 $\rho=q/\mu$ 修正策略梯度。为了避免极端 ratio 让少数 token 主导更新，TIS（Truncated Importance Sampling）进一步使用 $\min(\rho,C)$ 截断权重。

TIS 不会消除训推差异，也不能让 KL 自动归零；它是在保留吞吐的前提下，用一定偏差换取更低的梯度方差。它与 PPO clipping 的作用也不同：PPO ratio 管理当前策略相对旧训练策略的更新幅度，TIS ratio 校正旧训练策略与真实 rollout 行为策略之间的分布差异。

完整的公式推导、token/sequence/prefix 粒度、GRPO 接入伪代码、阈值选择、监控指标和适用边界，可继续阅读同目录的 [TIS 技术详解](<./TIS：用截断重要性采样缓解LLM RL训推不一致.md>)。
