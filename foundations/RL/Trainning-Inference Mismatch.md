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

### 1.2 MoE模型中的Expert 分配路由
