# 异步 RL 中的权重同步：16 个开源框架的传输与中断方案

> 权重同步不是单纯的“把 checkpoint 发给推理服务”。一个完整协议至少要回答两个彼此独立的问题：**新权重通过什么数据通道到达 rollout engine，以及推理在什么安全点切换到新权重。**

在大模型 RL 训练中，trainer 不断执行 `optimizer.step()`，rollout engine 则不断用当前策略生成新轨迹。如果两者部署在不同 GPU 池，训练侧产生的新参数必须周期性地送到 vLLM、SGLang 等推理引擎。同步太慢，rollout 会越来越 stale；同步时暂停推理，又会损失异步流水线本来想获得的吞吐。

因此，权重同步真正优化的是四个互相牵制的目标：

- 减少权重 materialize、格式转换与网络传输时间；
- 缩短或消除 rollout 的暂停窗口；
- 保证推理不会看到一份意外的“半新半旧”模型；
- 控制生成策略与当前训练策略之间的版本差。

本文主要整理 Hugging Face 在 2026 年 3 月发布的 [Keep the Tokens Flowing: Lessons from 16 Open-Source RL Libraries](https://huggingface.co/blog/async-rl-training-landscape#axis-3-weight-synchronisation-protocol) 中的权重同步部分，并结合该文的 partial rollout、LoRA 和 distributed backend 章节解释其工程含义。

> 文中的框架能力表是 **2026 年 3 月综述快照**，不是永久不变的产品规格。延迟数字也来自综述中的量级归纳，不是控制了模型大小、精度、互联拓扑和并行策略后的统一 benchmark。

资料口径上，本文用 2026 年 3 月 9～10 日附近的固定 commit 校勘原表；“当前状态”只补充核对到 2026 年 8 月 3 日的少数明确变更，并不代表对 16 个仓库最新版的完整审计。

---

## 一、为什么异步 RL 会把权重同步变成核心问题

### 1. 同步训练中的权重切换

传统同步 RL 的一轮通常是：

```text
rollout
  → reward
  → advantage / loss
  → backward
  → optimizer.step()
  → 更新 rollout model
  → 下一轮 rollout
```

生成和训练严格轮流执行。权重更新发生时没有在途请求，因此最容易保证：一条 trajectory 全部由同一个 policy version 生成。

代价也很明显：自回归生成时 trainer GPU 空闲，训练时 rollout GPU 空闲；长 CoT、tool use 和 agent trajectory 还会把长尾等待放大。

### 2. 异步训练中的两个独立循环

异步 RL 会把两类工作拆到不同 GPU 池：

```text
Inference GPU pool                         Training GPU pool

W_r 生成 rollout ───→ rollout buffer ───→ 用样本更新 W_k
       ↑                                      │
       └──────────── weight sync ─────────────┘
```

设 trainer 当前参数版本为 $k$，某条 rollout 使用的版本为 $r$，则它的策略版本差可以写成：

$$
\Delta_{\text{version}} = k-r
$$

只要生成和训练并行，$\Delta_{\text{version}}$ 就可能大于 0。更快、更频繁的同步通常能减小版本差，但不能单独消除 off-policy 问题；队列深度、生成时长、同步频率和 trainer 更新速度都会影响它。

### 3. 一次同步实际上有五段成本

“权重传输耗时”只是整个同步路径的一部分：

```text
训练分片中的 W_k
  → materialize / gather
  → training layout 转 inference layout
  → NCCL、IPC、文件或 HTTP 传输
  → rollout engine staging / load
  → 在安全点 commit，并发布 model_version = k
```

可以粗略写成：

$$
T_{\text{sync}}
=
T_{\text{materialize}}
+T_{\text{convert}}
+T_{\text{transfer}}
+T_{\text{install}}
+T_{\text{barrier}}
$$

其中：

- `materialize`：从 FSDP、ZeRO 或 Megatron shard 中收集待发送参数；
- `convert`：把训练侧命名、切分和 dtype 转成推理侧可加载布局；
- `transfer`：真正经过 GPU fabric、主机内存、对象存储或网络传输；
- `install`：推理引擎把参数装入模型，必要时重建量化或 kernel 派生状态；
- `barrier`：等待在途请求结束、终止或到达安全切换点。

所以，不能只看到“用了 NCCL”就推断同步一定快，也不能只看到“异步传输”就推断生成不会暂停。

---

## 二、比较方案时必须拆开两条轴

Hugging Face 综述最重要的判断，是把权重同步拆成两个正交问题。

### 1. Transport：权重字节怎么过去

这是 data plane，关注：

- 发送 full weights、LoRA adapter 还是其他 delta；
- 通过 NCCL collective、CUDA IPC、shared memory、filesystem 还是 HTTP；
- 按单个 parameter 发送，还是先 pack 成较大的 bucket；
- 是否需要跨训练与推理框架做 reshard、rename 和 dtype conversion。

### 2. Interrupt model：推理什么时候切换

这是 control plane，关注：

- 新请求是否继续进入 scheduler；
- 已经在 decode 的请求是继续、drain、abort 还是保存后恢复；
- 每次 forward 是否只看到一个完整版本；
- 一条 sequence 是否允许跨越多个 policy version；
- 新权重是一次 atomic commit，还是逐参数边收边装。

这两条轴没有一一对应关系。两个框架即使都用 NCCL，也可能分别选择 per-forward swap 和 per-batch blocking；同一个框架也可能在 NCCL 与 filesystem 之间提供多种 transport，却共享同一种暂停语义。

> **transport 决定“搬得多快”，interrupt model 决定“谁要等、等多久，以及在途 trajectory 发生什么”。**

---

## 三、主要权重传输机制

原综述给出的 transport 分类如下。表中的延迟只应理解为作者观察到的典型量级。

> **注意：下面两张分类表先忠实呈现原综述口径，其中包含已由同期源码证伪或需要降格的归类。代码校勘后的框架结论统一放在第六节。**

| 传输机制 | 综述中的延迟量级 | 综述归类的框架（未经代码校勘） | 核心特点 |
|---|---:|---|---|
| NCCL Broadcast | 约 100–500 ms | PipelineRL、SkyRL、SLIME、MILES、ROLL、OAT、NeMo-RL、PRIME-RL、open-instruct、AReaL | GPU 间直接广播，是全量权重同步的主流方案 |
| NCCL + Bucketing | 约 20 ms | verl | 对参数分组，部分实现还会 flatten / pack，以减少调度或 collective 开销 |
| KV / Shared Memory | 低 | TorchForge | 通过 key-value store 协调，用共享内存预取数据 |
| Filesystem + HTTP | 中 | PRIME-RL、AReaL、ART | 文件承载权重，HTTP 负责通知、版本与加载控制 |
| CUDA IPC | 很低 | NeMo-RL、MILES | 同机 GPU 间传递 CUDA memory handle，避免网络和多余拷贝 |
| JAX Cross-mesh | 低 | Tunix | 在不同 JAX device mesh 之间 reshard / `device_put` |
| HTTP PUT | 高 | verifiers-rl | 这是综述 Axis 3 的原始标注；同日期源码实际是 HTTP 控制请求 + PyNCCL 数据广播，详见版本说明 |
| Filesystem + Restart | 很高 | Atropos | 保存 checkpoint 后重启或重载推理服务，最朴素也最慢 |

同一框架可能同时出现在多行中，因为它会针对 colocated、同机 disaggregated 或跨节点部署提供不同 backend。表中用“或”连接的路径是可选项，不表示一次同步必须依次经过所有机制。

### 1. NCCL Broadcast：默认答案

典型做法是建立一个独立于训练通信组的 weight-update process group，把 trainer rank 和 rollout ranks 放进同一个 NCCL communicator：

```text
trainer rank / shards
        │
        │ broadcast(parameter or bucket)
        ▼
rollout replica 0: TP ranks
rollout replica 1: TP ranks
rollout replica 2: TP ranks
```

它的优势是权重可以留在 GPU 上，直接利用 NVLink、NVSwitch 或 RDMA 网络。真正困难的部分通常不是 `broadcast` 本身，而是：

- FSDP / ZeRO shard 如何聚合；
- Megatron TP、PP、EP 布局如何映射到 vLLM / SGLang 的 TP 布局；
- 多个 rollout replica 如何只接收自己需要的 slice；
- 参数名、fused QKV、MoE experts 和量化格式如何对应；
- 怎样避免为了 staging 同时保留新旧两份大模型而 OOM。

NCCL 只是 data plane。框架仍然经常通过 HTTP、Ray RPC 或其他 control plane 通知推理 worker：“下一个 tensor 是什么”“什么时候开始加载”“本次版本是否成功提交”。

### 2. Bucketing / Packing：先区分分组与打包

朴素实现会对 state dict 中的每个 parameter 分别发起一次 RPC 和一次 collective。大模型可能包含成千上万个 tensor，控制开销和同步次数会盖过纯带宽成本。

Bucketing 首先表示按累计字节数把 parameters 分组调度；packing 则更进一步，把一组 tensor flatten 到连续 buffer。两者都在优化小 tensor 过多的问题，但减少的开销不同：

```text
p0, p1, p2, ... pn
      ├─ size-bounded grouping → 一个 bucket 内仍逐 tensor broadcast
      └─ flatten / pack        → contiguous buffer → 更少的 broadcasts
```

综述把 verl 单列为 `NCCL + Bucketing`，并给出约 20 ms 的量级。这个数字不能脱离具体模型与硬件复用；真正可迁移的结论是：**当链路带宽已经很高时，减少 per-parameter 调用次数通常比继续优化单次传输更重要。**

vLLM 后来提供的 [`NCCLWeightTransferEngine`](https://docs.vllm.ai/en/stable/api/vllm/distributed/weight_transfer/nccl_engine/) 也把 `packed=True` 作为原生能力：多个权重可以成批 broadcast，再增量加载到模型中。

以综述快照中的 SLIME 为例，它按总字节阈值组成 parameter bucket，但 bucket 内仍对 tensor 发起异步 broadcast；这和 vLLM `packed=True` 的连续 flat-buffer 方案不能直接画等号。

### 3. CUDA IPC：映射同机 GPU allocation

CUDA IPC 不是把 tensor 内容复制到另一个进程，而是把一块 GPU allocation 的 handle 交给同机进程，让接收方映射这块内存。它非常适合 training 与 inference 分进程、但位于同一台机器的场景。

优势是低延迟、少拷贝；限制是不能把同机优化直接当成通用多节点协议。生命周期管理也更严格：发送方 buffer 不能在接收方使用完成前释放，进程崩溃和重启时还要清理失效 handle。

### 4. KV / Shared Memory：运行时协调与本机预取

综述把 TorchForge 的路径写作 `torchstore + shared memory prefetch`。这里的 key-value store 主要承担 metadata、版本和协调，shared memory 则让本机进程预取或交换数据。

它不必然等同于 CUDA IPC：shared memory 可能是 host-side 共享内存，CUDA IPC 则特指跨进程映射 CUDA allocation。两者都受益于本机拓扑，但内存位置、生命周期和拷贝路径不同。

### 5. Filesystem + HTTP：数据面与控制面解耦

这一类方案通常让 trainer 把 safetensors 或 adapter checkpoint 写到共享存储，再通过 HTTP 告诉 inference server：

```text
POST / publish checkpoint metadata or load request
  → pause or drain
  → load path=/checkpoints/policy-v42
  → validate
  → activate version 42
  → ACK
```

它的优点是容易跨语言、跨服务和跨故障域部署，checkpoint 也天然可审计、可回滚。代价是序列化、文件系统带宽、元数据操作和 reload 通常比 GPU-direct 传输慢。

当只同步体积很小的 LoRA adapter 时，这条路径会重新变得有吸引力：base model 不动，只保存并 hot-swap adapter，原本昂贵的全量文件传输不再存在。

### 6. JAX Cross-mesh：同一个问题的 XLA 版本

Tunix 位于 JAX / XLA 生态。训练 mesh 和推理 mesh 的 sharding 可以不同，因此同步的核心是跨 mesh `device_put` 与 reshard，而不是 PyTorch 生态常见的 NCCL process-group 扩展。

抽象层面仍然相同：先把训练布局转换成推理布局，再决定何时让生成侧看到新版本。

---

## 四、推理中断模型：新权重什么时候生效

原文称方案可以归纳为 “five conceptual tiers”，但当前表格实际只列出四档。本文按表格中的四种中断粒度整理，不补造第五档。

| 中断粒度 | 权重到达时发生什么 | 原综述列出的框架（未经代码校勘） |
|---|---|---|
| In-flight / per-forward-pass | sequence 不停止，在相邻 decode forward 之间切换权重 | PipelineRL、open-instruct（可选） |
| Per HTTP request：abort + resync | 终止在途请求，保存 partial tokens，再 prefix-resume 或放回队列重试 | SkyRL、SLIME、MILES |
| Soft pause：drain in-flight | 停止接收新请求，让已开始的请求自然结束，再同步权重 | PRIME-RL、AReaL、open-instruct（默认）、verl（async） |
| Per training step / batch | 等完整生成 batch 结束，训练和推理在同步点互相阻塞 | NeMo-RL、ROLL、OAT、TorchForge、Tunix、verifiers-rl、Atropos |

这张表是对原综述 Axis 3 的复述，不是代码审计后的最终事实。后文固定到 2026 年 3 月同日代码后可以看到，AReaL 实际会 abort 在途请求，NeMo-RL 已有可选 in-flight update，PRIME-RL 也没有明确的 active-request natural drain；因此框架选型应以后面的校勘表为准。

### 1. Per-forward-pass：暂停粒度最细，但 sequence 会跨版本

PipelineRL 把锁放到 transformer forward 的边界。一次 decode step 持锁执行，权重更新最多等待当前 forward 完成；swap 后，下一个 token step 继续使用新权重。

```text
token t:     forward(W_17) ── unlock
weight sync:                swap W_17 → W_18
token t+1:                              forward(W_18)
```

这样不会丢弃 partial rollout，也不会等待整条长序列完成。代价是同一条 sequence 可能由多个版本共同生成：前半段来自 $W_{17}$，后半段由已安装 $W_{18}$ 的模型继续计算。

这里的“$W_{18}$ 继续计算”只描述 parameter version。如果引擎保留了 $W_{17}$ 生成的 KV cache，那么后续 decode 的计算状态仍然跨版本；per-forward parameter atomicity 并不自动保证 cache / policy-state consistency。

还要进一步区分两种“生成不停”：

- **atomic model swap**：像 PipelineRL 一样，一次 forward 看到的参数集合要么全旧、要么全新；
- **per-parameter streaming**：open-instruct 的可选 in-flight 路径逐参数发送和加载，forward 可能运行在一份部分参数已更新、部分参数未更新的 hybrid model 上。

两者都没有停止 sequence，但一致性强度完全不同。前者是“跨 token 混合 policy version”，后者还可能是“同一 forward 使用混合 parameter version”。

### 2. Abort + prefix-resume：用重算换更细的同步点

SkyRL、SLIME 和 MILES 会在更新到来时处理当前 HTTP generation request：

- abort 正在执行的请求；
- 保留已生成 token、旧 logprob 和必要 metadata；
- 把 prefix 交给新权重继续生成，或把 partial work 回收到队列重试。

这比等整条 agent rollout 完成更及时，但 prefix-resume 不是零成本：新模型若要保持 cache 一致性，通常需要对 prefix 重新 prefill。旧 KV cache 是由旧权重计算出来的，不能在没有明确一致性设计的情况下直接当作新权重的 KV cache 使用。

### 3. Soft pause / drain：不浪费 token，但会被长尾拖住

Soft pause 先关闭入口，不再给 scheduler 分配新请求或新的 KV-cache slot；已经开始的 sequence 正常完成。队列排空后再同步，完成后重新开放 dispatch。

```text
gate new requests
  → wait(active_requests == 0)
  → transfer + install W_k
  → publish version k
  → reopen dispatch
```

它的语义很干净：旧请求全部由旧版本完成，新请求全部由新版本开始，也不会丢掉已生成 token。问题是 pause bubble 取决于最慢的在途 sequence。对于长上下文 coding agent，这个等待可能远大于权重传输本身。

### 4. Batch blocking：最容易正确，也最难充分异步

最粗粒度的方案只在 generation batch 或 training step 边界更新。同步点到来前不会存在可恢复的 partial rollout，因此实现、调试与失败恢复都最简单。

它适合先建立正确 baseline，也适合 rollout 较短、同步不频繁的训练；但当单个 batch 被长尾 completion 拖到分钟级时，推理与训练仍然在大粒度 barrier 上轮流等待。

### 5. ART：不落入原文四档表的同步特例

ART 没有出现在 Axis 3 的 interrupt table 中。综述后文把它描述为 gather-all-then-train 的同步设计：先收齐 rollout，再进入训练；切换时没有在途生成。其主要路径只 hot-swap LoRA adapter，base weights 不移动。

因此它不是一种新的异步中断粒度，而是通过“同步阶段划分 + adapter-only update”大幅缩小了权重同步问题。

---

## 五、真正需要定义的是一致性语义

“是否阻塞”还不够描述一个协议。至少应分别检查三层一致性：

| 一致性层级 | 要求 | 违反后的现象 |
|---|---|---|
| Parameter-set atomicity | 一次 commit 要么发布完整 $W_k$，要么继续使用完整 $W_{k-1}$ | 推理模型由新旧 layer / tensor 拼成 |
| Forward atomicity | 一次 transformer forward 期间参数版本不变化 | 单个 token 的 logits 由不稳定参数集合计算 |
| Cache / state consistency | decode 使用的 KV / prefix state 与当前参数版本兼容，或明确记录其跨版本语义 | 参数已是 $W_k$，attention state 仍由 $W_{k-1}$ 计算 |
| Sequence homogeneity | 一条 sequence 的全部 token 来自同一版本 | trajectory 内部存在分段 policy version |

不同方案做的是不同选择：

- per-parameter streaming 可能连 parameter-set atomicity 都不保证；
- PipelineRL 的 atomic per-forward swap 可以保证一次 forward 的参数集合完整，但不保证 KV state 或整条 sequence 单版本；
- drain-then-sync 可以保住旧 sequence / 新 sequence 的版本边界；
- batch blocking 最容易提供最强、最直观的一致性。

这也解释了为什么 in-flight update 必须和 rollout metadata 一起设计。至少要保存：

```text
token_ids
old_logprobs
model_version（按 sample、segment 或 token）
loss_mask
resume / abort boundary
```

如果训练只给整条 trajectory 标一个版本，而实际在 token 1537 处切过权重，那么这个版本标签无法表达真实 behavior policy。保存 rollout 时的 `old_logprobs` 能支持重要性比率计算，但系统仍需要知道哪些 token 属于哪个版本，才能做诊断、过滤和更细粒度的 staleness control。

> **“传输完成”不等于“模型已原子发布”，“模型已发布”也不等于“当前所有 trajectory 都由新模型生成”。**

---

## 六、16 个框架的方案快照

下表把原综述的 transport、interrupt、partial-rollout 和 global-overview 信息合并到同一视图。它适合用来理解设计空间，不应替代对具体版本代码和配置的检查。

| 框架 | 权重传输路径 | 权重切换与在途 rollout | 主要取舍 |
|---|---|---|---|
| [AReaL](https://github.com/inclusionAI/AReaL) | Chunked / bucketed NCCL，或 filesystem safetensors + HTTP | 暂停入口并 abort running / waiting requests，再同步和恢复；partial rollout 的后续处理取决于上层 rollout 管理 | 同步边界明确且不等待最长请求，但会中断在途生成 |
| [ART](https://github.com/OpenPipe/ART) | 保存并 hot-swap LoRA adapter，base model 不传输 | gather-all-then-train；同步时原则上没有在途请求 | adapter 很小、实现清楚，但不是 generation / training 真正持续重叠的模式 |
| [Atropos](https://github.com/NousResearch/atropos) | filesystem checkpoint + inference restart / reload | batch boundary，无 partial rollout | 最容易跨服务实现，延迟和重启开销最大 |
| [MILES](https://github.com/radixark/miles) | Disaggregated 使用 NCCL；同机场景可用 CUDA IPC | abort 当前请求，把 partial rollout recycle 回 buffer | 更快接收新权重，但需要可靠保存和重放 partial work |
| [NeMo-RL](https://github.com/NVIDIA-NeMo/RL) | 非共置使用 NCCL；共置 vLLM 可用 CUDA IPC + ZMQ，SGLang 使用 HTTP | 原文 Axis 3 表列为 step / batch blocking；同日期官方实现已支持可选 in-flight update，关闭该选项时 drain | transport 抽象完整，但 Axis 3 的无条件 blocking 归类不准确 |
| [OAT](https://github.com/sail-sg/oat) | ZeRO-3 gather 后逐参数 NCCL | training-step / batch boundary，无 partial rollout | 逻辑直接；逐参数调用和 blocking 限制 overlap |
| [open-instruct](https://github.com/allenai/open-instruct) | NCCL broadcast | 默认 drain；可选 in-flight per-parameter update | 一套系统可选强一致性或最大 overlap；in-flight 模式的一致性最弱 |
| [PipelineRL](https://github.com/ServiceNow/PipelineRL) | 独立 NCCL process group + HTTP notification | per-forward atomic swap；sequence 隐式继续 | 几乎不停止生成，但 trajectory 可能跨多个 policy version |
| [PRIME-RL](https://github.com/PrimeIntellect-ai/prime-rl) | Filesystem safetensors + HTTP，或 NCCL | 原综述归入 soft pause；官方快照是 vLLM collective RPC 安装、reset prefix cache，随后取消超龄 group，未见显式 natural drain | 多种传输适应不同部署；需要协调更新、group cancellation 和废弃结果 |
| [ROLL](https://github.com/alibaba/ROLL) | Dedicated update group 上的 NCCL | training-step / batch blocking，无通用 partial resume | 后端覆盖广、语义直接，但更新形成粗粒度 barrier |
| [SkyRL](https://github.com/NovaSky-AI/SkyRL) | NCCL process group | abort + retry with prefix | 适合长 rollout 的细粒度重同步；代价是 abort 与 prefix prefill |
| [SLIME](https://github.com/THUDM/slime) | NCCL process group + size-bounded parameter buckets | abort + recycle partial rollout | 兼顾较快传输与 partial reuse，但它不是 vLLM packed flat-buffer；token、logprob、版本边界管理也更复杂 |
| [TorchForge](https://github.com/meta-pytorch/torchforge) | torchstore / KV coordination + shared-memory prefetch | batch blocking，无 partial rollout | PyTorch-native actor 体系、同机路径高效；同步粒度仍较粗 |
| [Tunix](https://github.com/google/tunix) | JAX cross-mesh reshard / transfer | batch blocking，无 partial rollout | 适合 TPU / XLA mesh，不能直接套用 PyTorch process-group 实现 |
| [verl](https://github.com/verl-project/verl) | NCCL + checkpoint-engine buckets | async 模式可 drain；fully async 路径保存 partial token / logprob 后恢复 | 传输和 partial rollout 能力完整，但状态机与版本管理更复杂 |
| [verifiers-rl](https://github.com/PrimeIntellect-ai/verifiers) | HTTP POST 传参数 metadata，tensor 经 PyNCCL broadcast | depth-1 / batch boundary，无 partial rollout | 浅流水线限制 staleness；Axis 3 的 “HTTP PUT” 把控制面误写成了权重数据面 |

### 版本边界与原文口径差异

原综述自身有几处不能静默抹平的差异：

1. interrupt 段落说有五档，但表中只有四档；
2. 正文把 PipelineRL 称为与“其他所有框架”不同的 outlier，但同一张表也把 open-instruct 的 opt-in 模式放在 never-stop 档；
3. NeMo-RL 在 Axis 3 表中被列为 batch blocking，global overview 却标注 in-flight continuation。核对 2026 年 3 月 9 日的 [NeMo-RL 固定版本文档](https://github.com/NVIDIA-NeMo/RL/blob/280d3aae1625fac463e2529812eaf4191fd778c2/docs/guides/async-grpo.md#L160-L169) 与 [实现](https://github.com/NVIDIA-NeMo/RL/blob/280d3aae1625fac463e2529812eaf4191fd778c2/nemo_rl/algorithms/async_utils.py#L529-L575) 可见，当时已经存在 `in_flight_weight_updates` 选项：启用时不等待 pending generation，关闭时才 drain。因此这不是单纯的后续功能演进，而是 Axis 3 的归类过度简化；
4. AReaL 被归为 soft pause / drain，但同日期的 [vLLM server 实现](https://github.com/inclusionAI/AReaL/blob/b48b013945f505d9cabb8e19c84efed9fb4330cb/areal/engine/vllm_ext/areal_vllm_server.py#L227-L241) 明确执行 `abort_all_reqs`，其 [FSDP 更新路径](https://github.com/inclusionAI/AReaL/blob/b48b013945f505d9cabb8e19c84efed9fb4330cb/areal/engine/fsdp_engine.py#L1085-L1142) 是 pause → chunked / bucketed NCCL → continue。因此“已有序列自然完成”的描述与该快照代码不符；
5. PRIME-RL 被归为 soft pause / drain，但 2026 年 3 月快照的 [vLLM update endpoint](https://github.com/PrimeIntellect-ai/prime-rl/blob/51bd1ad5d49b9dc020c33dd0191db37f6d3ad7f6/src/prime_rl/inference/vllm/server.py#L183-L189) 直接 collective-RPC 安装权重并 reset prefix cache；[scheduler](https://github.com/PrimeIntellect-ai/prime-rl/blob/51bd1ad5d49b9dc020c33dd0191db37f6d3ad7f6/src/prime_rl/orchestrator/scheduler.py#L242-L307) 在更新后按 off-policy age 取消 stale groups。源码不足以支持“所有 active sequence 先自然完成”的强表述；
6. verifiers-rl 在 transport 表中被写成 HTTP PUT，global overview 则是 PyNCCL broadcast。核对 2026 年 3 月 10 日的 [client 实现](https://github.com/PrimeIntellect-ai/verifiers/blob/afde5f80e6ee538fa67aeb99799dba3db80bd7b1/packages/verifiers-rl/verifiers_rl/rl/inference/client.py#L85-L153) 可见，HTTP POST 只传 parameter name、dtype 和 shape 等控制 metadata，tensor 通过 `PyNcclCommunicator.broadcast` 发送；global overview 更准确；
7. SLIME 在 transport 表中归入普通 NCCL，在 global overview 中写作 bucketed NCCL。核对同日期的 [SLIME 分桶逻辑](https://github.com/THUDM/slime/blob/2640e6cd98c864231b570425e0877dcff295984c/slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py#L90-L170) 与 [broadcast 实现](https://github.com/THUDM/slime/blob/2640e6cd98c864231b570425e0877dcff295984c/slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py#L310-L338) 可见，它已经按 buffer-size 阈值组 bucket；不过 bucket 内仍逐 tensor broadcast，不等同于 packed flat-buffer；
8. ART 出现在 filesystem + HTTP transport 行中，却没有进入 interrupt table。两种描述并不冲突：当时它先 [保存 adapter checkpoint，再请求 rollout 服务 load LoRA](https://github.com/OpenPipe/ART/blob/d69345e3e823be125643d6206bf7b71e1e36828f/src/art/unsloth/service.py#L577-L624)；需结合后文的同步 LoRA swap 理解；
9. Axis 3 声明只讨论 disaggregated 模式，却又把主要服务于同机 / colocated 路径的 CUDA IPC 放入 transport 表。它可以作为框架的另一部署后端理解，但不应拿来与跨节点 NCCL 做无条件比较。

### 截至 2026 年 8 月 3 日的有限更新

当前 [NeMo-RL release notes](https://github.com/NVIDIA-NeMo/RL/releases) 也明确描述了可选的 in-flight weight update；其 [WeightSynchronizer 接口文档](https://docs.nvidia.com/nemo/rl/nightly/apidocs/nemo_rl/nemo_rl.weight_sync.interfaces.html) 把 IPC / ZMQ、HTTP 和 NCCL 封装为不同 transport，并要求 rollout workers 全局更新到同一版本。此后 ART 又加入了 [merged / full-weight NCCL](https://github.com/OpenPipe/ART/commit/fb261243fd5d7f6102a41c79c37e20b65c5b0e00) 和 [in-flight LoRA](https://github.com/OpenPipe/ART/commit/8015c0806083fc958a3cc01326be8df48bd5ac22)；verifiers-rl 的独立 trainer 代码则已从 verifiers 仓库 [移除](https://github.com/PrimeIntellect-ai/verifiers/commit/a497ee5242b20a814a8671cb6d130919ebdaf5e0)，当前项目推荐使用 PRIME-RL。复现任何一行方案都应该固定 commit。

因此，选型时应该把综述当作架构地图，再以目标 commit 的代码、配置 schema 和 release notes 为准。

---

## 七、如何选择同步方案

### 1. 先按部署拓扑排除不可能项

| 场景 | 更自然的起点 | 原因 |
|---|---|---|
| trainer 与 rollout 共用 GPU | sleep / wake、in-place reshard、CUDA IPC | 不需要跨节点搬运完整权重，但无法真正同时使用同一 GPU 训练和生成 |
| 同节点、不同 GPU / 进程 | CUDA IPC、shared memory、NCCL | 可以利用本机高速互联并避免文件落盘 |
| 同集群、独立 GPU 池 | NCCL + packing / bucketing | GPU-direct、高带宽，适合频繁 full-weight sync |
| 网络受限或服务边界强 | filesystem / object store + HTTP | 运维边界清楚，容易重试、回滚和审计 |
| JAX / TPU | cross-mesh transfer | 参数布局和运行时原语与 PyTorch 路径不同 |

### 2. 再按 rollout 时长决定中断粒度

- rollout 很短：batch blocking 往往足够，先保证正确性；
- rollout 较长但不能浪费 token：soft pause / drain；
- agent rollout 很长，更新又必须及时：保存 partial state，再 prefix-resume；
- 极端追求 inference occupancy：per-forward atomic swap，但必须接受并记录 sequence 内版本变化；
- 如果只同步 LoRA：adapter-only hot-swap 通常会让 transport 成本显著下降，此时协议复杂度可能比纯带宽更重要。

### 3. 大模型和 MoE 先看布局转换，而不是只看链路

Dense HF model 的权重名和切分已经可能不同；MoE 还要处理 expert parallelism。训练侧每个 EP rank 只拥有部分 experts，而推理侧可能使用另一套 TP / EP 布局。

这时同步瓶颈可能变成：

```text
EP ranks gather experts
  → fused / unfused parameter conversion
  → target TP / EP slicing
  → transfer
  → inference-side load and kernel repack
```

如果没有明确的 source-layout → target-layout 转换计划，再快的 NCCL 也只会更快地传错 tensor。

### 4. 不要把同步频率当成固定常数

每步都同步可以降低 version lag，却可能让通信与暂停吞掉吞吐；每 $K$ 步同步可摊薄开销，却会增加 stale rollout。

更合理的触发条件可以同时考虑：

- trainer / rollout 的当前 `model_version` gap；
- buffer 深度和样本年龄分布；
- 最近同步耗时占 step time 的比例；
- active sequence 的数量和剩余长度估计；
- 可接受的 abort token budget；
- loss 侧是否有 version rejection 或 importance-sampling correction。

权重同步与 staleness management 是相邻但不同的机制：前者控制新参数何时到达，后者决定旧策略数据还能不能训练、如何加权。

---

## 八、一个可落地的权重同步协议应包含什么

如果显存与 backend 支持 staging，最稳妥的是 two-phase publish：先把完整新版本准备好，再原子发布。

```text
1. trainer 生成单调递增的 model_version = k
2. materialize 并转换目标权重
3. 传到 inference staging buffer / path
4. inference 校验 tensor 名、shape、dtype、数量和完整性
5. 等待协议规定的 safe point
6. atomic commit：W_(k-1) → W_k
7. 所有 rollout replica ACK(version=k)
8. scheduler 发布 version=k，并给新样本打版本标签
```

逐参数原地 streaming install 是另一份协议，不能同时承诺“新版本尚未完整时仍可无条件回退到内存中的完整旧版本”：

```text
1. 生成 model_version = k，并准备确定性的 tensor 顺序
2. quiesce inference；或者明确声明允许 in-flight hybrid model
3. 逐参数覆盖并记录 install progress
4. 校验完整性，重建权重派生状态
5. 成功后发布 version=k；失败则从已知 checkpoint 恢复或重启
```

前者用 staging memory 换原子性，后者用更低峰值内存换更复杂的故障恢复。协议必须明确自己属于哪一种，不能只写一个模糊的 `update_weights()`。

### 1. 正确性检查

- 更新是否幂等：同一个 version 重试不会重复破坏状态；
- staging / atomic 路径在传输中断时是否继续使用完整旧版本；streaming 路径是否禁止请求看到 incomplete model，或明确接受这种语义并提供恢复方案；
- 每个 inference replica 是否都安装成功；
- 参数名、shape、dtype 和目标 shard 是否一一对应；
- 量化参数、MoE 映射或 kernel-packed 等权重派生状态是否需要重建；
- resume 后的 token、logprob、loss mask 与 model version 是否对齐；
- KV cache 在跨版本继续时是重算、失效，还是被协议明确允许复用。

### 2. 必须监控的指标

```text
weight_materialize_ms
weight_convert_ms
weight_transfer_ms
weight_install_ms
generation_pause_ms
trainer_rollout_version_gap
rollouts_by_model_version
aborted_tokens / resumed_tokens
sync_success / retry / rollback count
```

尤其要同时区分 `weight_transfer_ms` 和 `generation_pause_ms`。一个传输可能在后台耗时很久却几乎不暂停生成，也可能传输本身很快，却因为 drain 长尾请求造成巨大的 pause bubble。

### 3. 最小验证集

上线新的同步 backend 前，至少应该验证：

1. 同一个固定输入在 trainer model 与完成同步后的 inference model 上，logits / logprob 误差处于预期范围；
2. 连续多次更新后，所有 rollout replica 的 version 和 parameter checksum 一致；
3. 在传输中途注入失败，不会把半更新模型暴露给请求；
4. 长序列在 update boundary 前后，token-level `old_logprobs` 与版本标签可追溯；
5. TP、PP、EP、LoRA、量化和 fused weights 的每一种目标配置都有映射测试；
6. sync、abort、drain 与 shutdown 同时发生时没有死锁。

---

## 九、几个常见误解

### 误解一：用了 NCCL 就是异步权重同步

NCCL 只说明数据怎么传。推理是否暂停，取决于 interrupt model 和 install protocol。

### 误解二：in-flight update 一定会产生“半新半旧”的单次 forward

不一定。per-forward atomic swap 可以让相邻 token 使用不同版本，同时保证每个 token 的一次 forward 使用完整版本；逐参数 streaming 才可能让一个 forward 看到 hybrid parameter set。

### 误解三：同步越快，数据就是 on-policy

不一定。深 rollout queue、长 generation、低同步频率都能造成 policy lag。同步延迟只是 staleness 的一个来源。

### 误解四：prefix-resume 可以直接复用旧 KV cache

不能默认如此。KV cache 依赖生成它的模型权重；换权重后要么重新 prefill，要么明确接受并建模跨版本 cache 的近似误差。

### 误解五：综述中的 20 ms 或 100–500 ms 可以直接用于容量规划

不能。全量参数字节数、dtype、训练与推理 shard 布局、节点数量、NVLink / RDMA 拓扑、bucket 大小和 staging 内存都会改变结果。容量规划必须在目标模型和真实集群上测量五段成本。

---

## 十、结论

主要 RL 框架没有收敛到唯一的权重同步协议，但已经形成清晰的设计谱系：

```text
传输机制：filesystem + HTTP | shared memory | CUDA IPC | NCCL | JAX cross-mesh
传输粒度：per-parameter | size-bounded buckets | packed flat buffers
中断模型：batch blocking | drain | abort/resume | per-forward in-flight
安装语义：staging + atomic commit | in-place streaming install
```

这些选项不是统一的性能排序。LoRA 文件热切换可能比 full-weight NCCL 更快；abort/resume 也可能因为重算 prefix 而输给 drain。真正的比较必须固定 payload、拓扑、并行布局和在途 workload，再同时测量传输时间、暂停时间、废弃 token 与版本差。

如果只记住一句话：

> **选择权重同步方案时，不要只问“用 NCCL 还是文件”，还要问“新权重在哪个安全点生效、在途 trajectory 怎么处理，以及训练侧能否还原每个 token 的真实 behavior policy”。**

---

## 参考资料

- [Keep the Tokens Flowing: Lessons from 16 Open-Source RL Libraries](https://huggingface.co/blog/async-rl-training-landscape) — Hugging Face, 2026
- [Hugging Face Blog 源文件：async-rl-training-landscape.md](https://github.com/huggingface/blog/blob/main/async-rl-training-landscape.md)
- [PipelineRL: Faster On-policy Reinforcement Learning for Long Sequence Generation](https://arxiv.org/abs/2509.19128)
- [PipelineRL 官方博客](https://huggingface.co/blog/ServiceNow/pipelinerl)
- [vLLM NCCLWeightTransferEngine API](https://docs.vllm.ai/en/stable/api/vllm/distributed/weight_transfer/nccl_engine/)
- [verl Fully Async Policy 文档](https://github.com/verl-project/verl/blob/main/docs/advance/fully_async.md)
- [NeMo-RL WeightSynchronizer 接口](https://docs.nvidia.com/nemo/rl/nightly/apidocs/nemo_rl/nemo_rl.weight_sync.interfaces.html)
- [NeMo-RL IPC Weight Synchronizer](https://docs.nvidia.com/nemo/rl/nightly/apidocs/nemo_rl/nemo_rl.weight_sync.ipc_weight_synchronizer.html)
