# verl 增量权重同步：从 Shard-local Delta 到跨训练后端 HF 导出

> `delta_sharded` 的关键不只是“少传一些权重”，而是把差分点下沉到每个训练 rank 的本地 shard，并要求训练后端直接产出最终 Hugging Face 坐标中的稀疏 patch。这样才能同时绕开全量参数 materialize、训练侧 full gather 和完整模型快照三项成本。

大模型 RL 训练通常维护两份策略模型：trainer 上的可训练模型，以及 vLLM、SGLang 等 rollout engine 中用于生成轨迹的推理模型。每次 `optimizer.step()` 之后，新的 actor 权重都要从训练并行布局迁移到推理布局。在 7B 模型上，这可能只是一个可以接受的同步点；到了 72B、235B MoE，权重同步会成为每轮都要支付的秒级甚至分钟级成本。

verl 在 2026 年 7～8 月合入的 `delta_sharded` 系列把这条链路改造成了稀疏增量协议。它不发送数值差 `W_new - W_old`，而是发送“发生变化的位置 + 该位置的新值”，接收端按位置覆盖。因此它没有阈值截断和累加漂移，目标是与同 dtype 的全量权重同步 bit-exact。

本文聚焦这套方案的设计与当前实现，而不是重复比较不同 RL 框架。框架横向比较可参见同目录的[《异步 RL 中的权重同步：16 个开源框架的传输与中断方案》](主要RL框架重同步方案.md)。

资料口径固定为 **2026 年 8 月 3 日**，代码基线为当天 verl `main` 的 commit [`8bda4220`](https://github.com/verl-project/verl/commit/8bda42207cc08a947a49587d38315647740b9e14)。仓库中的 [`docs/advance/delta_weight_sync.md`](https://github.com/verl-project/verl/blob/8bda42207cc08a947a49587d38315647740b9e14/docs/advance/delta_weight_sync.md) 部分段落仍停留在 Megatron 支持合入之前，因此本文以固定 commit 的代码和已合入 PR 为准。文中的性能数字来自 PR 作者在指定模型、硬件和拓扑上的报告，不是本文独立复现的统一 benchmark。

核心演进如下：

| 日期 | PR | 解决的问题 |
|---|---|---|
| 2026-07-16 | [#6974](https://github.com/verl-project/verl/pull/6974) | 引入 `delta_sharded`：本地 shard 差分、稀疏 gather、NCCL 广播和 SGLang 原地 apply |
| 2026-07-25 | [#7144](https://github.com/verl-project/verl/pull/7144) | 用 `BlockPlacement` 支持一般矩形分片，并把训练布局到 HF 坐标的转换下沉给 backend |
| 2026-07-29 | [#7085](https://github.com/verl-project/verl/pull/7085) | 接入 VeOmni FSDP2+EP，处理 fused expert stack 和手工 EP 切分 |
| 2026-07-31 | [#7181](https://github.com/verl-project/verl/pull/7181) | 接入 Megatron-Bridge TP/EP/ETP 和 hybrid-Mamba，并加入 `verify_every` |
| 2026-08-03 | [#7223](https://github.com/verl-project/verl/pull/7223) | 补齐 Megatron PP/VPP、全局参数目录、非 owner 占位行和 EP slot union |

---

## 一、先建立基线：全量权重同步到底花在哪里

一次权重同步不等于一次 `broadcast`。完整链路至少包含五段：

```text
trainer local shards
  │
  ├─ 1. materialize / all-gather
  ├─ 2. training layout → HF checkpoint layout
  ├─ 3. bucket / pack / trainer→rollout transport
  ├─ 4. HF layout → inference-engine internal layout
  └─ 5. install、cache invalidation 与 generation resume
```

可以把同步时间粗略写成：

$$
T_{sync}
=T_{materialize}
+T_{convert}
+T_{gather}
+T_{wire}
+T_{install}
+T_{control}
$$

其中，`T_wire` 只是网络传输本身。对于 FSDP、Megatron 或 MoE，真正昂贵的部分经常是把训练 shard 重新拼成推理端能够识别的 tensor：

- FSDP 需要从 DTensor 或 sharded state dict 中导出参数；
- Megatron 的 TP、EP、ETP、PP/VPP 布局要映射成 HF key 和 shape；
- fused QKV、`gate_up_proj` 和 grouped expert stack 需要拆分、重排或改名；
- rollout 端还要经过自己的 `load_weights()`，映射到 TP slice、fused kernel 或派生权重。

### 1. 全量 NCCL 路径的结构性成本

传统 `nccl` checkpoint engine 大致执行：

```text
每个 actor rank 的 local shard
        │
        ├─ full tensor assembly / conversion
        ▼
actor rank 0 的 HF tensor stream
        │
        ├─ bucket + NCCL broadcast
        ▼
每个 rollout checkpoint worker
        │
        ├─ same-GPU CUDA IPC
        ▼
rollout engine.load_weights(full tensors)
```

即使 GPU fabric 很快，训练侧仍然要为每次同步支付 full tensor assembly。模型越大、并行布局越复杂，这部分越难被传输 overlap 隐藏。

### 2. 为什么“先 full gather，再在 rank 0 做 diff”还不够

一个直觉方案是：先像全量路径一样把所有参数拼到 rank 0，再与 rank 0 保存的上一版完整模型做 diff，最后只广播变化部分。

它确实降低了 trainer→rollout 的 wire bytes，却保留了两个主要瓶颈：

- 每一步仍要 materialize 和 gather 完整参数；
- rank 0 仍要保存一份完整模型快照。

这也是 [#6974](https://github.com/verl-project/verl/pull/6974) 最终删除 full-gather `delta` 变体、只保留 `delta_sharded` 的原因。真正重要的优化不是“在 wire 前做 diff”，而是“在 full gather 前做 diff”。

## 二、核心思路：稀疏 patch，而不是数值增量

### 1. `delta` 实际发送什么

对本地 shard 中的每个元素，verl 不是计算：

$$
\Delta w_i=w_i^{new}-w_i^{old}
$$

而是比较新旧表示的 bit pattern：

```python
mask = new.view(integer_dtype) != snapshot.view(integer_dtype)
indices = nonzero(mask)
values = new[indices]
```

wire 上发送的是：

```text
(parameter / HF slot, indices, replacement values)
```

接收端做的是 overwrite：

```text
rollout_weight[indices] = replacement_values
```

因此，这是一份稀疏 replacement patch，而不是需要反复累加的数值 delta。它有三个直接后果：

- 没有近似阈值，不会因为丢弃“小 delta”而逐步漂移；
- 重放后的目标是与全量同步到 rollout 的表示 bit-exact；
- 稀疏率由 rollout 可见 dtype 的表示变化决定，而不是由梯度是否非零决定。

### 2. 为什么 BF16 表示会呈现时间稀疏性

FSDP、VeOmni 和 Megatron 的稳态 exporter 都会把 floating shard 转成 BF16 再做差分。即使 FP32 master weight 在优化器中发生了微小变化，只要它没有跨过 BF16 的 rounding boundary，rollout 可见的 BF16 bit pattern 就不变，也无需重新发送。

所以，[#6974](https://github.com/verl-project/verl/pull/6974) 中的 1～3% 更准确的含义是：

> 在报告所使用的小学习率 RL workload 中，相邻同步版本的 **BF16 changed-element ratio** 约为 1～3%。它不是所有模型、所有训练阶段都成立的常数。

MoE 还能叠加路由稀疏性：某一步没有被路由到的 expert 可能完全不更新。[#7085 的 235B 三步数据](https://github.com/verl-project/verl/pull/7085#issuecomment-5027537258)中，changed ratio 甚至从约 0.05% 降到 0.02%，但这个数字同样依赖 workload。

### 3. 稀疏 wire 的简单成本模型

设：

- 完整 rollout 权重共有 $N$ 个等 dtype 元素；
- 每个 value 占 $s$ 字节；
- changed-element ratio 为 $r$；
- 当前位置编码固定为 4 字节 `int32`。

忽略 manifest 后：

$$
B_{full}=Ns
$$

$$
B_{delta}\approx Nr(s+4)
$$

所以：

$$
\frac{B_{delta}}{B_{full}}\approx r\left(1+\frac{4}{s}\right)
$$

对于 BF16，$s=2$：

$$
\frac{B_{delta}}{B_{full}}\approx 3r
$$

当 $r=1\%\sim3\%$ 时，稀疏 payload 约为全量 value bytes 的 3～9%。忽略 metadata 和固定开销，BF16 的理论 wire break-even 是 $r<1/3$。当前实现没有 density-aware fallback；如果 changed ratio 超过约 33%，`indices + BF16 values` 的逻辑 payload 反而可能大于全量 BF16。

这里比较的是逻辑 payload，不是 NCCL 在具体拓扑上的物理链路总流量，也不代表同步时间会同比缩短。diff 扫描、格式转换、collective latency 和 rollout apply 仍然存在。

## 三、`delta_sharded` 为什么比普通 delta 更关键

### 1. 把 diff 推到每个训练 rank

`delta_sharded` 的稳态路径是：

```text
actor rank 0        actor rank 1                 actor rank n
local shard         local shard                  local shard
    │                    │                            │
BF16 bitwise diff   BF16 bitwise diff            BF16 bitwise diff
vs CPU snapshot     vs CPU snapshot               vs CPU snapshot
    │                    │                            │
changed idx/value   changed idx/value             changed idx/value
    └──────────────── sparse gather ─────────────────┘
                             │
                        actor rank 0
                             │
                    bucket + NCCL broadcast
                             │
                    rollout checkpoint workers
```

设完整 rollout 权重大小为 $M$，训练侧有效分片数为 $P$。理想情况下，每个 rank 只保存约 $M/P$ 的 CPU snapshot，并扫描自己的本地 shard。没有任何 rank 需要保存完整模型的历史副本。

这个设计同时减少：

- trainer 内部 gather 的数据量：从完整参数降为 changed elements；
- actor rank 0 的历史状态：从 full-model snapshot 降为本 rank shard snapshot；
- trainer→rollout 的逻辑 payload；
- rollout 端额外的 full-model staging mirror。

但要注意两个限定：

- 第一次 seed 和周期性 verify 仍走 full export；
- 稳态 diff 仍要扫描全部本地 shard，并把 CPU snapshot 搬回 device 比较，再刷新 snapshot。它不是 $O(nnz)$ 的差分算法。

### 2. 两套 collective，不要混为一谈

当前实现同时使用两类通信组：

```text
训练侧 torch.distributed groups
  └─ 所有相关 actor ranks 做 shard sparse gather

Ray NCCL trainer→rollout group
  └─ actor global rank 0 是 transport rank 0
  └─ rollout checkpoint workers 是 rank 1..R
```

非零 actor rank 不参加 trainer→rollout broadcast，但它们必须参加训练侧的 sparse gather 和 full seed export collective。最终只有 actor rank 0 负责把拼好的最终 HF patch 广播给所有 rollout worker。

## 四、一次同步的完整生命周期

### 1. Control plane：成功路径仍会暂停 rollout

`delta_sharded` 优化的是数据面，不等于无中断模型切换。当前 [`CheckpointEngineManager.update_weights()`](https://github.com/verl-project/verl/blob/8bda42207cc08a947a49587d38315647740b9e14/verl/checkpoint_engine/base.py#L486-L538) 的顺序是：

```text
abort in-flight requests，并保存 partial rollout
  → 汇总 rollout checkpoint workers
  → release KV cache，保留 live weights
  → prepare / build topology / init process group
  → actor send 与 rollout receive/apply 并发执行
  → finalize
  → resume KV cache
  → resume unfinished generation
```

因此，在成功路径上，虽然模型内存是逐 flush 原地修改的，但 generation 在整个切换期间处于 quiescent 状态。新请求恢复后看到的是完整的新版本，不会在一次 forward 中观察到“半新半旧”的权重。

更准确的说法是：

> 当前方案通过暂停请求获得成功路径上的 observational atomicity，而不是通过双模型指针交换实现真正的 atomic model swap。

### 2. Seed：第一次仍然是全量同步

每个 checkpoint-engine 实例起始时 `_shard_seeded=False`。第一次同步执行：

```text
backend.get_per_tensor_param()          # 已有的 full HF export
  → rank 0 流式发送 values-only dense buckets，rollout 并发 apply
  → full generator 遍历完成后，sender 标记 _shard_seeded=True
  → send_weights 调用 backend.prime_delta_snapshots()
  → manager 等待 actor send 与 rollout apply 两侧全部完成
```

Seed 直接复用各训练 backend 已经验证过的全量导出：FSDP full-tensor assembly、VeOmni expert restack、Megatron-Bridge `export_hf_weights()`。这样新 job 或 actor 进程重启后会自然重新建立 authoritative base，也避免为增量协议重新实现一套首次加载逻辑。

所有 actor rank 都必须完整迭代 full export generator，因为其中可能包含 collective；只有 actor rank 0 真正组 bucket 和发送。Seed 不携带 positions，只发送 values。这里没有逐 flush ACK，也不能把 `prime_delta_snapshots()` 理解为“确认 rollout 已完整加载之后才执行”的 barrier；双方的一致完成由 manager 在本次同步尾部统一等待。因此 seed/prime 中途失败同样不能安全地在原进程内直接重试。

### 3. Steady：本地差分、坐标转换、稀疏传输

之后每次同步执行：

```text
get_per_tensor_param_delta_shard()
  → local shard 与 CPU snapshot 做 bitwise compare
  → 立即刷新 snapshot
  → backend 转成最终 HF slot / coordinate
  → 按 gather group 批量 sparse gather
  → actor rank 0 组装 DeltaFlush
  → ZMQ manifest + NCCL positions/values
  → SGLang custom loader 原地 apply
```

Actor worker 会把整个 training engine 交给 delta engine，而不是先生成普通 weight iterator，因为 seed、steady 和 snapshot prime 共同组成一个有状态协议。相关状态机在 [`DeltaShardedCheckpointEngine.send_weights()`](https://github.com/verl-project/verl/blob/8bda42207cc08a947a49587d38315647740b9e14/verl/checkpoint_engine/delta_checkpoint_engine.py#L572-L666)。

### 4. Verify：周期性全量幂等校验

设置 `verify_every=K` 后，每第 K 个 steady sync 会在同一个 receive session 末尾追加一次 full HF export：

```text
sparse apply
  → full authoritative weights 再发送一次
  → rollout loader 记录真实 copy_ destinations 的旧值
  → 走真实 model.load_weights() 链路
  → 比较 load 前后是否 bit-identical
  → 最终 flush 汇总整次 sweep 的 mismatch 并抛错
```

判据是：如果 rollout 已经正确积累所有 delta，再重放 trainer 当前的全量权重应该是幂等操作。它能检测 sparse 路径相对同一条 full-load 链路的偏差，覆盖 converter、slot、网络和真实 SGLang loader 实际触及的 `copy_` destinations，是比 checksum 更强的端到端相对状态 oracle。

但 verify 不能证明 full exporter/loader 的绝对完备性或绝对语义正确：如果 full export 与 delta path 一致地漏掉同一个参数，或者同一条 loader 链路始终把它映射到错误 destination，幂等 sweep 仍可能通过。

## 五、稳态导出契约：只允许最终 HF 坐标穿过边界

[#7144](https://github.com/verl-project/verl/pull/7144) 是整条演进中最重要的抽象升级。它把权重命名、布局转换、diff 和 snapshot 全部放到 training backend 一侧，checkpoint engine 只接收一种规范化 entry：

```text
(
  slots,         # [(hf_name, hf_shape), ...]
  dtype_str,
  counts[K],     # 当前 rank 对每个 slot 的 changed-element 数
  hf_idx,        # 已在最终 HF slot 内的坐标
  hf_val,        # replacement values
  gather_group,
)
```

这里存在三套坐标空间：

```text
1. trainer local shard coordinates
        │  placement translation
        ▼
2. logical/global trainer tensor coordinates
        │  backend-owned name/layout conversion
        ▼
3. 最终 HF tensor slot coordinates
        │  checkpoint engine 只从这里开始工作
        ▼
4. SGLang internal TP/fused layout
        由 SGLang 自己的 model.load_weights() 处理
```

Checkpoint engine 不应该理解 QKV fusion、expert id、Megatron TP dim 或 Mamba mapping。它只需要知道：哪些 rank 的最终 HF pieces 要在什么 group 中合并。

### 1. Slot table 的三个不变量

一个训练参数可能映射到一个或多个 HF tensor。普通 dense 参数通常只有一个 slot；fused expert stack 可能映射到每个 expert 的 `gate_proj`、`up_proj` 和 `down_proj`。

协议依赖三个不变量：

1. **Alignment**：所有 rank 以完全相同的顺序枚举同一组 slot；无贡献 rank 也必须输出 zero counts。
2. **Disjointness**：不同 contributing rank 对同一个 slot 的坐标集合不重叠，因此 rank 0 可以直接 concat。
3. **Bounds**：当前位置使用 `int32`，单个最终 HF slot 必须少于 $2^{31}$ 个元素。

如果 rank 之间按各自 payload bytes 决定何时触发 collective，nnz 不同会导致调用序列分叉并死锁。因此 `_GatherQueue` 只按 entry 数量触发 batch；所有 rank 看到相同 counts matrix 后，再按 slot 边界确定性地切 sub-round，尽量把每个 rank 的 blob 控制在 byte budget 内。它不是严格上限：单个 slot 自身超限时不会继续切碎，`K=1` 时也没有 slot boundary 可切。

### 2. `BlockPlacement`：从 `Shard(0)` 推广到一般矩形分片

最初的实现主要依赖 `Shard(0)`，它在 flatten 后是连续区间，只需：

$$
global\_idx=local\_idx+flat\_offset
$$

但 `Shard(1)`、多维 mesh 或“手工 EP 切分 + DTensor 切分”得到的是高维矩形 block。`BlockPlacement` 用三个量描述它：

```text
local_shape
global_offset
full_shape
```

对 local flat index 做 mixed-radix 分解，得到本地多维坐标；加上每维 `global_offset`，再按 `full_shape` 的 stride 重组为 global flat index。`Shard(0)` 仍走单次加法 fast path。

当前实现明确拒绝：

- `_StridedShard`，因为本地 tensor 不是单个矩形 block；
- 多个 Shard dim 再混入 Replicate dim；
- 无法提供静态 slot table 的 VeOmni converter。

Converter 能否保持 NaN sentinel 并不会被运行时自动证明，而是当前设计的结构假设，需要 differential tests 和周期 verify 共同兜底。

## 六、三个训练 backend 如何实现同一契约

### 1. FSDP：identity mapping 是最简单的情况

FSDP exporter 位于 [`FSDPEngine.get_per_tensor_param_shard()`](https://github.com/verl-project/verl/blob/8bda42207cc08a947a49587d38315647740b9e14/verl/workers/engine/fsdp/transformer_impl.py#L861-L918)：

- 从 state dict 获取 DTensor 或普通 tensor；
- 执行 key normalization；
- floating local shard 转 BF16；
- 构造 `ShardSpec`；
- 假设 source name/shape 已经等于最终 HF name/shape；
- 只做 placement coordinate translation。

FSDP1 的 state-dict export 仍需要把整份本地模型 shard stage 到 GPU；FSDP2 的 state dict 只收集 DTensor reference，参数可以按需 lazy stage。后者正是 [#7005](https://github.com/verl-project/verl/pull/7005) 优化的路径。

FSDP identity exporter 的边界也很明确：如果 full export 需要额外的 unfuse、重排或一对多转换，steady path 不能假装它仍是 identity，必须像 VeOmni 一样提供 backend converter。

另外，代码注释将这条 shard export 限定为 “Non-LoRA base path only”，但当前没有像 Megatron 那样显式 assert。实践中应把 FSDP `delta_sharded + LoRA` 视为尚未完成 fail-closed 和端到端验证的组合。

### 2. VeOmni：手工 EP 与 DTensor 双重切分

VeOmni 的 grouped experts 会被切两次：

```text
global fused expert stack
  ├─ expert dim 先由 EP 手工切分
  └─ 每个 EP-local expert block 再由 FSDP/HSDP DTensor 切分
```

第一层切分不在 DTensor placement 中，单靠 `param.placements` 无法还原全局位置。[#7085](https://github.com/verl-project/verl/pull/7085) 因此在“所有 experts 的虚拟 full tensor”上构造显式 `BlockPlacement`：

```text
global expert offset
= ep_rank * n_local_experts + DTensor block offset
```

同时声明：

- `gather_group=trainer WORLD`；
- HSDP replica 只有 coordinate 0 贡献；
- `to_hf_chunk` 复用 VeOmni 自己的 MoE handler；
- `hf_slots` 枚举 fused stack 最终产生的逐 expert HF tensors。

稳态 converter 只为 touched expert rows 构造 NaN row，把变化值填进去后执行真实 handler。输出中的非 NaN survivor 就是最终 HF slot 中的变化位置。这要求 converter 是 dim-0 separable，并且能够保持 NaN；典型的 slice、rename、transpose 和 permutation 都满足。

这使下面的映射可以在 sender 侧完成：

```text
mlp.experts.gate_up_proj [E, 2I, H]
  → mlp.experts.{e}.gate_proj.weight
  → mlp.experts.{e}.up_proj.weight
```

### 3. Megatron-Bridge：不重写 mapping，而是探测真实 converter

Megatron 的难点不是一个简单 block offset，而是 TP/EP/ETP、QKV fusion、GQA、Mamba 和 PP ownership 共同决定的转换逻辑。重新手写一套 delta converter 很容易与 Megatron-Bridge 的 full export 漂移。

[#7181](https://github.com/verl-project/verl/pull/7181) 的选择是复用 Bridge 自己的 `get_conversion_tasks()` 和 `megatron_to_hf()`：

```text
local shard delta
  → scatter 到 local-shape NaN probe buffer
  → 执行 comm-stubbed Megatron-Bridge converter
  → 从输出中提取 non-NaN survivors
  → 最终 HF slot / index / value
```

所谓 comm-stubbed probe，是复制 mapping tree，并把 process group 替换成 `_ProbeGroup`：

- `size()` 和 `rank()` 保持真实 TP/EP/ETP 值，让 converter 的 shape/value math 走生产路径；
- gather helper 不做真实通信，而是把当前 rank shard 放在正确 rank 位置，其他 rank 用 pooled NaN block；
- 如果 converter 通过未 stub 的 API 尝试通信，立即抛错，而不是静默生成错误坐标。

这种方法适合 rearrange、concat、de-interleave 和保持 NaN 的逐元素变换；如果 converter 会把多个 shard 做算术 blend，NaN 可能污染有效 contribution，必须扩展协议或为该 mapping 单独处理。每次升级 Megatron-Bridge 后都应重跑 real converter 与 probe assembly 的 differential oracle。

### 4. PP/VPP：全局参数目录和 zero-count placeholder

[#7223](https://github.com/verl-project/verl/pull/7223) 利用 Bridge 已有的 global parameter directory 支持 PP/VPP：

- 参数名跨 PP ranks all-gather 后确定性排序；
- tied embedding 去重；
- 非本 stage 参数用 `param=None` placeholder 表示；
- placeholder 仍输出相同 directory row 和 zero counts，保持 collective lockstep；
- PP>1 时每个参数通过 WORLD group merge，保证 global actor rank 0 总在接收组内；
- DP/CP replica 通过 `contributes=False` 去重；
- uneven first/last pipeline stage split 由真实 ownership 自然表达。

EP rank 的本地 Megatron 参数名可能带不同 expert id，因此 slot table 不能只按参数名合并。当前实现按 directory row 做 rank-order union，并将结果一次性分发给所有 rank。

当前 backend 覆盖可概括为：

| Training backend | 当前覆盖 | 主要限制 |
|---|---|---|
| FSDP1/FSDP2 | DTensor/普通 tensor、一般矩形 `BlockPlacement`、identity HF mapping | 非 identity converter 需 backend 扩展；LoRA 未完整 fail-closed |
| VeOmni | FSDP2/HSDP + EP、fused expert stack | converter 需 dim-0 separable、静态 slot、保 NaN；LoRA 尚未完整支持 |
| Megatron-Bridge | TP、EP、ETP、PP/VPP、hybrid-Mamba | 不支持 deprecated vanilla bridge；显式拒绝 LoRA；mapping 需符合 probe 假设 |

## 七、Sparse gather、bucket 和 wire protocol

### 1. Batched variable-length gather

每个 rank 的 nnz 不同。[`gather_slot_entries_to_rank0()`](https://github.com/verl-project/verl/blob/8bda42207cc08a947a49587d38315647740b9e14/verl/checkpoint_engine/delta_sync/sparse_gather.py#L62-L156) 的流程是：

1. all-gather 每个 slot 的 counts，得到 `world × K` 矩阵；
2. 所有 rank 根据同一矩阵计算确定性的 sub-round cuts；
3. 每轮把本 rank 的 idx 和 val pad 到该轮最大长度；
4. 分别 gather positions 和 values；
5. group rank 0 按 `(rank, slot)` 切回，再按 slot concat。

默认 `batch_gather=32`，可以把几十个参数的 counts/idx/val 一起 gather，减少 per-parameter collective 和 host sync。

需要注意 padded gather 的 rank 0 内存模型。`max_round_bytes` 只是按 slot 边界尽量约束每个 rank 的 blob；单 slot 超限仍会越过 budget。它也不约束 rank 0 为整个 group 分配的 receive buffers 总和，极端不均衡的 nnz 分布仍可能放大 rank 0 峰值。64M entry slicing 发生在 gather 之后，所以同样不能限制 gather 阶段的峰值。

### 2. Streaming flush

Rank 0 将 gathered slot delta 按 bucket 组成 `DeltaFlush`：

```text
DeltaFlush
  ├─ params: [DeltaParam, ...]
  ├─ positions: packed int32 bytes
  ├─ values: replacement value tensor
  └─ checksum
```

每个 `DeltaParam` 记录：

- HF name、dtype、shape；
- 它在 positions blob 中的 byte range；
- 它在 values tensor 中的 element range。

`_FlushBucket` 使用 one-flush lookahead：只有看到下一包或执行最终收尾时，才能知道上一包是不是 `is_last`。这样 `verify_every` 可以把最后一包 sparse 标成非最后，再接 dense verify stream。

稳态单个 entry 最多包含 64M changed elements，主要是为了限制 receiver 把 `int32` positions 转成 `int64` 时的临时内存。这个 cap 不会把一个巨大参数的 full shape 拆小。

### 3. ZMQ 发送控制信息，NCCL 发送数据

每个 flush 的：

- manifest、`is_last`、shape、dtype、checksum 通过 ZMQ PUB/SUB 发送；
- positions 和 values 通过 Ray NCCL group 广播。

Actor rank 0 会先复制到 CuPy-owned staging buffer。原因是 Ray NCCL 在独立 CUDA stream 上 enqueue；如果直接广播一个很快被释放的 PyTorch view，allocator 可能在 kernel 真正读取前复用 storage。显式 staging 把 buffer 生命周期交给 CuPy，并在一次完整的同步发送结束后释放 pool。

当一个 steady sync 完全没有 changed elements 时，只发送 ZMQ terminal marker，不发 NCCL payload。

### 4. Bucket size 不是所有中间内存的硬上限

`update_weights_bucket_megabytes` 是重要调优项，但它不是严格的全局峰值上限：

- `_FlushBucket` 是先加入 piece，再判断是否 seal，单个 piece 可以越过 cap；
- dense seed 把一个完整 HF parameter 作为一个 piece，不会像全量 NCCL chunking 那样拆单参数；
- sparse gather 的 rank 0 padded receive allocation 是 group-wide 的；
- receiver 仍要为每个 touched parameter 建立 full-shape masked tensor。

所以更准确的峰值模型是：

```text
sender: 若干 gather buffers + 约两个 streaming flush + staging
receiver: 一个 wire flush + 最大 touched HF tensor 的 dense mask
          + positions int64 / torch.where 等临时量
```

它避免的是 full-model mirror，而不是把所有瞬时内存严格限制为 $O(nnz)$。

## 八、SGLang 如何原地应用 sparse patch

当前 `delta_sharded` 只支持 SGLang rollout。非 SGLang backend 会在 [`CheckpointEngineWorker`](https://github.com/verl-project/verl/blob/8bda42207cc08a947a49587d38315647740b9e14/verl/checkpoint_engine/base.py#L283-L320) 初始化时抛出 `NotImplementedError`。

SGLang server 启动时，verl 自动注册：

```text
verl.workers.rollout.sglang_rollout.delta_loader.apply_delta
```

每个 rollout checkpoint worker 接收完整 model-global flush，再通过 same-GPU CUDA IPC 将两到三个 sentinel tensor 交给对应 SGLang TP process：

```text
__delta_spec__
__positions__    # 只有 sparse flush 才有；dense seed/verify 不携带
__values__
```

[`delta_loader.apply_delta()`](https://github.com/verl-project/verl/blob/8bda42207cc08a947a49587d38315647740b9e14/verl/workers/rollout/sglang_rollout/delta_loader.py#L61-L228) 执行：

1. 重新计算 positions + values checksum；
2. 对每个参数分配 full-shape NaN tensor；
3. 用 positions 把 replacement values 写入对应位置；
4. 临时把 `Tensor.copy_` 改写为：

```python
copy_(dst, torch.where(torch.isnan(src), dst, src))
```

5. 调用 SGLang 原生 `model.load_weights()`。

这个设计很巧妙：verl 不必理解 SGLang 内部 TP slice、fused parameter 或模型专用 loader。最终 HF 坐标中的 sparse patch 先被还原成 NaN-masked HF tensor，再让 SGLang 自己完成最后一跳映射。整个过程无需 fork SGLang。

但 receiver apply 并不是严格的 $O(nnz)$：每个 touched HF parameter 都会创建 full-shape tensor，`torch.where` 也扫描完整 destination。稀疏性主要节省 trainer gather、wire 和 full-model staging，不保证 rollout install 成本与 nnz 成正比。

## 九、正确性工程：需要分清四层证据

“没有报错”和“rollout 权重等于 trainer 权重”之间还有很长距离。当前实现提供了四层不同强度的证据：

| 层次 | 机制 | 能证明什么 | 不能证明什么 |
|---|---|---|---|
| 传输完整性 | 每个 flush 的 positions+values checksum | sender encode 到 SGLang loader 之间 payload 没有被明显破坏 | converter、slot、name mapping 是否正确 |
| 编码正确性 | bitwise diff、round-trip、masked apply tests | 稀疏 patch 可以无损重建同 dtype 目标 | 多 backend 的复杂布局是否映射正确 |
| 转换正确性 | BlockPlacement tests、Megatron real-vs-probe differential oracle | local shard 到最终 HF 坐标的转换与 full path 一致 | 运行中是否漏掉某一轮或某个参数 |
| 状态正确性 | `verify_every=K` 全量幂等 sweep | 本次 full export 经真实 loader 实际触及的 destinations，重放是否幂等 | full exporter/loader 是否一致地遗漏或错误映射；也不提供事务回滚 |

行为层的 reward、KL、log-prob correlation 或 W&B 曲线只能作为统计证据。不同 sampling run 不会逐 token bitwise 相同，因此“曲线在噪声范围内重合”不等于状态级证明。

### 1. Fail-loud 边界

当前代码会明确拒绝或报错的情况包括：

- rollout 不是 SGLang；
- encoding 不是 `indices`；
- 没有 seed snapshot 或 shard size 改变；
- seed full export 出现重复 HF name；
- 单个 HF slot 超过 `int32` coordinate range；
- wire checksum mismatch；
- `_StridedShard` 或不支持的 multi-Shard + Replicate；
- FSDP identity path 收到 converter spec；
- VeOmni converter 没有静态 slots；
- Megatron 使用 deprecated vanilla bridge、LoRA 或未 stub 通信；
- `verify_every` 发现 rollout state mismatch。

这种 fail-loud 风格很重要，因为权重同步最危险的失败模式不是 crash，而是训练继续运行、rollout 却在使用旧权重或错误权重。

### 2. Checksum 不是状态一致性证明

Checksum 只覆盖单个 flush 的 positions 和 values。它不能发现：

- backend 漏导出某个参数；
- converter 把变化映射到了错误 slot；
- 两个 rank 的坐标发生重叠；
- SGLang loader 把正确 HF tensor 写到了错误 internal destination；
- 某次 delta 因控制流错误完全没有发送。

这些问题需要 differential tests 和 `verify_every`；即便如此，若 sparse 与 full 两条路径一致地遗漏同一状态，仍需要 tensor-set equivalence 等额外 oracle。

### 3. NaN 是协议 sentinel

VeOmni/Megatron probe 和 SGLang masked apply 都把 NaN 当作“该位置无 contribution/不覆盖”。协议真正要求的是 authoritative replacement 不能把 NaN 当作普通新值。

如果新的权重值本身变成 NaN，它在 sparse apply 中会与 untouched sentinel 混淆，可能保留 rollout 旧值。`Inf` 不与 `torch.isnan()` sentinel 冲突，技术上能够传输和 apply，但通常也意味着训练异常。生产监控仍应检查 non-finite 参数；周期 verify 可以暴露其中一部分 trainer/rollout divergence。

### 4. 成功路径可观察原子，失败路径并非事务

Manager 会先 abort generation，因此成功时不会有请求看到中间 flush。但协议没有 shadow model、commit pointer 或 rollback log：

- receiver 是逐 flush 修改 live weights；
- sender 的 CPU snapshot 也在遍历 exporter 时逐参数刷新；
- checksum 或 apply 在中途失败时，两侧都可能处于部分推进状态。

失败要分成两类看：

- sparse checksum/apply 中途失败时，live weights 和 sender snapshots 都可能只推进了一部分；重试同一个 delta 没有一致 base，不能恢复事务；
- verify mismatch 只在完整 dense sweep 的最终 flush 汇总后抛出，此时本次 full loader 实际触及的 destinations 已被 authoritative values 重写；但 manager 没有异常恢复 `finally`，KV cache 和 generation 仍可能停在暂停状态，而且 mismatch 本身说明契约已被破坏。

两类异常都应按 fail-stop 处理。当前没有公开的 reset/reseed hook；安全恢复需要重建 actor checkpoint-engine/worker 和对应 rollout，使新 engine 的 `_shard_seeded=False` 自然触发 full seed。只重建 rollout replica 而保留原 actor snapshot 不够。

### 5. Snapshot 与 topology 必须共享同一个 base

`_shard_seeded` 和 CPU snapshots 只存在 actor worker 内存中，不进入训练 checkpoint。完整 job 重启会自然重新 seed，这是安全的；但如果 actor 仍存活，只新增一个从旧 checkpoint 或 dummy weights 启动的 rollout replica，sender 仍可能发送相对上一版 base 的 delta。

当前 `add_replicas/remove_replicas` 没有为 delta 自动触发 reseed 的逻辑，Ray NCCL group 的 world size 也默认固定。因此不要把现有实现理解为已经支持无缝 elastic scale-up 或单独 rollout 热重启；若要保留 actor 进程，必须先实现显式 reset/reseed handshake。

---

## 十、性能数据应该怎样读

### 1. FSDP：用已经优化过的全量 baseline 比较

[#7144](https://github.com/verl-project/verl/pull/7144) 报告的 H100、GSM8K GRPO、V1 `separate_async`、SGLang 数据如下。这里的 `nccl` 已包含 [#7005](https://github.com/verl-project/verl/pull/7005) 的 FSDP2 staging 修复：

| 模型与拓扑 | `delta_sharded` | 全量 `nccl` | 报告加速 |
|---|---:|---:|---:|
| Qwen2.5-7B，1+1 nodes | 3.9～4.9 s | 5.5～6.0 s | 约 1.3× |
| Qwen2.5-32B，2+2 nodes，offload on | 11.2～11.9 s | 17.7～18.1 s | 1.55× |
| Qwen2.5-32B，2+2 nodes，offload off | 6.2 s | 14.2 s | 2.3× |
| Qwen2.5-72B，4+4 nodes，gen TP8，offload off | 12.0～13.0 s | 28.5～29.1 s | 2.3× |

这组结果说明：即使 full NCCL 已经移除了明显的 staging 浪费，shard-local sparse gather 仍有收益；但 7B 快互联场景的提升远小于大模型。

### 2. VeOmni EP：MoE 稀疏性会放大收益

[#7085](https://github.com/verl-project/verl/pull/7085) 报告：

| 模型与拓扑 | `delta_sharded` | 全量 `nccl` | 说明 |
|---|---:|---:|---|
| Qwen3-30B-A3B，VeOmni EP8，2×8 GPU disaggregated | 7.1 s median | 32.2 s median | 4.5×（50-step median） |
| Qwen3-235B-A22B，EP8×FSDP8，8+2 nodes，gen TP16 | 公开逐步数据约 11.4～14.9 s | 246～266 s | 作者报告约 21×；seed 仍为 full export |

[30B 的三步原始数据](https://github.com/verl-project/verl/pull/7085#issuecomment-4998563485)是 delta 5.8/7.4/7.9 s、full 32.2/31.9/33.2 s；表中采用作者后续公开的 [50-step median](https://github.com/verl-project/verl/pull/7085#issuecomment-5029383288)，避免混用统计口径。[235B 表](https://github.com/verl-project/verl/pull/7085#issuecomment-5027537258)则采用作者公开的逐步原始数据：delta 14.9/11.8/11.4 s，对 full 255.9/246.5/266.1 s，逐步约为 17.2×/20.9×/23.3×，所以“约 21×”成立。

需要特别指出，[#7085](https://github.com/verl-project/verl/pull/7085) PR body 的另一个版本写成 delta 8.8～9.0 s 对 full 246～266 s，却仍标注约 21×，两者按算术并不一致。本文采用的是带逐步明细的原始评论，而不是这组互相矛盾的摘要数字。235B headline 也不能与 FSDP 表直接横比：它使用不同 training backend、模型结构、并行拓扑和 changed ratio。

### 3. Megatron：PP 会把 gather group 的代价重新带回来

[#7181](https://github.com/verl-project/verl/pull/7181) 在 235B TP4×EP16×PP1 上报告 steady update 约 23.6～23.7 s，而相同模型与拓扑下的全量 NCCL 约 235.0～237.6 s，按表中时间计算约为 9.9～10.1×，整体接近一个数量级改进。PR body 同时标注了 8.2～9.7×，这个倍率范围与它列出的时间不能严格对应，因此本文只把时间值作为主口径。

[#7223](https://github.com/verl-project/verl/pull/7223) 加入 PP/VPP 后，在 235B TP4×PP8×EP4 的 PR 测试拓扑上报告：

| `delta_sharded` | 全量 `nccl` | 报告加速 |
|---:|---:|---:|
| 97.5～111.1 s | 220.5～236.8 s | 2.1～2.4× |

为什么 PP8 的收益小了？因为当前 PP>1 为保证 actor global rank 0 能收到任意 stage 的参数，所有 directory row 都通过 WORLD group merge；placeholder 和更大的 collective group 带来了固定成本。这是用通用性换来的实现简洁，未来仍有 owner subgroup + rank0 relay 等优化空间。

### 4. 同步阶段加速不等于整轮训练等比例加速

[#6974 的 50-step 7B 结果](https://github.com/verl-project/verl/pull/6974#issuecomment-4922885197)中，权重同步从 2.64 s 降到 1.73 s，延迟降低 34.5%，等价于 1.53× 加速；whole-step 从 8.35 s 降到 7.42 s，延迟只降低 11.1%，等价于 1.125× 加速。

如果 rollout generation、reward 或 optimizer step 才是主瓶颈，权重同步即使快 10×，端到端收益仍受 Amdahl 定律限制。评估时至少要同时看：

- `timing_s/update_weights` 或 `timing_s/param_sync`；
- 整个 train step wall time；
- rollout pause window；
- changed ratio 和 payload；
- seed、steady、verify 三类同步分别花多少时间。

## 十一、与全量权重同步优化是什么关系

`delta_sharded` 不是对所有路径的替代。最近半年的几项全量同步优化仍然重要，而且为 delta 提供了更公平的 baseline。

### 1. 超大单参数 chunking（#6091）

[#6091](https://github.com/verl-project/verl/pull/6091) 面向 Transformers 5 的 fused MoE `gate_up_proj`：单个参数可能达到数 GB。过去 NCCL/NIXL bucket 和 CUDA IPC buffer 必须大于最大参数，导致 colocated 路径约 `2 × max_weight_size`、fully async 路径约 `3 × max_weight_size` 的额外峰值。

#6091 允许：

- IPC buffer 小于最大参数时绕过固定 buffer；
- NCCL/NIXL sender 将大参数切 chunk，receiver 再合并。

这是全量路径的内存工程，不是 delta 协议的一部分。当前 delta dense seed 并没有完全复用同样的单参数 chunking，因此超大 HF tensor 的 seed 峰值仍需单独评估。

### 2. 跳过 FSDP2 whole-shard staging（#7005）

FSDP2 `state_dict()` 只收集 DTensor reference，原实现却仍在导出前把整个 local shard 搬到 GPU，导出后再 offload。[#7005](https://github.com/verl-project/verl/pull/7005) 移除这次纯开销：

| 指标 | 修复前 | 修复后 |
|---|---:|---:|
| 7B export/generator creation | 3.3～4.5 s | 0.07 s |
| 每步 full NCCL sync | 6.65～6.93 s | 3.36～3.49 s |

这也提醒我们：不要把所有收益都归因于网络。一次看似通信优化的问题，可能主要耗在参数 staging。

### 3. 避免 SGLang bucket 冗余 clone（#6738）

对已经拥有紧凑连续 storage 的大权重再次 `clone()` 会瞬时翻倍内存，尤其容易在 fused MoE weight 上 OOM。[#6738](https://github.com/verl-project/verl/pull/6738) 只对 non-contiguous tensor 或 larger-storage view 做 compact clone。

这些优化与 delta 是互补关系：当 workload 不够稀疏、rollout backend 不是 SGLang、使用 LoRA/量化路径，或需要可靠 fallback 时，成熟的 full sync 仍是基线方案。

## 十二、如何启用、观测和调优

### 1. 最小配置

当前典型配置是 disaggregated trainer/rollout + SGLang：

```bash
actor_rollout_ref.hybrid_engine=False \
actor_rollout_ref.rollout.name=sglang \
actor_rollout_ref.rollout.checkpoint_engine.backend=delta_sharded \
+actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.delta_sharded.encoding=indices
```

可选调优项：

```bash
actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=2048 \
+actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.delta_sharded.batch_gather=32 \
+actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.delta_sharded.verify_every=10
```

当前代码只接受 `encoding=indices`。仓库旧示例注释曾写还支持 `deltas`，但 [`DeltaShardedCheckpointEngine.__init__()`](https://github.com/verl-project/verl/blob/8bda42207cc08a947a49587d38315647740b9e14/verl/checkpoint_engine/delta_checkpoint_engine.py#L398-L418) 会对其他值直接 assert。

### 2. 四个内建指标

Sender 在 seed 和 steady sync 中返回同一组指标；seed 会把 `changed_ratio` 记为 1.0：

```text
checkpoint_engine/changed_ratio
checkpoint_engine/changed_elems
checkpoint_engine/payload_mbytes
checkpoint_engine/flushes
```

其中 `changed_ratio` 是 changed elements / total elements。只有在定宽、同 dtype 场景下，它才近似等价于 raw value bytes ratio；真实 steady wire 还包含每个 changed element 的 4-byte index。周期 verify 追加的 full payload 不计入该次 steady sync 返回的 `payload_mbytes/flushes`。

建议同时监控：

- seed/steady/verify 分段耗时；
- trainer/rollout log-prob diff 与 Pearson correlation；
- rollout cache release/resume 时间；
- actor rank 0 和 rollout worker 的峰值显存；
- node pinned host memory；
- checksum/verify mismatch 和 non-finite parameter 数量。

### 3. 参数如何调

`update_weights_bucket_megabytes`：

- 大一些可以减少 flush 和 NCCL launch；
- 太大会增加 sender staging、receiver wire buffer 和 CuPy pool 峰值；
- 它不是单参数、padded gather 或 full-shape decode 的硬上限。

`batch_gather`：

- 大一些可减少 collective 次数和 host sync；
- 过大可能增加单轮 padded gather 的临时内存和首包等待时间；
- 默认 32 是工程折中，不应脱离模型 tensor 数和 topology 盲目放大。

`verify_every`：

- `0` 表示关闭；
- 新模型、新 Bridge 版本、新并行布局验证时可设为 1 或较小值；
- 稳定运行中可按能接受的 full-export 成本周期性抽检；
- verify 的 full payload 当前不计入 steady `payload_mbytes/flushes`，分析监控时要单独考虑。

### 4. 上线前检查表

在把 full NCCL 切到 `delta_sharded` 前，至少确认：

- rollout 是 SGLang，且不是当前未支持的 SGLang PD delta 组合；
- training backend 是 FSDP、VeOmni 或新 Megatron-Bridge 的已覆盖布局；
- 不依赖尚未验证的 LoRA、QAT、FP8/MXFP4 delta；
- 每个最终 HF slot 小于 $2^{31}$ 个元素；
- 参数目录和 shard geometry 在 actor 生命周期内固定；
- rollout replica 与 actor snapshot 拥有同一个 full-seed base；
- changed ratio 长期显著低于 BF16 的约 33% wire break-even；
- 用 `verify_every=1` 或较小 K 跑过真实多步 E2E；
- 对 worker crash、rollout restart、scale-up 做过恢复演练：重建 actor checkpoint-engine/worker 以触发 full seed，或先实现显式 reset/reseed hook；
- 保留通过重启或重建 workers，以 `checkpoint_engine.backend=nccl` 回退的方案；当前不能只动态改配置热切 backend。

---

## 十三、未解决问题与演进方向

前文的使用边界可以进一步归纳为几组待补齐的系统能力：

| 未解决问题 | 当前影响 | 优先演进方向 |
|---|---|---|
| Sparse apply 与 transport 覆盖面 | rollout 只支持 SGLang，传输只在 CUDA/NCCL 路径；vLLM、TRT-LLM、HCCL/NPU 不可直接复用 | 为 inference backend 定义统一 indexed-update plugin，并把 transport 与 patch contract 解耦 |
| 高 changed ratio 与 index 开销 | 没有 density-aware fallback；当 $r(s+4)\ge s$ 时，BF16 sparse payload 已不如 dense values | 根据预测的端到端成本自动切 dense/full wire；探索 bitmap、run-length 或分层 encoding |
| Bucket 只是软上限 | dense seed 的超大单参数、单-slot sparse gather、receiver full-shape mask 都可能越过 bucket | 对 seed 做参数内 chunking，使 wire/staging 更接近 bucket 上限；真正的全局硬上限还需要 chunk-native gather 和 indexed apply |
| 没有版本与事务协议 | 逐 flush 修改 live weights，失败不能 rollback；新 replica、热重启和 elastic scale-up 没有共同 base | 增加 base/version、ACK、commit/abort、reset/reseed handshake；必要时配合 shadow state 或可回滚 journal |
| 量化 delta 未完成 | vLLM FP8/MXFP4 refit、TRT-LLM FP8 refit 仍是全量或专用 reload，不能等同于 quantized sparse wire | 采用 quantize-then-diff，并让 receiver 同步更新 scale、派生权重和 kernel state |
| LoRA 与新 mapping 仍靠人工守约 | Megatron 显式拒绝 LoRA，VeOmni 有 TODO，FSDP 缺少完整 guard；新增 converter 可能破坏 slot/NaN 假设 | 将 slot overlap、seed/steady tensor-set equivalence、real-vs-probe differential oracle 变成 fail-closed gate |
| NaN sentinel 与 dense receiver work | NaN replacement 会与 untouched 混淆；每个 touched parameter 仍要 full-shape densify 和 `torch.where` | 加入 NaN/non-finite guard，并让 inference backend 原生接受 indexed update，避免 NaN mask |

---

## 十四、这套设计真正值得复用的地方

verl 的增量权重同步可以总结为四个设计判断：

1. **差分必须发生在 full gather 之前**。否则只优化 wire，不解决 materialize、gather 和 rank 0 full snapshot。
2. **跨层协议应该使用 final consumer coordinates**。Checkpoint engine 不应理解训练 backend 的模型布局。
3. **正确性要分层验证**。Checksum、bitwise round-trip、converter differential test 和 full idempotence sweep 各自覆盖不同风险。
4. **数据面优化不等于切换协议**。Sparse transport 可以缩短 pause，但当前请求一致性仍来自 abort/quiesce，而不是 lock-free atomic swap。

> `delta_sharded` 最有价值的地方不是某个 21× headline，而是它把“谁理解权重布局、谁负责稀疏通信、谁负责最终安装”分成了清晰的契约：training backend 产出最终 HF patch，checkpoint engine 只搬 patch，inference backend 用自己的 loader 完成最后一跳。

---

## 代码索引

- Checkpoint engine manager 与 control plane：[`verl/checkpoint_engine/base.py`](https://github.com/verl-project/verl/blob/8bda42207cc08a947a49587d38315647740b9e14/verl/checkpoint_engine/base.py#L361-L538)
- `delta_sharded` seed/steady/wire：[`verl/checkpoint_engine/delta_checkpoint_engine.py`](https://github.com/verl-project/verl/blob/8bda42207cc08a947a49587d38315647740b9e14/verl/checkpoint_engine/delta_checkpoint_engine.py#L218-L666)
- Batched sparse gather：[`verl/checkpoint_engine/delta_sync/sparse_gather.py`](https://github.com/verl-project/verl/blob/8bda42207cc08a947a49587d38315647740b9e14/verl/checkpoint_engine/delta_sync/sparse_gather.py#L41-L156)
- Wire manifest 与 checksum：[`verl/checkpoint_engine/delta_sync/encode.py`](https://github.com/verl-project/verl/blob/8bda42207cc08a947a49587d38315647740b9e14/verl/checkpoint_engine/delta_sync/encode.py#L14-L90)
- Backend delta contract：[`verl/workers/engine/base.py`](https://github.com/verl-project/verl/blob/8bda42207cc08a947a49587d38315647740b9e14/verl/workers/engine/base.py#L161-L227)
- 通用 snapshot/diff/export：[`verl/workers/engine/utils.py`](https://github.com/verl-project/verl/blob/8bda42207cc08a947a49587d38315647740b9e14/verl/workers/engine/utils.py#L162-L242)
- `ShardSpec` 与 `BlockPlacement`：[`verl/workers/engine/spec.py`](https://github.com/verl-project/verl/blob/8bda42207cc08a947a49587d38315647740b9e14/verl/workers/engine/spec.py#L55-L262)
- FSDP exporter：[`verl/workers/engine/fsdp/transformer_impl.py`](https://github.com/verl-project/verl/blob/8bda42207cc08a947a49587d38315647740b9e14/verl/workers/engine/fsdp/transformer_impl.py#L861-L918)
- VeOmni EP exporter：[`verl/workers/engine/veomni/utils.py`](https://github.com/verl-project/verl/blob/8bda42207cc08a947a49587d38315647740b9e14/verl/workers/engine/veomni/utils.py#L142-L381)
- Megatron-Bridge probe/export：[`verl/workers/engine/megatron/delta_export.py`](https://github.com/verl-project/verl/blob/8bda42207cc08a947a49587d38315647740b9e14/verl/workers/engine/megatron/delta_export.py#L14-L436)
- SGLang sparse loader：[`verl/workers/rollout/sglang_rollout/delta_loader.py`](https://github.com/verl-project/verl/blob/8bda42207cc08a947a49587d38315647740b9e14/verl/workers/rollout/sglang_rollout/delta_loader.py#L14-L228)
- CPU contract tests：[`tests/checkpoint_engine/test_sharded_delta.py`](https://github.com/verl-project/verl/blob/8bda42207cc08a947a49587d38315647740b9e14/tests/checkpoint_engine/test_sharded_delta.py)、[`test_block_placement.py`](https://github.com/verl-project/verl/blob/8bda42207cc08a947a49587d38315647740b9e14/tests/checkpoint_engine/test_block_placement.py)、[`test_sglang_loader.py`](https://github.com/verl-project/verl/blob/8bda42207cc08a947a49587d38315647740b9e14/tests/checkpoint_engine/test_sglang_loader.py)
- Distributed oracles：[`test_sharded_delta_gather.py`](https://github.com/verl-project/verl/blob/8bda42207cc08a947a49587d38315647740b9e14/tests/special_distributed/test_sharded_delta_gather.py)、[`test_sglang_delta_loop.py`](https://github.com/verl-project/verl/blob/8bda42207cc08a947a49587d38315647740b9e14/tests/special_distributed/test_sglang_delta_loop.py)、[`test_mcore_probe_differential.py`](https://github.com/verl-project/verl/blob/8bda42207cc08a947a49587d38315647740b9e14/tests/special_distributed/test_mcore_probe_differential.py)

## 参考资料

- [#6974：sharded delta weight sync over NCCL](https://github.com/verl-project/verl/pull/6974)
- [#7144：BlockPlacement 与 backend-owned HF export](https://github.com/verl-project/verl/pull/7144)
- [#7085：VeOmni FSDP2+EP delta export](https://github.com/verl-project/verl/pull/7085)
- [#7181：Megatron-Bridge TP/EP/ETP 与 hybrid-Mamba](https://github.com/verl-project/verl/pull/7181)
- [#7223：Megatron PP/VPP steady delta export](https://github.com/verl-project/verl/pull/7223)
- [#6091：NCCL/NIXL large-weight chunking](https://github.com/verl-project/verl/pull/6091)
- [#7005：FSDP2 export staging 优化](https://github.com/verl-project/verl/pull/7005)
- [#6738：SGLang bucket clone OOM 修复](https://github.com/verl-project/verl/pull/6738)
- [verl Delta Weight Sync 文档](https://github.com/verl-project/verl/blob/8bda42207cc08a947a49587d38315647740b9e14/docs/advance/delta_weight_sync.md)
