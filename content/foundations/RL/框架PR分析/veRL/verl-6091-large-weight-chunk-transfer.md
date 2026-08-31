# verl #6091: 让大 MoE 权重同步不再被通信 Bucket 绑架

> PR: [verl-project/verl#6091](https://github.com/verl-project/verl/pull/6091)  
> 标题: `[rollout,vllm] feat: split large weight into chunks in NCCL/NIXL checkpoint engine`  
> 状态: merged，2026-04-27  
> 核心问题: 大 MoE 模型里，单个权重 tensor 可能比 weight sync bucket 还大，迫使系统分配异常大的通信 buffer，显著抬高 rollout 权重同步的显存峰值。

---

## 1. 为什么这条 PR 重要？

在 LLM 强化学习训练里，系统不只是训练模型，还要不断把最新模型搬到 rollout engine。

以 PPO、GRPO、DAPO 这类训练为例，训练系统通常至少有两类 worker：

| 角色 | 负责内容 |
|---|---|
| Trainer worker | 反向传播、优化器更新、维护最新 policy 权重 |
| Rollout worker | 用当前 policy 做推理，生成 response / trajectory |

一次训练循环大致是：

```text
rollout worker 生成样本
  ↓
reward / advantage 计算
  ↓
trainer worker 更新 policy
  ↓
rollout worker 同步新权重
  ↓
继续生成下一批样本
```

普通推理服务加载权重通常是启动成本，加载一次后长期 serving。RL 训练不同：policy 持续更新，rollout engine 也要周期性 refit 最新权重。

所以 weight synchronization 不是边角逻辑，而是 RL 系统的常规路径。它决定：

- 大模型 rollout 是否会在更新权重时 OOM。
- fully async / disaggregated rollout 能否扩到更大模型。
- 每轮训练是否被权重搬运拖慢。
- 通信 bucket 能否按效率配置，而不是被最大参数尺寸强行决定。

#6091 就是围绕这个问题做的一个底层修复。

---

## 2. 旧设计：Bucketed Weight Transfer

最直观的权重同步方式是一个 tensor 一个 tensor 地发：

```text
send layer0.weight
send layer0.bias
send layer1.weight
...
```

但一个大模型可能有几百到几千个参数 tensor。如果每个 tensor 都独立发送，会带来大量控制面开销：

- metadata 消息多。
- ACK 次数多。
- CUDA IPC handle 数量多。
- receiver 每次只处理一个小 tensor，调度效率差。
- 小传输太多，带宽利用率不好。

因此常见系统会引入固定大小的通信 buffer，也就是 bucket。

发送侧把多个小 tensor 的 bytes 拼进 bucket：

```text
bucket buffer:
  [layer0.weight bytes][layer0.bias bytes][layer1.weight bytes]...
```

然后通过 metadata 描述每个 tensor 在 bucket 里的位置：

```text
{
  "layer0.weight": {
    shape,
    dtype,
    offset
  },
  "layer0.bias": {
    shape,
    dtype,
    offset
  }
}
```

接收侧根据 `offset + shape + dtype` 从 bucket 里切出 tensor view，再把这些权重交给 rollout engine。

这个设计对大量小 tensor 很合理。它把很多碎片化传输聚合成少量 bucket 传输，减少控制面开销，也更容易吃满通信带宽。

但它隐含了一个前提：

> 每个单独权重 tensor 都必须放得进 bucket。

过去这个前提通常成立。大 MoE 模型开始让它失效。

---

## 3. 数据面和控制面：ZMQ、CUDA IPC、Shared Memory

verl 的 vLLM colocated weight transfer 使用的是一种典型的控制面 / 数据面分离设计：

```text
ZMQ metadata + CUDA IPC / shared memory
```

这里的关键是：ZMQ 不负责传大权重本身。

可以把这套机制理解成：

```text
ZMQ: 递控制消息
CUDA IPC / shared memory: 共享真正的数据
```

ZMQ 发送的是小消息：

- 当前 bucket 里有哪些 tensor。
- 每个 tensor 的 `name` / `shape` / `dtype`。
- 每个 tensor 在 bucket 里的 `offset`。
- 当前 bucket 是否是最后一个。
- receiver 处理完成后的 ACK。
- 如果走 direct IPC，则发送某个 tensor 的 IPC handle。

真正 GB 级别的 tensor 数据不应该通过 ZMQ 直接序列化发送。那会引入额外 CPU copy、Python object 序列化和更高内存峰值。

### 3.1 CUDA IPC

CUDA IPC 允许同一台机器上的两个进程共享一块 GPU 显存。

发送进程里有一个 GPU tensor：

```text
trainer process:
  tensor on cuda:0
```

它可以导出一个很小的 IPC handle。接收进程拿到这个 handle 后，可以把同一块 GPU memory 映射到自己的进程里：

```text
rollout process:
  tensor view pointing to the same GPU memory
```

也就是说，CUDA IPC 传的不是 tensor 数据本身，而是“如何访问这块 GPU 显存”的 handle。

一个容易混淆的点是：

> CUDA IPC 本身不要求额外 bucket。

CUDA IPC 可以直接共享任意一块 GPU tensor。旧实现里之所以有 bucket，不是 CUDA IPC 的要求，而是 verl 为了批量传输小 tensor 设计的 batching 协议。

旧路径共享的是 bucket buffer：

```text
trainer:
  bucket_buffer = torch.empty(B, dtype=uint8, device="cuda")
  export bucket_buffer handle

每一轮:
  copy many small weights into bucket_buffer
  send metadata through ZMQ
```

所以旧路径中：

```text
CUDA IPC 共享的是 bucket buffer，不是每个原始 weight tensor。
```

### 3.2 Shared Memory

shared memory 是 CPU 侧共享内存，适用于当前代码判断 IPC 不可用的设备或场景。

流程类似：

```text
sender 创建 CPU shared memory
sender 把 tensor bytes 写进去
receiver 映射同一块 shared memory
receiver 再把数据拷到目标 device
```

它通常比 CUDA IPC 慢，因为数据会经过 CPU 侧共享内存，并且接收侧常常还要 `.to(device)`。在当前 vLLM rollout weight transfer 里，`use_shm = not is_support_ipc()`；CUDA 会直接认为支持 IPC，NPU 则会根据 Ascend HDK / CANN 版本判断是否支持 IPC。不支持 IPC 时才 fallback 到 shared memory。

#6091 只给超大 tensor 的 direct path 加了 CUDA IPC 支持。`use_shm=True` 的 shared-memory 路径仍然要求单个 tensor 能放进 bucket。

---

## 4. 问题：一个 Tensor 比 Bucket 还大

旧 bucketed transfer 最致命的假设是：

```text
weight.nbytes <= bucket_size
```

如果某个 tensor 大于 bucket，旧实现只能报错，或者要求用户把 `rollout.checkpoint_engine.update_weights_bucket_megabytes` 调大。

这在大 MoE 模型上会变成真实问题。新的 Transformers / MoE 权重布局里，一些 projection 会被 fuse 成很大的 tensor，比如：

```text
gate_up_proj: [num_experts, 2 * intermediate_dim, hidden_dim]
```

PR 描述中提到，Qwen/Qwen3.5-397B-A17B 的 `gate_up_proj` 在 bf16 下可达 8GB 级别。

如果 bucket 必须大于 8GB，那么每次 weight sync 都要准备 8GB 级别的通信 buffer。有些路径还不是一个 buffer，而是 double buffer，再叠加 CUDA IPC bucket。

这就导致一个结构性浪费：

> 单个异常大的 tensor，把所有通信 buffer 的尺寸都绑架了。

#6091 的核心思路是把“平均情况”和“长尾情况”拆开处理：

```text
小 tensor:
  继续走 bucketed transfer

大 tensor:
  NCCL/NIXL 路径切 chunk
  vLLM colocated 路径 direct IPC
```

---

## 5. 显存峰值怎么估算？

这里讨论的是 weight sync 期间的临时额外显存，不是训练总显存。训练总显存还包括模型参数、optimizer state、gradients、activations、KV cache、vLLM engine 内部 cache 等。

先定义：

```text
W = 最大单个权重 tensor 的大小
B = update_weights_bucket_megabytes 对应的 bucket 大小
```

PR 前有一个硬约束：

```text
B >= W
```

### 5.1 vLLM Colocated Bucket Path

旧路径里，发送侧已经有完整的 source weight：

```text
source weight tensor: W
```

为了通过 bucketed CUDA IPC 发给 receiver，还要把它 copy 到 CUDA IPC bucket：

```text
communication bucket: B
```

因为旧逻辑要求 `B >= W`，所以如果最大 tensor 是 8GB，bucket 也至少要 8GB。

粗略峰值是：

```text
source weight + bucket
= W + B
≈ W + W
= 2W
```

PR 后，如果 `W > B`，这个大 tensor 不再 copy 到 bucket，而是直接发送它自己的 CUDA IPC handle。

此时峰值变成：

```text
source weight + small-weight bucket
= W + B
```

如果 `W = 8GB`、`B = 512MB`：

```text
PR 前: 约 16GB
PR 后: 约 8.5GB
```

注意，这不是消灭了 8GB tensor，而是消灭了“为了传这个 8GB tensor，再额外准备一个 8GB bucket”的倍增项。

### 5.2 NCCL / NIXL Checkpoint Engine

NCCL/NIXL 路径更重，因为 checkpoint engine 本身有 double buffer：

```text
send_buf: B
recv_buf: B
```

后面 rollout/vLLM 更新权重时，还可能经过 CUDA IPC buffer 或类似 update bucket：

```text
ipc/update buffer: B
```

如果只看通信 buffer，旧路径大致是：

```text
2B + B = 3B
```

因为 PR 前 `B >= W`，所以通信 buffer 额外开销近似：

```text
3B ≈ 3W
```

如果把 source full tensor 也算进去，则是：

```text
source weight + communication buffers
= W + 3B
≈ W + 3W
= 4W
```

PR 描述中提到的 `max_weight_size * 3`，主要是通信 buffer 口径。严格分析时要说明口径：只算通信 buffer 是 `3W`，把 source tensor 也算进去接近 `4W`。

PR 后，NCCL/NIXL 可以把大 tensor 切成不超过 `B` 的 chunks 发送。于是 `B` 不再需要大于 `W`：

```text
source weight: W
checkpoint double buffer: 2B
ipc/update buffer: B
```

粗略峰值变成：

```text
W + 3B
```

如果只看通信 buffer，则是从：

```text
3W
```

降到：

```text
3B
```

这就是 #6091 最主要的系统收益。

---

## 6. #6091 的设计

这条 PR 没有推翻原有 weight sync 架构，而是在两个关键位置给大 tensor 开了特殊路径。

改动涉及 7 个文件：

```text
verl/checkpoint_engine/base.py
verl/checkpoint_engine/nccl_checkpoint_engine.py
verl/checkpoint_engine/nixl_checkpoint_engine.py
verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py
tests/checkpoint_engine/test_correctness_on_gpu.py
tests/checkpoint_engine/test_utils.py
tests/utils/test_bucketed_weight_transfer.py
```

### 6.1 NCCL/NIXL: Split and Merge

PR 在 checkpoint engine 基础层引入了 `TensorMeta`：

```text
name
shape
dtype
chunk_offset
chunk_size
offset
```

其中：

- `chunk_offset`: chunk 在原始 tensor byte view 里的起点。
- `chunk_size`: chunk 的 byte 大小。
- `offset`: chunk 被放入当前通信 bucket 后，在 bucket 里的起点。

发送侧通过 `split_weight_chunks` 把 tensor 看成 uint8 byte buffer：

```python
buffer = weight.view(-1).view(torch.uint8)
```

然后按 bucket size 切：

```text
chunk_size = min(bucket_size, weight.nbytes - chunk_offset)
```

小 tensor 只有一个 chunk。大 tensor 会被拆成多个 chunk。

接收侧通过 `merge_weight_chunks` 重组。如果发现原 tensor 大于 bucket size，就先分配完整目标 tensor：

```python
merge_weight = torch.empty(tensor_meta.shape, dtype=tensor_meta.dtype, device=chunk.device)
```

然后把每个 chunk 写到对应 byte range：

```text
merge_buffer[chunk_offset : chunk_offset + chunk_size] = chunk
```

收到最后一个 chunk 后，yield 完整 `(name, weight)`。

NCCL 和 NIXL 的 `send_weights` 因此从遍历：

```text
(name, weight)
```

变成遍历：

```text
(tensor_meta, chunk)
```

接收侧先拿到 chunk，再通过 merge 还原成原来的 tensor 流。对上层 rollout adapter 来说，接口仍然是 `(name, weight)`。

### 6.2 vLLM Colocated: Direct IPC for Large Tensor

vLLM colocated bucketed transfer 的策略更简单：

```text
小 tensor:
  copy 到 bucket buffer
  receiver 按 offset 切片

大 tensor:
  不进 bucket
  直接发送这个 tensor 的 CUDA IPC handle
```

代码路径大致是：

```text
if offset + weight.nbytes > bucket_size and bucket_meta is not empty:
    flush current bucket

if weight.nbytes > bucket_size:
    direct_send_large_weight(name, weight)
else:
    copy weight bytes into bucket
```

direct path 会导出该 weight 的 IPC handle：

```python
handle = reduce_tensor(weight)
```

metadata 里带上 `handle`：

```text
{
  "gate_up_proj": {
    name,
    shape,
    dtype,
    offset: 0,
    handle: CUDA_IPC_HANDLE
  }
}
```

receiver 看到 `handle is not None` 后，不再从 shared bucket 切片，而是：

```python
tensor = rebuild_ipc(handle, self.device.index)
```

这条路径保留了 bucket 对小 tensor 的 batching 优势，同时避免大 tensor 被复制进一个同样巨大的 bucket。

---

## 7. 测试覆盖

PR 调整了 checkpoint engine 测试，让 NCCL/NIXL correctness test 可以传入更小的 `bucket_size_mb`，从而验证 bucket 小于某些权重时仍然能正确同步。

更直接的是 `tests/utils/test_bucketed_weight_transfer.py` 新增了 `test_large_weight`：

```text
bucket_size_mb = 1

embedding:    4MB
gate_up_proj: 4MB
lm_head:      4MB
```

这个测试构造了多个大于 bucket 的 tensor，验证 CUDA IPC direct path 能正确传输，并且小 tensor 仍然可以混在普通 bucketed transfer 里。

这正好覆盖了 #6091 的关键场景：

```text
large tensor > bucket size
```

---

## 8. 边界和后续空间

#6091 的方向很清楚，但权重通信代码对生命周期和同步非常敏感。几个边界值得继续关注。

### 8.1 Direct IPC 前的同步

PR 讨论里有一个合理提醒：direct IPC 路径在导出 handle 前最好显式同步：

```python
get_torch_device().synchronize()
handle = reduce_tensor(weight)
```

原因是 GPU tensor 可能刚经历异步 copy 或 compute。如果 sender 在写入还没完成时就导出 IPC handle，receiver 理论上可能看到未完成的数据。

bucket path 在 flush 前有 synchronize。direct path 最好保持同样语义。

### 8.2 Shared Memory 仍不支持超大 Tensor

当前 direct large weight 只支持 CUDA IPC：

```python
assert not self.use_shm
```

所以在当前代码里，NPU 不一定总走 shared memory；只有 `is_support_ipc()` 判断不支持 IPC 时才会走。对于这些 `use_shm=True` 的 fallback 路径，如果单个 tensor 大于 bucket，仍然需要调大 bucket，或者等待 shared memory direct/chunk path 的后续实现。

### 8.3 Merge 仍然需要完整 Tensor

NCCL/NIXL chunking 避免了超大通信 bucket，但接收侧 `merge_weight_chunks` 仍然会分配完整 tensor。

因此峰值仍然至少包含：

```text
max_weight_size
```

更进一步的优化是把 chunk 直接写入 rollout engine 的目标参数存储，而不是先 merge 成完整临时 tensor。这个方向更复杂，因为它要求目标 engine 支持按 slice 更新参数。

### 8.4 类型标注还有清理空间

PR 合并时还有一些静态类型层面的粗糙处：

- `TensorMeta.offset` 标成 `int`，但初始化时用了 `None`。
- `split_weight_chunks` 的输入标注是同步 `Generator`，但实际通过 `ensure_async_iterator` 同时支持 async iterator。
- `merge_weight_chunks` 参数标注也是同步 `Generator`，但函数内部用的是 `async for`。
- NCCL `_receive_weight_chunks` 的返回类型标注和实际 yield 不完全一致。

这些不一定影响运行，但如果后续引入更严格的静态检查，应该补掉。

---

## 9. 总结

#6091 的关键价值不是让 weight sync 免费，而是移除了一个不合理的显存放大项。

旧设计里：

```text
一个 8GB gate_up_proj
  → bucket 必须至少 8GB
  → double buffer / IPC buffer 也被迫进入 8GB 级别
```

新设计里：

```text
小 tensor 继续 bucketed transfer
大 tensor 走 chunk 或 direct IPC
bucket size 回到通信效率参数
```

这是一个典型的系统优化：平均情况继续 batching，长尾情况走特殊路径。

对大 MoE policy 的 RL rollout refit 来说，这类改动不改变算法，也不会体现在模型能力指标里。但它可能决定一个训练任务是能顺利进入下一轮 rollout，还是在第一次权重同步时就 OOM。
