# veRL 全量权重同步：从训练分片到 Rollout Engine 的完整链路

> veRL 的“全量同步”不是先保存一份完整 checkpoint，再让推理服务重新加载；它是一条在线 refit 流水线：训练后端逐个重建 Hugging Face 语义下的完整逻辑 tensor，传输层按 bucket 搬运，vLLM 或 SGLang 再把它写入自己的 TP/EP 本地布局。

大模型强化学习通常同时维护两份策略：

- actor/trainer 上的可训练模型，布局可能是 FSDP、Megatron TP/PP/EP 或 VeOmni；
- rollout engine 中的生成模型，布局通常由 vLLM 或 SGLang 的 TP/DP/EP 决定。

一次 `optimizer.step()` 之后，trainer 中的参数已经更新，而 rollout engine 仍持有旧版本。下一轮采样前，veRL 必须把新策略从“训练并行布局”迁移到“推理并行布局”。这就是权重同步。

本文只讨论 **full named-tensor sync**：每轮覆盖全部逻辑权重。增量路径可继续阅读同目录的[《veRL 增量权重同步：从 Shard-local Delta 到跨训练后端 HF 导出》](veRL增量权重同步方案.md)。

资料口径固定为 **2026 年 8 月 13 日**，代码基线为 veRL `main` 的 commit [`535c4779`](https://github.com/verl-project/verl/commit/535c47799b537a0e3602b9839344ee53d2a47128)。文中的 benchmark 数字均来自官方仓库或 PR 在特定模型、硬件和拓扑下的报告，不代表跨环境可直接复现的统一性能。

---

## 一、“全量”到底是什么意思

### 1. 全量的是语义覆盖范围，不是内存形态

设模型的逻辑参数为：

$$
\mathcal{W}=\{W_1,W_2,\ldots,W_n\}
$$

全量同步要求一次成功更新最终覆盖集合中的每个参数：

$$
\forall W_i \in \mathcal{W},\quad
W_i^{rollout}\leftarrow W_i^{actor}
$$

但这不表示 veRL 会同时在一张 GPU 上构造一份完整模型副本。当前 FSDP exporter 是 generator：参数被消费到时才执行 `DTensor.full_tensor()`，一张一张产生完整逻辑 tensor。

因此下面三句话可以同时成立：

- 语义上，每轮同步全部权重；
- 导出时，只逐 tensor materialize；
- 传输时，再把 tensor 拼成有限大小的 bucket。

“full”回答的是**哪些参数被更新**；generator 和 bucket 回答的是**这些参数以什么内存节奏被更新**。

### 2. 它不是训练 checkpoint

训练 checkpoint 主要服务于故障恢复，通常包含模型、优化器、学习率调度器、随机数状态和训练进度，并可能写入本地盘或对象存储。

rollout 权重同步只服务于在线推理 refit：

~~~text
optimizer.step()
    │
    ├─ 从训练 shard 导出最新模型参数
    ├─ 在线传给 rollout worker
    ├─ 写入推理引擎
    └─ 清理旧权重关联的推理缓存
~~~

它不会为了每个训练 step 先落一份可恢复 checkpoint，也不会同步 optimizer state。

### 3. 一次同步至少有六段成本

~~~text
actor local shards
  │
  ├─ 1. materialize：从训练分片重建逻辑 tensor
  ├─ 2. convert：训练命名/融合布局 → HF named tensor
  ├─ 3. checkpoint-engine transport：跨节点广播，可选
  ├─ 4. local handoff：同 GPU 跨进程 CUDA IPC / shared memory
  ├─ 5. install：推理 loader 写入 TP/EP 本地参数
  └─ 6. control：暂停请求、清 cache、恢复生成
~~~

可以粗略记为：

$$
T_{sync} =
T_{export} + T_{convert} + T_{CE}
+ T_{IPC} + T_{install} + T_{control}
$$

这些阶段存在有限 overlap，实际耗时不一定严格相加。但只报告 NCCL 带宽，无法代表端到端同步时间。

---

## 二、当前主线架构：Model Engine、Checkpoint Engine 与 ServerAdapter

veRL 当前实现已经不再以旧版 `FSDPVLLMShardingManager` 为中心。主调用链是：

~~~text
PPO trainer
  │
  └─ CheckpointEngineManager.update_weights(global_steps)
       │
       ├─ colocated：backend = naive
       │    └─ ActorRolloutRefWorker.update_weights()
       │         ├─ ModelEngine.get_per_tensor_param()
       │         └─ ServerAdapter.update_weights()
       │
       └─ disaggregated：backend = nccl / 其他 CE
            ├─ actor CheckpointEngine.send_weights()
            └─ rollout CheckpointEngine.receive_weights()
                 └─ ServerAdapter.update_weights()
~~~

三个抽象的职责边界很清楚：

| 组件 | 主要职责 | 不负责什么 |
|---|---|---|
| Model Engine | 从 FSDP、Megatron、VeOmni 等训练布局导出 HF 语义 tensor | 不决定 rollout 的本地 TP slice |
| Checkpoint Engine | 在分离部署时跨 actor/rollout 节点传输权重流 | 不等于磁盘 checkpoint |
| ServerAdapter | 把 tensor 交给 vLLM/SGLang，并处理引擎生命周期 | 不负责 optimizer 或反向传播 |

全量路径默认的 wire contract 是：

~~~text
wire_format = named_tensors
item        = (parameter_name, full_logical_tensor)
~~~

定义可见 [`CheckpointEngine`](https://github.com/verl-project/verl/blob/535c47799b537a0e3602b9839344ee53d2a47128/verl/checkpoint_engine/base.py#L96-L205) 和 [`BaseRollout.update_weights`](https://github.com/verl-project/verl/blob/535c47799b537a0e3602b9839344ee53d2a47128/verl/workers/rollout/base.py#L49-L64)。

### 1. 默认配置并不是 NCCL

当前 [`CheckpointEngineConfig`](https://github.com/verl-project/verl/blob/535c47799b537a0e3602b9839344ee53d2a47128/verl/workers/config/rollout.py#L124-L141) 的关键默认值是：

~~~yaml
checkpoint_engine:
  backend: naive
  update_weights_bucket_megabytes: 2048
~~~

`naive` 适用于 trainer 与 rollout colocated 的常规同步训练；NCCL Checkpoint Engine 主要解决 actor GPU 池与 rollout GPU 池分离时的跨节点全量广播。

“naive”只表示不额外建立远端 Checkpoint Engine 数据面，并不表示整个 state dict 通过普通 Python 对象逐项复制。当前 vLLM/SGLang server 通常仍是独立进程，最后一跳仍会用 CUDA IPC 或相应的共享机制。

---

## 三、什么时候触发同步

同步 trainer 在初始化完成后先同步一次，之后每个 actor step 结束时再次同步。入口见 [`PPOTrainerSync.on_init_end/on_step_end`](https://github.com/verl-project/verl/blob/535c47799b537a0e3602b9839344ee53d2a47128/verl/trainer/ppo/v1/trainer_sync.py#L31-L42)。

典型时序是：

~~~text
第 k 轮 rollout（使用 W_k）
       │
       ▼
reward / advantage / actor loss
       │
       ▼
optimizer.step() 产生 W_(k+1)
       │
       ▼
update_weights(global_steps=k+1)
       │
       ▼
第 k+1 轮 rollout（使用 W_(k+1)）
~~~

在同步训练中，更新天然位于 generation 与下一次 generation 之间。在支持 partial rollout 的分离式流程中，管理器还会先中止并保存未完成请求，安装新权重后再恢复它们。

这里要区分两个版本概念：

- `global_steps` 会被写入 rollout server，便于标识当前模型版本；
- 真正的权重一致性来自完整加载和 cache invalidation，而不是这个整数标签本身。

---

## 四、训练侧导出：先统一成 HF 逻辑坐标

训练后端与 rollout 后端可以使用完全不同的并行拓扑。veRL 没有要求 rollout 直接理解 actor 的 shard，而是把 HF named tensor 作为中间语义层：

~~~text
FSDP / Megatron / VeOmni training layout
                    │
                    │ ModelEngine exporter
                    ▼
        HF name + full logical shape
                    │
                    │ rollout loader
                    ▼
        vLLM / SGLang local TP/EP layout
~~~

这相当于把复杂问题拆成两个转换：

1. 训练布局 → HF checkpoint 语义；
2. HF checkpoint 语义 → 推理引擎布局。

### 1. 不同 Model Engine 的导出方式

| 训练后端 | 全量导出的主要动作 |
|---|---|
| FSDP | 取得 state dict，转换 key，对 DTensor 逐项 `full_tensor()`，再处理 MoE unfuse |
| Megatron | 通过 Megatron Bridge 导出，完成 TP/PP/EP fusion 和 HF key 映射 |
| VeOmni | 逐项聚合 DTensor，并对 EP expert 做 broadcast、restack 和模型专用转换 |
| TorchTitan | 通过 state-dict adapter 转 HF 命名，并处理 tied weight、EP gather |

Megatron 当前入口见 [`get_per_tensor_param`](https://github.com/verl-project/verl/blob/535c47799b537a0e3602b9839344ee53d2a47128/verl/workers/engine/megatron/transformer_impl.py#L1007-L1045)。它不是把某个 TP rank 的 local shard 原样发给 rollout，而是借助 Bridge 导出推理 loader 能理解的逻辑权重。

### 2. dtype 不应该被想当然地统一成 BF16

全量 named-tensor 路径保留 exporter 产生的 dtype。vLLM 的 bucket sender 甚至明确避免统一 cast，因为 MoE gate 等参数可能必须保持 FP32。

所以，总传输量应按各 tensor 的实际 dtype 计算：

$$
B_{full}=\sum_i numel(W_i)\times sizeof(dtype_i)
$$

不能直接用“参数量 × 2 字节”覆盖所有模型。只有明确知道所有 wire tensor 均为 BF16/FP16 时，这个近似才成立。

---

## 五、FSDP 全量导出的关键：惰性 `full_tensor()`

当前 FSDP 实现位于 [`FSDPEngine.get_per_tensor_param`](https://github.com/verl-project/verl/blob/535c47799b537a0e3602b9839344ee53d2a47128/verl/workers/engine/fsdp/transformer_impl.py#L949-L1033)，核心逻辑可以简化为：

~~~python
params = module.state_dict()
params = convert_weight_keys(params, module)

per_tensor_param = (
    (
        name,
        param.to(device, non_blocking=True).full_tensor()
        if isinstance(param, DTensor)
        else param,
    )
    for name, param in params.items()
)
~~~

### 1. 为什么 generator 很重要

如果先把全部 DTensor 都 all-gather，再构造一个完整 GPU state dict，峰值显存会接近“训练 shard + 一整份 full model + rollout weights”。generator 将这一过程改成按需发生：

~~~text
消费 W1 → all-gather W1 → 发送/加载 W1
消费 W2 → all-gather W2 → 发送/加载 W2
...
消费 Wn → all-gather Wn → 发送/加载 Wn
~~~

这避免了额外的整模型副本，但没有消除每个 tensor 的 all-gather，也不能让单个超大 tensor 的峰值消失。

### 2. 所有 actor rank 都必须消费同一生成器

`DTensor.full_tensor()` 是 FSDP mesh 上的 collective。即使只有 actor rank 0 负责向远端 rollout 广播，其他 actor rank 仍必须：

- 以一致顺序遍历参数；
- 进入相同的 `full_tensor()`；
- 等待 collective 完成。

否则 rank 0 会在 all-gather 中永久等待。

这也是 NCCL Checkpoint Engine 中其他 actor rank 虽然被标成 CE rank `-1`，仍然要把 generator 完整消费掉的原因。实现见 [`NCCLCheckpointEngine.send_weights`](https://github.com/verl-project/verl/blob/535c47799b537a0e3602b9839344ee53d2a47128/verl/checkpoint_engine/nccl_checkpoint_engine.py#L231-L307)。

### 3. FSDP2 已经避免一次不必要的整 shard staging

对 FSDP2 非 LoRA 模型，当前 exporter 不再先把整个 local shard 从 CPU 搬到 GPU，再逐 tensor 导出；它直接让每个 DTensor 在被消费时执行 `.to(device).full_tensor()`。

官方 [PR #7005](https://github.com/verl-project/verl/pull/7005) 在 Qwen2.5-7B、2 节点、8 GPU FSDP2 trainer + 8 GPU SGLang、参数 offload 开启的特定测试中报告：

| 指标 | 优化前 | 优化后 |
|---|---:|---:|
| exporter 创建阶段 | 3.3～4.5 s | 约 0.07 s |
| 每步全量同步 | 6.65～6.93 s | 3.36～3.49 s |

这是工作负载相关的 PR 数据，不是所有模型上的固定收益。

优化后剩下的主要开销仍可能是逐参数 collective。[Issue #7015](https://github.com/verl-project/verl/issues/7015) 在同类环境中观察到约 340 次独立 `full_tensor()` all-gather，占约 3.4 秒总时间中的 2.4～2.7 秒，并提出按 layer/bucket 批量 unshard 的方向；截至本文基线，它仍应视为待优化问题，而不是已落地能力。

---

## 六、colocated 的 `naive` 路径

默认路径的实现位于 [`ActorRolloutRefWorker.update_weights`](https://github.com/verl-project/verl/blob/535c47799b537a0e3602b9839344ee53d2a47128/verl/workers/engine_workers.py#L719-L805)。

其顺序可以还原为：

~~~text
ActorRolloutRefWorker.update_weights(mode="naive")
  │
  ├─ 关闭 expandable-segments，清理 allocator cache
  ├─ rollout.resume(tags=["weights"])      # free_cache_engine 开启时
  ├─ actor.engine.get_per_tensor_param()
  ├─ rollout.update_weights(named tensors)
  ├─ actor 参数重新 offload 到 CPU          # param_offload 开启时
  ├─ 清理 allocator cache
  └─ rollout.resume(tags=["kv_cache"])
~~~

这里有两个常见误解。

### 1. colocated 不等于同一 Python 进程

Model Engine worker 和 vLLM/SGLang inference worker 可以在同一 GPU 上，但处于不同进程。veRL 仍需要一条同 GPU 跨进程通道：

~~~text
actor process
   │
   │ CUDA IPC handle + ZMQ metadata
   ▼
vLLM worker process on the same GPU
~~~

因此，即使没有跨节点 NCCL，仍有 bucket copy、IPC 握手和 `model.load_weights()` 成本。

### 2. `free_cache_engine` 只控制内存生命周期

它决定 rollout weights/KV cache 是否 sleep、何时 wake，不应该决定“这一步是否同步”。同步语义与显存释放策略是两件事。官方曾在 [PR #4248](https://github.com/verl-project/verl/pull/4248) 专门修复这类耦合。

---

## 七、disaggregated NCCL：actor rank 0 广播，其他 actor rank 协同导出

当 trainer GPU 池与 rollout GPU 池分离时，数据路径多出一跳：

~~~text
Actor FSDP group                           Checkpoint Engine group

actor rank 0 ─┐                            CE rank 0
actor rank 1 ─┼─ full_tensor collectives ─ actor rank 0
...          │                                  │
actor rank A ─┘                                  │ NCCL broadcast
                                                 ├──────── rollout CE rank 1
                                                 ├──────── rollout CE rank 2
                                                 └──────── rollout CE rank R
                                                               │
                                                               │ same-GPU IPC
                                                               ▼
                                                      inference worker process
~~~

### 1. Checkpoint Engine 组的 world size 不是 actor + rollout

[`build_topology`](https://github.com/verl-project/verl/blob/535c47799b537a0e3602b9839344ee53d2a47128/verl/checkpoint_engine/nccl_checkpoint_engine.py#L161-L173) 创建的 CE 拓扑是：

~~~text
CE rank 0       = actor global rank 0
CE rank 1...R   = R 个 rollout CheckpointEngineWorker
CE world size   = R + 1
其他 actor rank = -1，不加入 CE group
~~~

其他 actor rank 不发送远端数据，但必须参加训练侧的 FSDP/Megatron 导出 collective。这样既避免多个 actor sender 重复广播，又保持训练分片能够正确重建。

### 2. 元数据走 ZMQ，权重字节走 NCCL

每个 bucket 的控制信息包含：

- tensor 名称、shape 和 dtype；
- bucket 内 offset；
- 大 tensor 的 chunk offset、chunk size；
- bucket 实际长度与是否为最后一个 bucket。

ZMQ PUB/SUB 发送这些 metadata；NCCL broadcast 发送 GPU 上的 `uint8` byte buffer。二者缺一不可：

~~~text
ZMQ metadata：这段 byte 对应谁、怎么还原
NCCL payload：真正的参数字节
~~~

当前实现只广播 bucket 中实际使用的 slice，而不是把最后一桶未使用的尾部也发出去，见 [commit #7107](https://github.com/verl-project/verl/commit/1ddf0713eeddd8b1c4ec862292a970037c5b2096)。

### 3. 双缓冲与 tensor chunk

NCCL Checkpoint Engine 准备两个固定大小 buffer：

~~~text
send_buf / recv_buf
~~~

发送端和接收端在两者之间交换角色，使“消费上一桶”和“接收下一桶”能够交替进行。它可以形成有限流水，但不应理解成所有 materialize、NCCL 和安装阶段都能完全重叠。

如果单个 tensor 大于 bucket，CE 会用 `split_weight_chunks()` 切块；接收端再用 `merge_weight_chunks()` 还原完整 tensor。相关实现见 [`base.py`](https://github.com/verl-project/verl/blob/535c47799b537a0e3602b9839344ee53d2a47128/verl/checkpoint_engine/base.py#L541-L609)。

[PR #6091](https://github.com/verl-project/verl/pull/6091) 引入了这条大 tensor 路径，主要针对 MoE fused weight 可能单张达到数 GB、无法放进常规 bucket 的情况。chunk 限制了网络 staging buffer，却不保证接收、重组和推理安装期间完全不出现一张完整大 tensor。

---

## 八、同步前后如何管理在途请求和 KV cache

分离式 [`CheckpointEngineManager.update_weights`](https://github.com/verl-project/verl/blob/535c47799b537a0e3602b9839344ee53d2a47128/verl/checkpoint_engine/base.py#L486-L538) 执行：

~~~text
1. abort 并保存未完成的 partial rollout
2. 创建临时 rollout worker group
3. release KV cache，保留模型权重存储
4. 建立或复用 Checkpoint Engine 通信组
5. actor send 与 rollout receive/apply 并发启动
6. finalize 本轮 buffer/通信资源
7. resume KV cache
8. 恢复未完成请求
~~~

### 1. 为什么权重切换前要让请求静止

关键原因是：**veRL 不是“瞬间”换掉整个模型，而是把新权重分成多个 bucket，逐批写进正在服务的模型。**

假设 rollout 当前使用旧权重 $W_k$，trainer 刚产生新权重 $W_{k+1}$。权重更新到一半时，模型内部会短暂处于这种状态：

~~~text
已经更新的层：W_(k+1)
还没更新的层：W_k
~~~

如果这时仍让某个请求继续生成，它的一次 forward 就可能同时用到新旧两版权重。得到的不是 $W_k$ 模型或 $W_{k+1}$ 模型的输出，而是一个临时的“拼接模型”的输出。

为了避免这件事，管理器会先把未完成的请求停下来并保存，等全部新权重安装完成后再恢复：

~~~text
请求使用 W_k 生成
        ↓
暂停并保存未完成请求
        ↓
逐批安装 W_(k+1)
        ↓
新权重全部就位后，恢复请求
~~~

这样可以保证，正常成功路径上运行的每次 forward，看到的要么是完整的 $W_k$，要么是完整的 $W_{k+1}$。

注意，这不是底层权重的“原子切换”。veRL 没有额外构造一份完整的新模型，然后用一个指针瞬间替换旧模型；它只是在更新期间把请求拦在外面。文中所说的**控制面一致性**，指的就是这个意思。

### 2. vLLM 与 SGLang 的 cache 生命周期并不完全相同

- SGLang 当前会真正执行只释放/恢复 KV cache 的接口，代码见 [`async_sglang_server.py`](https://github.com/verl-project/verl/blob/535c47799b537a0e3602b9839344ee53d2a47128/verl/workers/rollout/sglang_rollout/async_sglang_server.py#L511-L524)。
- 当前 vLLM adapter 的 `release_kv_cache()`/`resume_kv_cache()` 仍是占位 no-op；请求暂停与 cache 清理由其他接口完成，见 [`vllm_async_server.py`](https://github.com/verl-project/verl/blob/535c47799b537a0e3602b9839344ee53d2a47128/verl/workers/rollout/vllm_rollout/vllm_async_server.py#L813-L897)。

因此不要根据管理器的统一方法名推断两个 backend 都释放了同样数量的显存。

### 3. cache 必须与模型版本一起失效

KV、prefix、multimodal 或 encoder cache 都隐含依赖产生它们的权重版本。同步完成后，vLLM 会清理相应 cache，SGLang 会统一 `flush_cache()`。否则即使参数本身已经正确更新，复用旧 cache 仍可能让输出混入旧策略状态。

---

## 九、NCCL 之后还有第二跳：rollout CE → 推理进程

Checkpoint Engine receiver 得到 full named tensor 后，并没有直接成为 vLLM/SGLang 的 model runner。rollout CheckpointEngineWorker 与真正的 inference worker 通常是同 GPU、不同进程：

~~~text
actor rank 0
   │
   │ NCCL across nodes
   ▼
rollout CheckpointEngineWorker
   │
   │ CUDA IPC / shared memory / SGLang tensor serialization
   ▼
vLLM or SGLang inference worker
   │
   └─ model.load_weights() → local TP/EP shard
~~~

这带来两个重要结论：

1. Checkpoint Engine benchmark 若只测“receive weight”，不等于端到端 rollout refit；
2. 调整 CE bucket 只能直接控制第一跳的 staging，第二跳仍有自己的 buffer、握手和大 tensor 规则。

---

## 十、vLLM 的安装路径：bucketed CUDA IPC

vLLM 入口是 [`ServerAdapter.update_weights`](https://github.com/verl-project/verl/blob/535c47799b537a0e3602b9839344ee53d2a47128/verl/workers/rollout/vllm_rollout/vllm_rollout.py#L208-L246)，它明确只接受 `wire_format="named_tensors"`。

### 1. 发送与接收

~~~text
ServerAdapter
  ├─ collective_rpc("update_weights_from_ipc")
  │    └─ 让 vLLM workers 进入接收状态
  │
  └─ BucketedWeightSender
       ├─ GPU uint8 bucket
       ├─ ZMQ 发送 handle/shape/dtype/offset
       ├─ vLLM worker 映射 CUDA IPC buffer
       └─ ACK 后 sender 才复用 bucket
~~~

代码位于：

- [`BucketedWeightSender`](https://github.com/verl-project/verl/blob/535c47799b537a0e3602b9839344ee53d2a47128/verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py#L103-L161)
- [`BucketedWeightReceiver`](https://github.com/verl-project/verl/blob/535c47799b537a0e3602b9839344ee53d2a47128/verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py#L264-L302)

ZMQ 传的是控制信息，tensor 字节通常留在 GPU，通过 CUDA IPC handle 被另一进程映射。设备不支持 GPU IPC 时可以回退到 POSIX shared memory。

### 2. 超大 tensor 的规则

- CUDA IPC 路径中，单 tensor 大于 bucket 时可以绕过 bucket，直接共享该 tensor 的 IPC handle；
- shared-memory fallback 当前要求 tensor 能放入 bucket，否则会断言失败；
- 所以在 shared-memory 环境中，bucket 至少要覆盖最大 wire tensor。

注意：CE 第一跳即使已经把大 tensor 切成 chunk，`merge_weight_chunks()` 仍会在交给 vLLM 前恢复 tensor 语义；不能把 CE chunk 自动等同于 vLLM loader 永远只看到 chunk。

### 3. 真正写入模型

vLLM worker 侧 [`update_weights_from_ipc`](https://github.com/verl-project/verl/blob/535c47799b537a0e3602b9839344ee53d2a47128/verl/workers/rollout/vllm_rollout/utils.py#L232-L337) 最终对普通参数调用：

~~~python
model.load_weights(param_updates)
~~~

推理模型 loader 根据当前 vLLM 的 TP/EP 布局选择本 rank 需要的 slice，并处理 fused parameter。FP8/QAT、ModelOpt、MTP drafter、buffer 和 LoRA 还有专门分支。

某些 `process_weights_after_loading()` 只在整轮权重加载完后调用一次，因为派生权重转换可能不是幂等的。把它错误地放到每个 bucket 后面，既慢，也可能重复转换模型状态。

---

## 十一、SGLang 的安装路径：IPC 描述聚合与逐桶 update

SGLang 入口见 [`ServerAdapter.update_weights`](https://github.com/verl-project/verl/blob/535c47799b537a0e3602b9839344ee53d2a47128/verl/workers/rollout/sglang_rollout/sglang_rollout.py#L291-L369)。

~~~text
每个本地 actor/CE rank 的 named tensors
       │
       ├─ get_named_tensor_buckets()
       ├─ GPU tensor IPC serialization
       ├─ infer_tp CPU mesh gather_object()
       │      # 聚合 handle/metadata，不搬完整 tensor 到 leader CPU
       ├─ TP leader 发 UpdateWeights 请求
       ├─ 各 SGLang TP worker 反序列化对应 GPU tensor
       └─ 全部 bucket 完成后 flush_cache()
~~~

实际更新函数来自 [SGLang 官方 `weight_sync.utils.update_weights`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/weight_sync/utils.py)。

### 1. 只有 leader 发请求，不等于只有 leader 导出

所有相关 rank 都必须消费 tensor generator，尤其在 FSDP 下必须一起参加 `full_tensor()`。只有 HTTP dispatch 可以由 TP leader 执行。

### 2. SGLang bucket 不会切开单个 tensor

veRL 的 [`get_named_tensor_buckets`](https://github.com/verl-project/verl/blob/535c47799b537a0e3602b9839344ee53d2a47128/verl/workers/rollout/sglang_rollout/utils.py#L90-L143) 只在 tensor 边界分桶：

~~~text
若当前 bucket + tensor > B：
  先 flush 旧 bucket
  新 bucket 从完整 tensor 开始
~~~

若一个 tensor 自身大于 B，这个 bucket 也会大于 B。它与 CE 的 tensor chunk 机制不同。

此外，若 tensor 是某个可复用大 buffer 的 view，就必须 clone 成独立紧凑存储，否则 bucket 会钉住底层大 buffer，甚至在下一轮 receive 覆盖后读到错误内容；如果 `full_tensor()` 已产生拥有独立 storage 的紧凑 tensor，则重复 clone 反而会放大显存峰值。当前实现只对 view clone，这一演进可参考 [PR #6738](https://github.com/verl-project/verl/pull/6738) 和后续修复 [commit `12daa78`](https://github.com/verl-project/verl/commit/12daa7855706202a21dea24e78ad6aceb9cc8307)。

---

## 十二、bucket 的成本模型与显存边界

设：

- 模型 full wire bytes 为 `M`；
- 配置 bucket bytes 为 `B`；
- 最大单 tensor bytes 为 `L`；
- rollout receiver 数量为 `R`。

### 1. bucket 不减少总语义字节

每个 rollout receiver 仍需逻辑上得到约 `M` 字节：

$$
B_{receiver}\approx M
$$

bucket 只改变：

- 单次 collective / IPC 的粒度；
- buffer 峰值；
- launch、metadata 与 ACK 次数；
- 不同阶段可流水的空间。

NCCL broadcast 的物理链路流量取决于 collective 算法和网络拓扑，不能简单写成 `M × R`；但“每个 receiver 最终获得完整模型”的语义不变。

### 2. CE 显存不只有一个 bucket

NCCL Engine 明确使用两个 buffer，因此 prepare 期间基础额外开销约为：

$$
M_{CE\ buffers}\approx 2B
$$

还应叠加：

- 当前正在 `full_tensor()` 的完整参数；
- 接收端重组超大 tensor 的存储；
- vLLM/SGLang 第二跳 buffer 或保留中的 tensor bucket；
- 推理模型本体及尚未释放的 cache；
- allocator fragmentation 和非权重运行时开销。

所以“2 GB bucket 只多占 2 GB”是错误的；仅 CE 双缓冲就大约是 4 GB，端到端峰值还会更高。

### 3. 太小与太大的权衡

| bucket 选择 | 好处 | 代价 |
|---|---|---|
| 较小 | staging 峰值低，更早开始消费 | collective、ZMQ metadata、IPC ACK 次数增加 |
| 较大 | 大包带宽利用率更好，launch 更少 | 双缓冲和第二跳峰值高，首桶等待时间更长 |
| 大于最大 tensor | 对 shared-memory fallback 更稳妥 | 在大 MoE tensor 上可能完全不可接受 |

默认 2048 MB 只能当起点，不是通用最优值。

---

## 十三、四个容易混为一谈的配置维度

### 1. `reshard_after_forward`

控制 FSDP 在训练 forward 后是否重新分片，属于 actor 的参数生命周期。即使 forward 后 reshard，全量同步时仍需重新构造 rollout 可加载的逻辑 tensor。

### 2. `param_offload`

控制 actor 参数在训练阶段之外是否放到 CPU。它影响同步前后的 H2D/D2H 与显存峰值，但不会改变 full sync 的覆盖范围。

### 3. `free_cache_engine`

控制 rollout weights/KV cache 的 sleep/wake 策略。它影响共置时能否给训练腾出显存，不决定 wire 是 full 还是 delta。

### 4. full / delta wire format

决定每轮发送所有 named tensor，还是发送 shard-local sparse replacement patch。它才直接改变同步语义字节数。

可以把四者放在不同坐标轴上：

~~~text
训练布局：FSDP / Megatron / VeOmni
驻留策略：GPU resident / CPU offload / rollout sleep
传输协议：naive IPC / NCCL CE / 其他 CE
更新语义：full named tensors / delta flushes
~~~

只看其中一个配置，无法推断整条同步链路。

---

## 十四、LoRA、量化与 MoE 的特殊语义

### 1. LoRA 不一定是在做“全量同步”

在 colocated naive 路径中：

- `lora.merge=false`：必要时先同步一次 base，之后主要同步 adapter；稳态不属于本文定义的全模型 full sync；
- `lora.merge=true`：每轮在 merge context 内把 base + adapter 合成普通 HF named tensors，再走全量加载。

FSDP merged LoRA 必须在 merge context 尚未退出时 materialize tensor，否则 state dict 可能仍 alias live storage，退出后恢复 base weight，最终悄悄发出未合并参数。当前实现用 [`_merged_lora_per_tensor_param`](https://github.com/verl-project/verl/blob/535c47799b537a0e3602b9839344ee53d2a47128/verl/workers/engine/fsdp/transformer_impl.py#L1035-L1064) 保持 generator 生命周期。

一个当前实现边界值得特别注意：非 naive 的普通 Checkpoint Engine 分支调用 `get_per_tensor_param()` 后丢弃了返回的 `peft_config`，见 [`engine_workers.py`](https://github.com/verl-project/verl/blob/535c47799b537a0e3602b9839344ee53d2a47128/verl/workers/engine_workers.py#L750-L758)。因此不要未经端到端验证就宣称 `NCCL + merge=false` 会自动完成 adapter hot update；需要远端全量同步时，优先采用已验证的 merged/full 组合。

### 2. FP8/QAT

全量传输不代表一定直接 memcpy 到参数：

- SGLang FP8 路径可在 trainer/adapter 侧按名字量化；
- vLLM 对 QAT、ModelOpt、FP8 可能先准备量化状态，再加载，再执行后处理；
- wire bytes、安装时间和临时显存都可能偏离 BF16 dense 模型的直觉。

### 3. MoE 与 fused tensor

MoE exporter 可能需要 unfuse、expert restack 或布局转换。`gate_up_proj`、QKV 或 expert stack 还可能形成数 GB 的单 tensor：

- CE 跨节点阶段可以 chunk；
- vLLM CUDA IPC 可对超大 tensor 走 direct handle；
- vLLM shared-memory fallback 和 SGLang tensor-boundary bucket 仍需单独检查；
- FP32 router/gate 还会提高实际 wire bytes。

---

## 十五、正确性边界：成功路径一致，不等于事务式提交

### 1. 正常成功路径

veRL 通过以下约束保证正常路径：

- 同步前暂停或中断在途生成；
- 所有 actor rank 以相同顺序参加 exporter collective；
- metadata 与 byte payload 一一对应；
- receiver 在 buffer 被复用前完成 load 或 clone；
- CUDA/NCCL kernel 完成后才释放 buffer；
- 全部参数安装后统一做 postprocess 和 cache invalidation；
- 最后恢复 generation，并更新 `global_steps`。

### 2. 中途失败没有自动 rollback

参数按 bucket 写入 live model。若第 20 桶加载后第 21 桶失败，当前机制没有另一份完整旧模型可供原子回滚。因此故障处理应遵循：

~~~text
本轮 sync 失败
  → 不要恢复 generation
  → 保持 replica 隔离
  → 重建通信/进程状态
  → 从头重放一次完整同步
  → 校验后再恢复服务
~~~

“请求在同步时被暂停”只保证成功路径不会观察半更新模型，不等于底层具备数据库事务。

### 3. ACK 和 tensor lifetime 是协议的一部分

CUDA IPC 共享的是 storage，不是自动生成的不可变副本。发送端过早复用 bucket，接收端就可能读取已被下一桶覆盖的数据。因此：

- vLLM 第二跳收到 ACK 后才复用 buffer；
- NCCL receiver 在 yield/换桶之间显式同步；
- SGLang 对指向复用 buffer 的 view 做紧凑 clone。

这些看似“多余”的 synchronize、ACK 和 clone，实际都在维护 tensor lifetime。

---

## 十六、怎样读官方 benchmark

### 1. Checkpoint Engine README 的数字不是端到端 refit

官方 [`verl/checkpoint_engine/README.md`](https://github.com/verl-project/verl/blob/535c47799b537a0e3602b9839344ee53d2a47128/verl/checkpoint_engine/README.md#L42-L60) 报告了如下 NCCL 样例：

| 模型与拓扑                                                                                   |   后端 |    时间 |        带宽 |
| --------------------------------------------------------------------------------------- | ---: | ----: | --------: |
| Qwen3-30B-A3B-Base；4×8 H100；CX-7 400 Gbps；actor FSDP world size 2；30 个 rollout receiver | NCCL | 约 7 s | 8.25 GB/s |

README 明确注明 rollout 侧“只接收权重，不包含到 vLLM/SGLang 的 CUDA IPC”。因此它主要度量 Checkpoint Engine 广播，不包含：

- CE receiver → inference worker 的第二跳；
- vLLM/SGLang loader 的完整安装；
- cache flush 和请求恢复；
- 某些真实训练布局下的全部导出开销。

### 2. 优先测阶段，而不是只测一个总数

建议至少拆出：

| 阶段 | 建议观测 |
|---|---|
| export | state dict 获取、key conversion、每参数 full_tensor 时间 |
| CE transport | 总 bytes、bucket 数、NCCL receive GB/s |
| local handoff | IPC/shared-memory bucket 数、ACK 等待 |
| install | load_weights 与 postprocess 时间 |
| control | abort/drain、cache flush、resume 时间 |
| memory | actor、CE worker、inference worker 各自 peak allocated/reserved |

当前 NCCL Engine 会记录 send/receive time 和 receiver 侧 GB/s，但 full backend 返回的 per-sync metrics 仍很有限。生产环境通常需要在外层补充 profiling。

---

## 十七、调优顺序

### 1. 先确认部署路径

~~~text
actor 与 rollout colocated？
  ├─ 是：优先看 naive + ServerAdapter IPC
  └─ 否：继续看 Checkpoint Engine
          ├─ NCCL/HCCL/NIXL 等第一跳
          └─ CE → inference process 第二跳
~~~

如果路径判断错了，调 NCCL bucket 可能对默认 naive 作业完全没有作用。

### 2. 再确认最大 tensor 与可用显存

启动前统计每个导出 tensor 的：

- name；
- shape；
- dtype；
- nbytes；
- 是否为 view；
- 是否大于配置 bucket。

尤其关注 MoE expert stack、fused QKV 和 `gate_up_proj`。

### 3. 从默认 bucket 做双向扫描

以 2048 MB 为起点，在相同模型与拓扑上分别测试更小和更大 bucket：

- 若显存紧张或 OOM，先减小 bucket；
- 若 bucket/ACK/collective 次数过多且显存充足，再增大；
- 若使用 shared memory，先保证最大 tensor 能放入；
- 若使用 SGLang，记住单 tensor 不会被其 helper 切开。

### 4. 优先消除不必要的数据移动

比起盲目扩大 bucket，下面几项通常更值得先核对：

- FSDP2 是否走按 tensor staging，而不是整个 shard 往返 CPU/GPU；
- dtype 是否被无意提升；
- tensor 是否因 view lifetime 被重复 clone；
- rollout 权重/KV cache 是否按预期 sleep/wake；
- 是否把只需 adapter 的 LoRA 作业误配成 merged full sync；
- 是否每步都重建本可复用的通信组。

`rebuild_group=false` 可以复用拓扑稳定的 NCCL group；如果 replica 拓扑实际变化，则必须正确重建，不能复用失效 rank 映射。

---

## 十八、排障地图

| 现象 | 优先检查 |
|---|---|
| 卡在第一个 `full_tensor()` | 是否有 actor rank 未消费 generator，或参数遍历顺序不一致 |
| exporter 很慢但 NCCL 很快 | 逐参数 all-gather、CPU offload staging、格式转换 |
| ZMQ metadata 到了但 NCCL 卡住 | CE rank/world size、group 名称、网络/NCCL 配置 |
| CE 显示完成但 rollout 输出仍旧 | 第二跳 IPC、loader 是否完成、cache 是否清理、global step 是否更新 |
| 权重偶发损坏 | buffer 是否在 receiver 完成前复用、view 是否指向下一轮会覆盖的 storage |
| 调小 bucket 仍 OOM | 最大单 tensor、CE merge、SGLang 不切 tensor、第二跳 buffer |
| vLLM shared-memory 路径断言 | 单 tensor 大于 bucket；该 fallback 不支持 direct large-tensor IPC |
| 只有 partial rollout 出错 | abort/drain/resume 时序与 cache 生命周期 |
| LoRA 同步后效果像 base model | merge context 生命周期、`merge` 配置、peft metadata 是否到达远端 |
| 换 replica 数后 hang | 旧通信组是否被错误复用，`rebuild_group` 是否匹配拓扑 |

官方 GPU 正确性测试会反复同步并校验权重，也覆盖复用/重建 group 的组合，可参考 [`test_correctness_on_gpu.py`](https://github.com/verl-project/verl/blob/535c47799b537a0e3602b9839344ee53d2a47128/tests/checkpoint_engine/test_correctness_on_gpu.py#L33-L81)。

生产验收还应增加：

1. 固定输入在同步前后的 logits/log-prob 对比；
2. 同一 actor checkpoint 与 rollout 加载结果的抽样 checksum；
3. 连续多轮同步，捕获 buffer 复用类偶发错误；
4. 注入中途失败，确认 replica 不会带着半更新权重恢复生成；
5. 对 FP8、LoRA、MoE、MTP 分别建立模型专用 golden case。

---

## 十九、旧版 ShardingManager 为什么不应再作为主线

很多旧文章会从：

~~~text
FSDPVLLMShardingManager.__enter__()
  → state_dict
  → full_tensor
  → model_runner.model.load_weights
~~~

解释 veRL 权重同步。这对 v0.5 时代是成立的，但当前主线已经迁移。

- [PR #3285](https://github.com/verl-project/verl/pull/3285) 弃用 rollout sharding managers；
- [PR #4280](https://github.com/verl-project/verl/pull/4280) 推进 model runner 分离与 CUDA IPC refit；
- [PR #4775](https://github.com/verl-project/verl/pull/4775) 引入 Checkpoint Engine 抽象；
- [PR #5031](https://github.com/verl-project/verl/pull/5031) 引入统一的 Checkpoint Engine Manager。

旧实现仍适合用来理解“FSDP full tensor → inference load”的思想来源，但排查当前代码时应从：

~~~text
CheckpointEngineManager
  → ActorRolloutRefWorker
  → ModelEngine
  → CheckpointEngine
  → ServerAdapter
~~~

开始。

---

## 二十、全量与增量方案的边界

| 维度 | 全量 named-tensor sync | `delta_sharded` |
|---|---|---|
| 每轮覆盖 | 所有逻辑参数 | 变化位置的 replacement patch |
| 训练侧重建 | 通常需要逐 tensor full gather/fusion | 稳态目标是从 local shard 直接导出 patch |
| wire bytes | 约为完整模型实际 dtype 字节数 | 与 changed-element ratio 和索引开销有关 |
| 接收端 | 标准 vLLM/SGLang full-weight loader | 需要支持稀疏原地 apply 的专用路径 |
| 复杂度 | 语义直接、兼容面更广 | snapshot、布局映射、flush、校验更复杂 |
| 适用场景 | 首次 seed、变化密集、兼容性优先 | 相邻版本变化稀疏且同步成为瓶颈 |

增量同步并不会让全量路径消失：

- rollout 第一次启动仍需要一份完整基线；
- snapshot 丢失或验证失败时需要 full resync；
- 新模型后端通常先打通 full loader；
- 权重变化密集时，delta 的索引和协议开销可能不再划算。

因此更合理的理解是：

~~~text
full sync  = 正确性基线 + 通用恢复路径
delta sync = 满足条件时的稳态优化
~~~

---

## 二十一、总结

veRL 当前全量权重同步可以压缩成四句话：

1. Model Engine 把 FSDP、Megatron、VeOmni 等训练 shard 转成 HF 语义的完整 named-tensor 流；
2. colocated 作业直接经 ServerAdapter 做同 GPU 跨进程更新，分离部署则先由 Checkpoint Engine 跨节点广播；
3. rollout 端的 vLLM/SGLang loader 再把完整逻辑 tensor 写入自己的 TP/EP 本地布局，并清理旧模型 cache；
4. “全量”意味着每轮覆盖所有参数，不意味着整模型同时 materialize，也不意味着只有一次 NCCL broadcast。

整条路径中最容易被低估的并不是某个单独 API，而是三个边界：

- 训练 shard 重建完整 tensor 的 collective 成本；
- NCCL 之后到推理进程的第二跳；
- bucket-by-bucket live update 所要求的请求暂停、buffer lifetime 和 cache 一致性。

只有把 export、跨节点 transport、本机 IPC、engine install 和控制面分别测量，才能判断瓶颈究竟在“权重太多”、在“布局转换太慢”，还是在“同步协议之外的那一跳”。

---

## 代码索引

| 主题 | 当前源码 |
|---|---|
| 同步 trainer 触发点 | [`trainer_sync.py`](https://github.com/verl-project/verl/blob/535c47799b537a0e3602b9839344ee53d2a47128/verl/trainer/ppo/v1/trainer_sync.py#L31-L42) |
| Checkpoint Engine 配置 | [`rollout.py`](https://github.com/verl-project/verl/blob/535c47799b537a0e3602b9839344ee53d2a47128/verl/workers/config/rollout.py#L124-L141) |
| CheckpointEngineManager | [`checkpoint_engine/base.py`](https://github.com/verl-project/verl/blob/535c47799b537a0e3602b9839344ee53d2a47128/verl/checkpoint_engine/base.py#L447-L538) |
| worker 权重更新 | [`engine_workers.py`](https://github.com/verl-project/verl/blob/535c47799b537a0e3602b9839344ee53d2a47128/verl/workers/engine_workers.py#L719-L805) |
| FSDP full exporter | [`fsdp/transformer_impl.py`](https://github.com/verl-project/verl/blob/535c47799b537a0e3602b9839344ee53d2a47128/verl/workers/engine/fsdp/transformer_impl.py#L949-L1064) |
| Megatron full exporter | [`megatron/transformer_impl.py`](https://github.com/verl-project/verl/blob/535c47799b537a0e3602b9839344ee53d2a47128/verl/workers/engine/megatron/transformer_impl.py#L1007-L1045) |
| NCCL Checkpoint Engine | [`nccl_checkpoint_engine.py`](https://github.com/verl-project/verl/blob/535c47799b537a0e3602b9839344ee53d2a47128/verl/checkpoint_engine/nccl_checkpoint_engine.py#L104-L381) |
| tensor chunk/merge | [`checkpoint_engine/base.py`](https://github.com/verl-project/verl/blob/535c47799b537a0e3602b9839344ee53d2a47128/verl/checkpoint_engine/base.py#L541-L609) |
| vLLM ServerAdapter | [`vllm_rollout.py`](https://github.com/verl-project/verl/blob/535c47799b537a0e3602b9839344ee53d2a47128/verl/workers/rollout/vllm_rollout/vllm_rollout.py#L208-L246) |
| vLLM bucketed IPC | [`bucketed_weight_transfer.py`](https://github.com/verl-project/verl/blob/535c47799b537a0e3602b9839344ee53d2a47128/verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py#L103-L302) |
| vLLM loader 扩展 | [`vllm_rollout/utils.py`](https://github.com/verl-project/verl/blob/535c47799b537a0e3602b9839344ee53d2a47128/verl/workers/rollout/vllm_rollout/utils.py#L232-L398) |
| SGLang ServerAdapter | [`sglang_rollout.py`](https://github.com/verl-project/verl/blob/535c47799b537a0e3602b9839344ee53d2a47128/verl/workers/rollout/sglang_rollout/sglang_rollout.py#L291-L369) |
| SGLang bucket helper | [`sglang_rollout/utils.py`](https://github.com/verl-project/verl/blob/535c47799b537a0e3602b9839344ee53d2a47128/verl/workers/rollout/sglang_rollout/utils.py#L90-L143) |
| GPU 正确性测试 | [`test_correctness_on_gpu.py`](https://github.com/verl-project/verl/blob/535c47799b537a0e3602b9839344ee53d2a47128/tests/checkpoint_engine/test_correctness_on_gpu.py#L33-L81) |

## 参考资料

- [veRL 官方仓库与本文固定 commit](https://github.com/verl-project/verl/tree/535c47799b537a0e3602b9839344ee53d2a47128)
- [Checkpoint Engine README](https://github.com/verl-project/verl/blob/535c47799b537a0e3602b9839344ee53d2a47128/verl/checkpoint_engine/README.md)
- [veRL Engine Workers 文档](https://verl.readthedocs.io/en/latest/workers/engine_workers.html)
- [veRL Model Engine 文档](https://verl.readthedocs.io/en/latest/workers/model_engine.html)
- [veRL Delta Weight Synchronization 文档](https://verl.readthedocs.io/en/latest/advance/delta_weight_sync.html)
- [SGLang 官方 weight-sync utility](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/weight_sync/utils.py)
- [PR #6091：支持单个超大 tensor 的 chunk 传输](https://github.com/verl-project/verl/pull/6091)
- [PR #7005：FSDP2 per-tensor staging 优化](https://github.com/verl-project/verl/pull/7005)
- [Issue #7015：逐参数 all-gather 的剩余开销](https://github.com/verl-project/verl/issues/7015)
