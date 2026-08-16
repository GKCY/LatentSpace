# FSDP：把模型状态拆散到多张 GPU 上

> FSDP（Fully Sharded Data Parallel，全分片数据并行）仍然是**数据并行**：每个 rank 处理不同的数据；它与 DDP 最大的区别，是不让每张 GPU 长期保存一整套参数、梯度和优化器状态，而是在计算某个 FSDP 单元时临时拼出该单元的完整参数，再按 `reshard_after_forward` 策略决定何时重新拆散。

如果只记住一句话：

> **FSDP 用通信换显存：平时每张 GPU 只保留模型状态的一份分片，计算当前 FSDP 单元前用 `all-gather` 临时还原完整参数，反向后用 `reduce-scatter` 聚合梯度并只留下自己负责的那一片。**

截至 2026 年 8 月，PyTorch 官方教程已经把旧的 FSDP1 标记为 deprecated，并推荐使用 FSDP2 的 `fully_shard` API。本文先讲不依赖版本的原理，再以 **FSDP2** 写可运行代码；最后给出 FSDP1 到 FSDP2 的名词对照，方便阅读旧项目。

本文代码以 **PyTorch 2.13 的 stable API 文档**为基线。FSDP2 和 DCP 在较旧版本中可能不存在或签名不同；运行前先检查当前环境：

```bash
python -c "import torch; print(torch.__version__); from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy"
```

若 import 失败，应安装与本机 CUDA 匹配的当前 PyTorch 版本，或按旧项目版本阅读第 19 节的 FSDP1 对照，不要把两套 API 混用。

读完本文，你应该能回答四个问题：

1. FSDP 到底分了什么，哪些显存又没有分？
2. `all-gather`、`reduce-scatter` 为什么会出现在 forward 和 backward 中？
3. 为什么“只在最外层调用一次 `fully_shard(model)`”通常不是好方案？
4. 遇到 OOM、卡死、变慢或 checkpoint 爆内存时，应该先查什么？

---

## 1. 先别急着看 FSDP：训练显存花到哪里了？

训练一个神经网络时，显存不只装“模型权重”。至少要考虑下面几类数据：

| 类别 | 含义 | 是否一定随参数量增长 |
|---|---|---:|
| 参数（parameters） | 当前模型权重 | 是 |
| 梯度（gradients） | 每个参数对应的梯度 | 是 |
| 优化器状态（optimizer states） | 例如 Adam 的一阶矩和二阶矩 | 是 |
| 激活（activations） | forward 为 backward 保存的中间结果 | 与 batch、序列长度、网络结构有关 |
| 临时缓冲区 | GEMM、attention、通信、dtype 转换等临时空间 | 视算子和运行时而定 |
| CUDA allocator 保留空间 | PyTorch 已向 CUDA 申请、但当前不一定被 tensor 使用的空间 | 视执行历史而定 |

为了建立数量级直觉，先做一个**故意简化的 FP32 + Adam 账本**。设模型有 $P$ 个参数：

- 参数：$4P$ bytes；
- 梯度：$4P$ bytes；
- Adam 一阶矩 $m$：$4P$ bytes；
- Adam 二阶矩 $v$：$4P$ bytes。

只算这四项：

$$
M_{\text{model states}}
\approx
4P+4P+4P+4P
=16P\ \text{bytes}
$$

一个 10 亿参数模型，仅这部分就约为 16 GB；70 亿参数则约为 112 GB。这里还**没有**算激活、通信缓冲区和碎片。

实际混合精度训练的账本会因实现而变化：计算参数、常驻参数、梯度归约和优化器状态可以使用不同 dtype，有些优化器还会维护 master weights。因此不要死记“每参数固定多少 bytes”；应把上面的计算当成理解分片收益的模型，并用 profiler 和显存统计验证真实任务。

### 1.1 DDP 为什么解决不了超大模型显存？

DistributedDataParallel（DDP）会在每个 rank 上保存完整模型副本：

```text
4 张 GPU，模型状态记作 [A B C D]

GPU 0: [A B C D] + 处理数据 batch 0
GPU 1: [A B C D] + 处理数据 batch 1
GPU 2: [A B C D] + 处理数据 batch 2
GPU 3: [A B C D] + 处理数据 batch 3
```

每个 rank 分别计算本地梯度，再用 `all-reduce` 得到相同的全局平均梯度。DDP 可以提高吞吐，但参数、梯度和优化器状态依然在每张卡上各放一整份。

所以 DDP 回答的是：

> 模型已经能放进一张 GPU，怎样用更多 GPU 同时处理更多数据？

FSDP 主要回答的是：

> 一整套模型状态放不进一张 GPU，怎样让多张 GPU 分摊它？

---

## 2. FSDP 的核心：长期分片，计算时临时还原

仍用四张 GPU，把一组参数想象成四片：

```text
完整参数组 W = [A B C D]

稳定状态：
GPU 0 只保存 [A]
GPU 1 只保存 [B]
GPU 2 只保存 [C]
GPU 3 只保存 [D]
```

但普通 `Linear`、`LayerNorm` 或 attention 算子并不知道怎样直接使用散落在四张卡上的权重。FSDP 的办法不是改写这些算子，而是在算这一组模块之前，让所有 rank 临时拿到完整参数：

```text
all-gather 前：
GPU 0: [A]    GPU 1: [B]    GPU 2: [C]    GPU 3: [D]

all-gather 后：
GPU 0: [A B C D]
GPU 1: [A B C D]
GPU 2: [A B C D]
GPU 3: [A B C D]
```

每个 rank 随后用相同的完整权重处理自己的不同数据。当前模块算完后，临时收集来的其他分片可以释放，只保留自己的那片。

因此，下面两句话并不矛盾：

- **FSDP 平时不在每张 GPU 上保存完整模型；**
- **计算某个 FSDP 单元时，每张 GPU 会临时拥有这个单元的完整参数。**

这也是 FSDP 的能力边界：如果**单个不可再拆的层**本身都无法在一张 GPU 上临时展开并计算，单靠 FSDP 仍然不够，需要 Tensor Parallel（TP）等把一次算子内部也切开。

后文会频繁使用三个动词：

| 术语 | 本文中的含义 |
|---|---|
| shard | 把完整 tensor 切开，每个 rank 只长期保存一片 |
| unshard | 通过 all-gather 临时还原完整参数 |
| reshard | 当前计算结束后释放其他 rank 的分片，重新只留本地片 |

“FSDP 单元”“参数组”或“通信组”指一次 `fully_shard` 调用所管理的那批参数。它是 all-gather/reduce-scatter 的边界，不等于整个 process group；前者回答“这次传哪些参数”，后者回答“哪些 rank 一起通信”。

### 2.1 FSDP 到底分片了什么？

完整分片模式下，FSDP 分摊的是三类模型状态：

1. **参数分片**：每个 rank 长期只保存一部分参数；
2. **梯度分片**：反向后每个 rank 只留下自己那部分参数对应的全局梯度；
3. **优化器状态分片**：每个 rank 的优化器只为本地参数分片维护状态并更新这一片。

FSDP **不会自动分片一切东西**：

- 激活通常仍属于各 rank 的本地 mini-batch；
- PyTorch FSDP2 不分片 buffers；
- 数据集不会因为套上 FSDP 就自动分给不同 rank；
- 某个当前正在计算的 FSDP 单元仍要临时还原；
- CUDA kernel workspace、通信 buffer 和 allocator 碎片依然存在。

因此，“8 张卡就一定把总显存除以 8”是错误的。更接近真实情况的粗略公式是：

$$
M_{\text{rank}}
\approx
\frac{
M_{\text{param}}+M_{\text{grad}}+M_{\text{optim}}
}{N}
+M_{\text{activation}}
+M_{\text{current full unit}}
+M_{\text{temporary}}
$$

其中 $N$ 是分片组大小。只有第一项接近按 $N$ 分摊，后面几项并不会自动除以 $N$。

继续使用前面的 FP32 + Adam 教学账本：10 亿参数的持久模型状态约为 16 GB。若在 8 个 rank 上完整分片，理想稳定状态约为每 rank 2 GB：

| 方法 | 参数/卡 | 梯度/卡 | Adam 状态/卡 | 持久模型状态/卡 |
|---|---:|---:|---:|---:|
| DDP | 4 GB | 4 GB | 8 GB | 16 GB |
| FSDP 完整分片，8 ranks | 0.5 GB | 0.5 GB | 1 GB | 2 GB |

这里使用十进制 GB，并且仍然没有计算激活、当前完整参数组和临时缓冲区。因此“2 GB”是稳定模型状态的直觉，不是“2 GB 显卡就能训练 10 亿参数模型”的承诺。

---

## 3. 先认清 rank、world size 和 process group

PyTorch 分布式训练通常采用“一张 GPU 对应一个进程”的方式。几个词必须分清：

| 名词 | 含义 | 8 卡单机示例 |
|---|---|---:|
| process / worker | 一个训练进程 | 共 8 个 |
| rank | 进程在整个通信组中的全局编号 | 0～7 |
| local rank | 进程在本机内的编号 | 0～7 |
| world size | 通信组中的总进程数 | 8 |
| process group | 哪些 rank 参加一组 collective | 默认是全部 8 个 |

两机、每机 8 卡时，`WORLD_SIZE=16`；每台机器上的 `LOCAL_RANK` 都是 0～7，但全局 `RANK` 是 0～15。

`torchrun` 会为每个进程设置这些环境变量。程序需要做两件关键的事：

```python
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl")
```

不要用全局 `RANK` 直接选择本机 GPU，也不要假设 elastic restart 后 rank 永远不变。对 CUDA GPU，NCCL 是通常的通信后端。

### 3.1 FSDP 仍然是数据并行

假设每个 rank 的本地 batch size 为 4，数据并行 world size 为 8，梯度累积 2 次，那么：

$$
B_{\text{global}}
=
B_{\text{per-rank}}
\times N_{\text{data-parallel}}
\times N_{\text{accumulation}}
=4\times8\times2=64
$$

每个 rank 必须拿到不同的数据分片。若八张 GPU 每次都读取完全相同的四条样本，那么在确定性计算下只是重复同一份梯度；即使 Dropout 等随机算子让梯度不逐值相同，有效数据覆盖仍没有增加，同样是在浪费数据并行能力。

---

## 4. 两个通信原语：all-gather 与 reduce-scatter

理解 FSDP 不需要先学完所有 collective，只要真正弄懂这两个。

### 4.1 `all-gather`：每个人交出一片，每个人拿到全集

输入：rank $i$ 持有第 $i$ 个参数分片。

输出：每个 rank 都获得按顺序拼接的完整参数。

```text
rank 0: [A] ─┐
rank 1: [B] ─┼─ all-gather ─> 每个 rank 都得到 [A B C D]
rank 2: [C] ─┤
rank 3: [D] ─┘
```

FSDP 在 forward 前需要它，是因为当前模块要用完整参数计算；若 forward 后立即重新分片，那么 backward 计算该模块梯度前还要再做一次。

### 4.2 `reduce-scatter`：先聚合所有人的值，再把结果切开

每个 rank 都会基于自己的本地数据得到当前参数组的本地梯度。我们既要把各 rank 梯度聚合起来，又不想让每个 rank 最后保留完整梯度。

`reduce-scatter` 把两件事合成一个 collective：

1. 对各 rank 的对应梯度做 reduce（训练语义下通常得到按数据并行组正确缩放的聚合梯度）；
2. 把聚合结果 scatter，让每个 rank 只留下其中一片。

```text
rank 0 本地梯度: [gA⁰ gB⁰ gC⁰ gD⁰]
rank 1 本地梯度: [gA¹ gB¹ gC¹ gD¹]
rank 2 本地梯度: [gA² gB² gC² gD²]
rank 3 本地梯度: [gA³ gB³ gC³ gD³]
                           │
                    reduce-scatter
                           │
rank 0 留下: [mean(gA)]
rank 1 留下: [mean(gB)]
rank 2 留下: [mean(gC)]
rank 3 留下: [mean(gD)]
```

这里的 `mean` 是为了表达“各 rank 本地 batch 等大、每个本地 loss 都按样本取 mean”时的最终训练语义；底层 collective 可以通过求和与缩放组合实现。

若 rank $r$ 有 $n_r$ 个有效样本或 token，本地 mean loss 不能再被各 rank 等权平均。真正的全局 mean 应按 $n_r$ 加权：

$$
L_{\text{global}}
=
\frac{\sum_r n_rL_r}{\sum_r n_r}
$$

变长序列、padding mask 和最后一个不完整 batch 都会触发这个问题。工程上可让各 rank 的有效计数一致，或显式聚合有效元素总数并按它缩放 loss/gradient。

### 4.3 用八个数亲手走一遍

设两个 rank 分别保存：

$$
W^{(0)}=[w_0,w_1,w_2,w_3],\qquad
W^{(1)}=[w_4,w_5,w_6,w_7]
$$

`all-gather` 后，两边都临时得到完整的 $W=[w_0,\ldots,w_7]$。两个 rank 用不同数据算出的本地梯度分别是：

$$
G^{(0)}=[1,2,3,4,5,6,7,8]
$$

$$
G^{(1)}=[10,20,30,40,50,60,70,80]
$$

若跨 rank 取平均，完整全局梯度为：

$$
\bar G
=\frac{G^{(0)}+G^{(1)}}{2}
=[5.5,11,16.5,22,27.5,33,38.5,44]
$$

`reduce-scatter` 不会让两边都长期保存 $\bar G$，而是直接给出：

```text
rank 0: [5.5, 11, 16.5, 22]
rank 1: [27.5, 33, 38.5, 44]
```

rank 0 只负责更新 $w_0$～$w_3$，所以只需要前半段梯度；rank 1 同理。下面这个 CPU 脚本没有真正启动分布式进程，但能模拟两种 collective 的张量语义，任何装有 PyTorch 的机器都可以运行：

```python
import torch

world_size = 2

# 参数分片：cat 模拟 all-gather。
full_parameter = torch.arange(8, dtype=torch.float32)
parameter_shards = full_parameter.chunk(world_size)
unsharded_on_each_rank = [
    torch.cat(parameter_shards) for _ in range(world_size)
]

assert all(
    torch.equal(parameter, full_parameter)
    for parameter in unsharded_on_each_rank
)

# 每个 rank 基于不同数据得到一份本地完整梯度。
local_gradients = [
    torch.arange(1, 9, dtype=torch.float32),
    10 * torch.arange(1, 9, dtype=torch.float32),
]

# stack + mean + chunk 模拟“平均归约后再 scatter”。
global_gradient = torch.stack(local_gradients).mean(dim=0)
gradient_shards = global_gradient.chunk(world_size)

print("rank 0 gradient shard:", gradient_shards[0])
print("rank 1 gradient shard:", gradient_shards[1])
assert torch.equal(torch.cat(gradient_shards), global_gradient)
```

真实 collective 不会按这几行 Python 实现；这里模拟的是输入输出语义。

### 4.4 它与 DDP 的 `all-reduce` 有什么关系？

可以在语义上把一次 all-reduce 拆成：

```text
all-reduce ≈ reduce-scatter + all-gather
```

DDP 在 backward 后需要让每个 rank 都得到完整的聚合梯度，所以会完成整个 all-reduce。

FSDP 的 optimizer 只更新本 rank 负责的参数分片，因此 backward 后做到 reduce-scatter 就够了。被省掉的“梯度 all-gather”并没有永远消失：下一轮 forward 需要计算时，FSDP 会 all-gather **更新后的参数分片**。

这就是理解 FSDP 最漂亮的一种方式：

> 它把 DDP 中“反向结束时立刻让所有 rank 都拥有完整结果”的思路，改成“只留下眼下需要更新的分片，等下一次真正计算时再拼参数”。

---

## 5. 一次训练迭代到底发生了什么？

假设模型由三个 Transformer block 组成，并且每个 block 是一个 FSDP 单元。忽略预取和通信/计算重叠，一次迭代可画成：

```text
初始化后的稳定状态：每个 rank 只有 W1/N、W2/N、W3/N

forward:
  all-gather W1 -> 算 block 1 -> 释放其他 W1 分片
  all-gather W2 -> 算 block 2 -> 释放其他 W2 分片
  all-gather W3 -> 算 block 3 -> 释放其他 W3 分片

backward（顺序相反）:
  all-gather W3 -> 算 dW3 -> reduce-scatter dW3 -> 释放完整 W3 与完整 dW3
  all-gather W2 -> 算 dW2 -> reduce-scatter dW2 -> 释放完整 W2 与完整 dW2
  all-gather W1 -> 算 dW1 -> reduce-scatter dW1 -> 释放完整 W1 与完整 dW1

optimizer step:
  rank i 用自己的梯度分片更新自己的参数分片
  Adam 的 m、v 也只为这个分片存在
```

如果某个单元设置为 forward 后**不立即 reshard**，它的完整参数会一直留到 backward，从而省掉 backward 前的第二次 all-gather，但显存更高。这就是 FSDP 中反复出现的主旋钮：

| 选择 | 显存 | 通信 | 常见用途 |
|---|---:|---:|---|
| forward 后 reshard | 更低 | backward 前多一次 all-gather | 内存优先 |
| forward 后不 reshard | 更高 | 少一次 all-gather | 吞吐优先 |

FSDP2 的 `reshard_after_forward=None` 有一个有意设计的默认行为：非 root FSDP 模块按 `True` 处理，root 模块按 `False` 处理。原因是 root 剩余参数通常在 forward 末尾和 backward 开始处相邻使用，保留它们往往比较划算。

### 5.1 FSDP 为什么可能比 DDP 慢？

FSDP 省下显存的代价是参数 all-gather。以 ring collective 的粗略通信量估算：

- DDP 每轮主要有一次梯度 all-reduce；
- 完整 reshard 的 FSDP 单元通常有 forward all-gather、backward all-gather、gradient reduce-scatter；
- 不 reshard 则可少一次参数 all-gather，但多占显存。

FSDP 的目标首先是让更大的模型或 batch 能跑起来，不保证在模型本来就轻松放得下时比 DDP 更快。实际性能取决于层大小、网络拓扑、batch、计算强度、wrap 边界和通信能否被计算覆盖。

 ### 5.2 一个粗略的通信量账本

设某个参数组有 $Q$ bytes，采用 ring collective，并只计算大 tensor 的理论 payload。按每个 rank 的单向发送量计算（接收量同量），一次 all-gather 或 reduce-scatter 大约是：

$$
\frac{N-1}{N}Q
$$

于是，在“参数和梯度字节数相同”的教学假设下：

$$
V_{\text{DDP all-reduce}}
\approx
2\frac{N-1}{N}Q
$$

$$
V_{\text{FSDP full shard}}
\approx
3\frac{N-1}{N}Q
$$

后一个式子的三段是 forward 参数 all-gather、backward 参数 all-gather、梯度 reduce-scatter，因此约为 DDP 状态同步 payload 的 1.5 倍。若 forward 后不 reshard，则少一次参数 all-gather，粗略回到两个 phase。

这只是建立直觉的近似，并不是网络监控工具一定显示的收发字节数。实际 collective 算法、分块、拓扑和并发都会影响结果；若参数 all-gather 使用 BF16、梯度 reduce-scatter 使用 FP32，还必须分别按 2 bytes 和 4 bytes 计算，不能继续套同一个 $Q$。

---

## 6. Wrap 边界为什么决定成败？

FSDP 不会神奇地逐个算子猜出最佳通信边界。每次调用 `fully_shard(module)`，都会创建一个参数通信组：该模块中尚未被更内层 `fully_shard` 管理的参数会被归到一起。每次 unshard 这组参数时使用一个 all-gather collective，backward 梯度使用一个 reduce-scatter collective；若 forward 后 reshard，同一迭代会分别在 forward 和 backward 前 unshard，因此会有两次参数 all-gather。

### 6.1 只 shard 最外层为什么不好？

假设模型有 100 层，却只做：

```python
fully_shard(model)
```

那么所有参数可能落在一个巨大的通信组里：

```text
all-gather(第 1～100 层全部参数)
-> 依次计算 100 层
-> reduce-scatter(全部梯度)
```

后果是：

- 计算期间要临时容纳整个模型参数，峰值显存很高；
- all-gather 是一个暴露在计算前的大阻塞操作；
- 没有“算第 $i$ 层时预取第 $i+1$ 层”的流水重叠机会。

### 6.2 切得越细越好吗？

也不是。把每一个很小的 `Linear` 都单独分组，会产生大量小 collective，启动延迟、hook 调度和同步开销可能压过收益。

对 Transformer，最自然的起点通常是：

```text
一个 Transformer block = 一个 FSDP 通信组
```

然后最后再对 root 调用一次 `fully_shard(model)`，让 embedding、final norm、output head 等没有被子模块接管的剩余参数也进入分片组。

### 6.3 为什么必须 bottom-up？

FSDP2 应先处理子模块，再处理根模块：

```python
for block in model.blocks:
    fully_shard(block)
fully_shard(model)
```

当 root 最后被处理时，已经属于 block 通信组的参数会被排除，root 只接管剩余参数。如果顺序反过来，root 已经先拿走所有参数，就无法得到预期的逐层分组。

选择边界时可以用三个问题判断：

1. 当前组完整展开时能否轻松放进单卡？
2. collective 是否大到足以有效利用带宽，而不是被启动延迟支配？
3. 当前组的计算是否足够覆盖下一组通信？

---

## 7. FSDP、DDP、ZeRO、TP、PP 到底是什么关系？

### 7.1 与 ZeRO 的关系

DeepSpeed ZeRO 用 Stage 1～3 描述逐步分片模型状态：

| 方案 | 参数 | 梯度 | 优化器状态 |
|---|---|---|---|
| DDP | 复制 | 复制 | 复制 |
| ZeRO Stage 1 | 复制 | 复制 | 分片 |
| ZeRO Stage 2 | 复制 | 分片 | 分片 |
| ZeRO Stage 3 | 分片 | 分片 | 分片 |
| FSDP 完整分片 | 分片 | 分片 | 分片 |

所以，从“分了哪些模型状态”看，FSDP 完整分片与 ZeRO Stage 3 是同一类思想。它们的 API、内部表示、调度策略和生态实现并不完全相同，不能把两个名字机械地当成同一个软件组件。

### 7.2 与 Tensor Parallel 的区别

FSDP 的每个 rank 处理不同数据；计算当前层时，每个 rank 通常临时拿到该层完整参数，并独立完成该层计算。

TP 则把一个矩阵乘法或张量本身切给多个 rank 协作完成，同一条样本会跨 rank 流动。

```text
FSDP：不同数据 × 临时完整的当前层
TP：  同一批数据 × 当前层内部也切开计算
```

因此：

- 模型状态总量放不下一张卡，但最大单层能放下：优先考虑 FSDP；
- 连最大单层/单个大算子都放不下，或 FSDP 扩展性遇到通信瓶颈：考虑 TP；
- 超大规模训练常组合 TP + FSDP，再视情况加 Pipeline Parallel（PP）。

### 7.3 与 Pipeline Parallel 的区别

PP 把连续层放在不同 stage，让 micro-batch 像流水线一样经过各 stage。它切的是模型深度，并会遇到 pipeline bubble 和调度问题。

FSDP、TP、PP 不是互斥选项，而是可以映射到不同的设备网格维度：

```text
data-parallel / FSDP 维：不同训练样本
tensor-parallel 维：    同一层内部的张量切分
pipeline-parallel 维：  不同连续层/stage
```

---

## 8. FSDP2 的最小可运行示例

下面是一个面向学习的完整脚本。它使用两张或更多支持 BF16 的 NVIDIA GPU、`torchrun`、NCCL 和 FSDP2。若 GPU 不支持 BF16，请先去掉 `mp_policy`，用 FP32 跑通流程，再根据硬件选择精度。模型很小，所以它不会展示显著的显存收益；它的目标是把正确的程序骨架一次写全。

将代码保存为 `fsdp_minimal.py`：

```python
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard


class Block(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.fc2 = nn.Linear(4 * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.fc1(self.norm(x)))
        return x + self.fc2(hidden)


class TinyModel(nn.Module):
    def __init__(self, dim: int = 1024, depth: int = 8) -> None:
        super().__init__()
        self.input = nn.Linear(dim, dim)
        self.blocks = nn.ModuleList([Block(dim) for _ in range(depth)])
        self.output = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x)
        for block in self.blocks:
            x = block(x)
        return self.output(x)


def main() -> None:
    # torchrun 会设置 LOCAL_RANK、RANK、WORLD_SIZE、MASTER_ADDR、MASTER_PORT。
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)

    try:
        # 各 rank 必须从相同初始权重开始；这里在建模前使用同一个 seed。
        torch.manual_seed(0)
        model = TinyModel()

        # 常驻分片保持原始 FP32；all-gather 后以 BF16 做 forward/backward；
        # 梯度用 FP32 reduce-scatter，提高数值稳健性。
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )

        # 必须 bottom-up：先每个 block，最后 root 接管剩余参数。
        for block in model.blocks:
            fully_shard(block, mp_policy=mp_policy)
        fully_shard(model, mp_policy=mp_policy)

        # 必须在 fully_shard 之后构造 optimizer，让它持有 DTensor 参数分片。
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

        # 每个 rank 使用不同 seed，模拟不同的数据分片。
        data_rng = torch.Generator(device=device)
        data_rng.manual_seed(1234 + rank)

        torch.cuda.reset_peak_memory_stats(device)

        for step in range(10):
            x = torch.randn(
                4, 128, 1024, device=device, generator=data_rng
            )
            target = torch.randn(
                4, 128, 1024, device=device, generator=data_rng
            )

            prediction = model(x)  # 不要直接调用 model.forward(x)
            loss = F.mse_loss(prediction.float(), target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # 只为了日志：把各 rank loss 求平均，再仅由 rank 0 打印。
            mean_loss = loss.detach().float()
            dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
            mean_loss /= dist.get_world_size()
            if rank == 0:
                peak_gib = torch.cuda.max_memory_allocated(device) / 1024**3
                print(
                    f"step={step:02d} "
                    f"loss={mean_loss.item():.6f} "
                    f"peak_allocated={peak_gib:.2f} GiB"
                )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

单机双卡启动：

```bash
torchrun --standalone --nproc-per-node=2 fsdp_minimal.py
```

单机八卡只需改成：

```bash
torchrun --standalone --nproc-per-node=8 fsdp_minimal.py
```

两机、每机八卡时，两台机器使用相同的 `MASTER_ADDR`、`MASTER_PORT` 和训练参数，只改变 `--node-rank`：

```bash
# node 0
torchrun \
  --nnodes=2 \
  --nproc-per-node=8 \
  --node-rank=0 \
  --master-addr=10.0.0.1 \
  --master-port=29500 \
  fsdp_minimal.py

# node 1
torchrun \
  --nnodes=2 \
  --nproc-per-node=8 \
  --node-rank=1 \
  --master-addr=10.0.0.1 \
  --master-port=29500 \
  fsdp_minimal.py
```

### 8.1 逐行理解最重要的五处

第一，先绑定本地 GPU，再初始化/使用 CUDA collective：

```python
torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl")
```

第二，各 rank 的**初始模型必须一致**。示例在建模前设置相同随机种子。加载 checkpoint 时，则应使用 DCP 等方式正确分发权重。

第三，`fully_shard` 是原地应用的。FSDP2 不像 FSDP1 那样用一个外层 wrapper 替换模型；它把参数转成按维度 0 分片的 `DTensor`，并给原模块注册通信 hook。参数的 fully qualified name 保持不变。

第四，optimizer 必须在 `fully_shard` 之后创建：

```python
fully_shard(model)
optimizer = AdamW(model.parameters(), ...)
```

此时 optimizer 拿到的是 DTensor 参数分片，所以它自然只为本地分片建立状态。顺序反过来会让 optimizer 持有错误的参数引用或破坏预期分片语义。

第五，要调用：

```python
model(x)
```

而不是：

```python
model.forward(x)
```

FSDP 依赖模块调用过程中的 pre/post hooks 在恰当时机 unshard/reshard。直接调 `forward` 会绕过默认 hook。若确实要把其他方法作为 forward 入口，应显式使用官方的 `register_fsdp_forward_method`。

---

## 9. FSDP2 为什么使用 DTensor？

FSDP2 在**分片稳定态**把 `model.parameters()` 注册为 `DTensor`。这包括刚完成 `fully_shard`、已经显式 `reshard()`，以及正常 backward 结束后的时刻。DTensor 不仅保存本 rank 的本地数据，还携带两类分布信息：

- `device_mesh`：哪些设备共同表示这个逻辑 tensor；
- `placements`：tensor 在每个 mesh 维度上是 `Shard` 还是 `Replicate`。

进入 forward/backward 前，hook 会把当前参数组 unshard 并暂时注册为普通完整 `Tensor`。若 `reshard_after_forward=False`，forward 与 backward 之间也会继续是普通完整 Tensor；root 模块默认就是这种策略。因此下面的观察代码应放在刚 `fully_shard` 后或一次完整迭代结束后，而不要任意插到 forward 与 backward 之间。也可以在相应 `FSDPModule` 上显式调用 `reshard()`；注意它不是递归操作，观察整个嵌套模型前要确保每个参数组都已回到分片态。

一维 FSDP mesh 上，参数默认沿维度 0 分片。可以在分片稳定态这样观察：

```python
from torch.distributed.tensor import DTensor

# 放在 fully_shard 完成后、第一次 forward 之前。
for name, param in model.named_parameters():
    assert isinstance(param, DTensor)
    print(
        name,
        "global shape:", tuple(param.shape),
        "local shape:", tuple(param.to_local().shape),
        "placements:", param.placements,
    )
```

注意：`param.shape` 表示逻辑上的全局 shape，`param.to_local().shape` 才是本 rank 实际持有的分片 shape。

FSDP1 会把一组参数 flatten、拼接，再切成 `FlatParameter`；FSDP2 改为 per-parameter DTensor 分片，原参数名更稳定，冻结参数、优化器、梯度裁剪和分片 checkpoint 的组合也更直观。

不要把 `to_local()` 当作普通完整参数使用。它只是一片，适合调试和实现明确理解 sharding 语义的高级逻辑。

---

## 10. 混合精度：三个 dtype 不要混为一谈

FSDP2 的 `MixedPrecisionPolicy` 最重要的两个参数是：

```python
MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
)
```

它表达的是：

| 时机 | dtype | 目的 |
|---|---|---|
| 平时保存的参数分片、optimizer step | 参数原始 dtype，示例为 FP32 | 保留更新精度 |
| unshard 后的参数与 forward/backward 计算 | BF16 | 降显存、提吞吐、减少 all-gather 字节数 |
| 梯度 reduce-scatter | FP32 | 提高跨 rank 累加的数值稳健性 |

若 `reduce_dtype=None` 且设置了 `param_dtype`，归约通常使用计算 dtype。把它显式设为 FP32 会增加通信字节数，但可能改善数值稳定性。这不是固定答案，应以 loss、梯度范数和吞吐测量为准。

FSDP 的 mixed precision 是**模块边界级**策略：进入 FSDP 模块时统一转换参数/输入。`torch.autocast` 则更像算子级策略，由每类算子决定使用何种精度。二者的语义不同，不要把配置机械重复叠加；采用框架集成时先查它究竟使用哪一种。

经验上：

- 支持 BF16 的训练 GPU 上，优先把 BF16 作为起点；
- BF16 通常不需要 loss scaling；
- FP16 的指数范围更小，常需要 `GradScaler`；
- LayerNorm、loss、softmax 等敏感部分是否保持 FP32，应结合模型实现和验证结果决定；
- “训练不报错”不等于数值等价，应与单卡或 DDP 基线对齐 loss 曲线。

---

## 11. 数据分片与 `DistributedSampler`

FSDP 只管模型状态和相关通信，不会替你切数据。Map-style dataset 的典型写法是：

```python
from torch.utils.data import DataLoader, DistributedSampler

sampler = DistributedSampler(
    dataset,
    num_replicas=dist.get_world_size(),
    rank=dist.get_rank(),
    shuffle=True,
)
loader = DataLoader(
    dataset,
    batch_size=per_rank_batch_size,
    sampler=sampler,
    shuffle=False,  # 已由 sampler 负责 shuffle
    num_workers=8,
    pin_memory=True,
)

for epoch in range(num_epochs):
    sampler.set_epoch(epoch)
    for batch in loader:
        ...
```

几个常见错误：

- 同时给 `sampler` 和 `shuffle=True`；
- 忘记每个 epoch 调 `sampler.set_epoch(epoch)`，导致每轮 shuffle 顺序相同；
- 不同 rank 的 DataLoader 迭代次数不同，某些 rank 先退出，其他 rank 卡在 collective；
- 把 per-rank batch 当成 global batch，扩卡后无意中把有效 batch 和学习率一起改变；
- 只在 rank 0 执行包含 FSDP forward/state-dict collective 的验证或保存逻辑。

最后一条尤其重要：**只让 rank 0 写日志/文件可以，只让 rank 0 单独进入 collective 不可以。** 所有参与该 process group 的 rank 必须以一致顺序执行通信。

---

## 12. 梯度累积与梯度裁剪

### 12.1 梯度裁剪

FSDP2 的参数是 DTensor，官方支持直接使用：

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

DTensor 会处理计算全局 norm 所需的跨 rank 语义。不要只对 `param.to_local()` 的局部分片算 norm，然后把它误认为完整模型的梯度范数。

### 12.2 梯度累积

最简单但通信较多的办法，是每个 micro-batch 都正常同步梯度。为了只在最后一个 micro-batch 同步，FSDP2 提供 `set_requires_gradient_sync`，相当于 FSDP1 的 `no_sync()`：

```python
optimizer.zero_grad(set_to_none=True)

for micro_step, batch in enumerate(micro_batches):
    is_last = micro_step == accumulation_steps - 1
    model.set_requires_gradient_sync(is_last)

    loss = compute_loss(model, batch) / accumulation_steps
    loss.backward()

# 恢复默认状态，避免下一轮意外不通信。
model.set_requires_gradient_sync(True)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
```

这里把 loss 除以累积步数，只有在**每个 micro-batch 的样本数或有效 token 数相同，并且 `compute_loss` 返回本 micro-batch 的 mean** 时，才与一个大 batch 的全局 mean 严格对齐。

若最后一个 micro-batch 更小或每批有效 token 数不同，应按有效元素数加权：先明确累积窗口的总有效计数，再让每批贡献与其计数成比例的梯度；或者让 loss 返回 sum，并在 optimizer step 前按窗口总计数统一缩放。不能无条件把每个局部 mean 都除以固定的 `accumulation_steps`。

关闭梯度同步会改变中间梯度的存储与生命周期，往往增加峰值显存。FSDP2 还提供 `set_reshard_after_backward(False)`，可在连续 micro-batch 之间保留完整参数以少做 all-gather，但会进一步以显存换通信。先保证数学等价，再逐项测量，不要一次打开所有“优化”开关。

---

## 13. Checkpoint：不要把分片模型当普通模型保存

FSDP2 的 `model.state_dict()` 默认包含 DTensor 分片。训练恢复通常应使用 PyTorch Distributed Checkpoint（DCP），由所有 rank 并行保存/加载，并保留分片信息。

一个精简的 model + optimizer 状态包装器如下：

```python
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful


class AppState(Stateful):
    def __init__(self, model, optimizer) -> None:
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        model_sd, optim_sd = get_state_dict(self.model, self.optimizer)
        return {"model": model_sd, "optimizer": optim_sd}

    def load_state_dict(self, state_dict) -> None:
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optimizer"],
        )
```

保存时，所有 rank 都调用：

```python
app_state = AppState(model, optimizer)
dcp.save(
    {"app": app_state},
    checkpoint_id="checkpoints/step_1000",
)
```

恢复时，先用同样结构创建并 shard 模型，再创建 optimizer，最后所有 rank 都调用：

```python
app_state = AppState(model, optimizer)
dcp.load(
    {"app": app_state},
    checkpoint_id="checkpoints/step_1000",
)
```

DCP 的重要特性是可以并行 I/O，并可在加载时根据新的集群拓扑重新分片。它通常会生成多个文件，而不是一个可以随手移动的 `.pt` 文件。多机训练时，示例中的目录必须是所有 rank 可访问的共享存储，或改用适合目标存储系统的 `StorageWriter` / `StorageReader`。

DCP 官方目前不承诺 state-dict checkpoint 跨 PyTorch 版本向后兼容。生产任务应固定 PyTorch 版本，把“真正恢复一次并继续跑若干步”纳入发布前演练，而不只是检查目录里出现了文件。

### 13.1 导出普通单文件权重

如果目的是发布模型或单进程推理，可以请求完整 state dict，并只让 rank 0 写文件：

```python
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)

full_state_dict = get_model_state_dict(
    model,
    options=StateDictOptions(
        full_state_dict=True,
        cpu_offload=True,
    ),
)

if dist.get_rank() == 0:
    torch.save(full_state_dict, "model_state_dict.pt")
```

`full_state_dict=True` 与 `cpu_offload=True` 同时启用时，非 0 rank 会得到空字典，从而避免同一节点在 CPU 上复制多份完整权重。虽然最终只有 rank 0 拿到内容并写文件，但所有 rank 仍要进入 `get_model_state_dict`，因为构造完整权重涉及 collective。

把这个普通单文件重新加载到 FSDP2 模型时，可以只让 rank 0 读盘，再让所有 rank 一起调用 `set_model_state_dict`，边广播边按目标 DTensor 布局分片：

```python
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    set_model_state_dict,
)

if dist.get_rank() == 0:
    full_state_dict = torch.load(
        "model_state_dict.pt",
        map_location="cpu",
        weights_only=True,
    )
else:
    full_state_dict = {}

set_model_state_dict(
    model,
    full_state_dict,
    options=StateDictOptions(
        full_state_dict=True,
        broadcast_from_rank0=True,
    ),
)
```

### 13.2 Checkpoint OOM 的典型原因

最危险的做法是让每个 rank 同时把整个模型 materialize 到 GPU 或 CPU：

```text
每张 GPU 拼完整权重 -> GPU OOM
同一节点的 8 个 rank 各放一份完整 CPU 权重 -> 主机 RAM OOM
8 个 rank 同时 torch.save 同一路径 -> 文件竞争或损坏
```

训练恢复优先保存分片 checkpoint；必须导出完整权重时，使用 CPU offload、rank 0 写盘，并估算主机内存和磁盘空间。

除模型和 optimizer 外，真正可恢复的训练还应保存：

- 当前 step / epoch；
- scheduler 状态；
- 随机数生成器状态；
- data sampler 或 dataloader 进度；
- AMP scaler（若使用）；
- 影响训练语义的配置与代码版本。

---

## 14. 大模型初始化：GPU OOM 可能发生在训练之前

小示例可以让每个进程先在 CPU 构造完整模型，再由 `fully_shard` 移到设备。但对百亿参数模型，一台机器上 8 个进程各建一份完整 CPU 模型，本身就可能耗尽主机内存。

常见办法是先在 `meta` device 上只创建 shape，不分配真实存储：

```python
with torch.device("meta"):
    model = HugeModel(config)

for block in model.blocks:
    fully_shard(block)
fully_shard(model)

# 随机初始化场景：分片后再为本 rank 分配真实存储。
model.to_empty(device="cuda")

# 这是 HugeModel 自己实现的递归初始化方法，不是 nn.Module 的通用保证。
model.reset_parameters()
```

普通 `nn.Module` 并不保证根模块有 `reset_parameters()`，也不保证某个自定义实现会正确递归初始化所有叶子模块。真实模型应提供与架构匹配的初始化函数，并检查每个本地分片都已脱离 meta device；不能机械复制最后一行。

预训练权重加载不应简单地让所有 rank 各读一份完整 checkpoint。可使用 DCP 的 state-dict API，让 rank 0 读取完整权重后分片并广播，或直接从 distributed checkpoint 按目标 mesh 加载。

需要区分三个 OOM 时刻：

| OOM 时刻 | 最可能的问题 |
|---|---|
| 模型构造/加载时 | 每个进程复制完整 CPU/GPU 模型；没有使用 meta 或分片加载 |
| 第一次 forward | FSDP 单元过大；root-only；输入/激活太大 |
| 第一次 optimizer step | optimizer 在 shard 前创建；额外 master copy；状态首次懒初始化 |

---

## 15. 激活检查点与 CPU offload：解决的是不同问题

### 15.1 Activation checkpointing

FSDP 主要减少参数、梯度、优化器状态；activation checkpointing 主要减少为 backward 保存的激活。后者通过在 backward 时重算部分 forward，拿计算换显存。

二者经常组合：

```text
FSDP：                 分摊模型状态
activation checkpoint：压缩激活峰值
```

如果长序列训练在 forward/backward 仍 OOM，即使参数分片看起来很理想，也要检查激活。注意 activation checkpointing 与“保存模型 checkpoint”只是同名，完全是两件事。

### 15.2 CPU offload

FSDP2 的 `CPUOffloadPolicy` 可以把参数分片、梯度分片和优化器状态放到 CPU，需要时再在 CPU 与 GPU 之间搬运：

```python
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard

fully_shard(
    module,
    offload_policy=CPUOffloadPolicy(pin_memory=True),
)
```

它可以进一步省 GPU 显存，但会增加：

- PCIe/NVLink-C2C 传输；
- CPU optimizer step 时间；
- 主机内存占用；
- pinned memory 压力。

所以 CPU offload 更像“模型否则根本跑不起来”的容量手段，而不是默认性能优化。先尝试合理 wrap、BF16 和 activation checkpointing，再评估是否值得 offload。

---

## 16. HSDP：为什么不一定在全世界范围内分片？

多机训练中，机内 NVLink/NVSwitch 通常远快于机间网络。若在 64 张 GPU 上做纯 FSDP，参数 all-gather 会跨所有节点，通信可能难以隐藏。

Hybrid Sharded Data Parallel（HSDP）使用二维 mesh：

```text
机内 shard：     一台机器的 8 张卡分摊模型状态
机间 replicate：不同机器复制同一套“8 卡分片布局”
```

这样，频繁的参数 all-gather/reduce-scatter 主要发生在高速机内链路；跨机器则同步复制组之间的梯度结果。代价是模型状态只按机内 shard size 分摊，而不是按总 GPU 数分摊。

```python
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

# 2 台机器，每台 8 卡；维度 0 做 replicate，维度 1 做 shard。
mesh_2d = init_device_mesh(
    "cuda",
    (2, 8),
    mesh_dim_names=("replicate", "shard"),
)

for block in model.blocks:
    fully_shard(block, mesh=mesh_2d)
fully_shard(model, mesh=mesh_2d)
```

选择纯 FSDP 还是 HSDP，本质是在权衡：

```text
更大的 shard group -> 更省显存，但 collective 范围更大
更小的 shard group -> 多复制模型状态，但可把高频通信留在快链路
```

不要只看 GPU 数量，要看 GPU 拓扑：哪些卡共享 NVLink、哪些通信会经过 InfiniBand 或 Ethernet。

---

## 17. 性能调优：先测，再动旋钮

### 17.1 通信与计算重叠

逐 block 分组后，理想时间线是：

```text
时间 --->

通信流: AG(block 1)  AG(block 2)  AG(block 3) ...
计算流:      FW(block 1)  FW(block 2)  FW(block 3) ...
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                    尽量让下一层通信藏在当前层计算下面
```

FSDP2 默认提供 implicit prefetching。通常先从默认策略开始；只有 profiler 明确显示 CPU 发射过慢或首个 all-gather 暴露时，再考虑 `set_modules_to_forward_prefetch`、`set_modules_to_backward_prefetch` 或手动 `unshard()`。

预取更多层并不免费：它会让多个完整参数组同时驻留，提高峰值显存。

### 17.2 Wrap 粒度

| profiler 现象 | 可能原因 | 调整方向 |
|---|---|---|
| 一个巨大 all-gather，GPU 随后才开始算 | 分组太粗 | 按 Transformer block 等自然单元拆分 |
| 数百个极小 collective，GPU 频繁空转 | 分组太细 | 合并小模块 |
| 通信基本被当前层计算覆盖 | 粒度合适 | 不要为“看起来更高级”继续改 |
| 峰值显存来自同时驻留多个 full unit | 预取过深或不 reshard | 减少预取、启用 reshard |

### 17.3 Batch 与算术强度

本地 batch 太小或序列太短时，每层计算很快，通信难以隐藏；增大本地 batch 可能提升吞吐，却又会增加激活显存。常见做法是配合 BF16、Flash Attention、activation checkpointing，在可接受显存内提高有效计算量。

### 17.4 测显存时看两个数

```python
allocated = torch.cuda.max_memory_allocated() / 1024**3
reserved = torch.cuda.max_memory_reserved() / 1024**3
```

- `allocated`：活跃 tensor 实际占用；
- `reserved`：PyTorch allocator 向 CUDA 保留的总量。

`nvidia-smi` 更接近进程向驱动占住的显存，不等于活跃 tensor 之和。比较配置时应：

1. 先 warm up 若干步；
2. 在同一位置 reset peak stats；
3. 使用相同 batch、序列长度和模型配置；
4. 同时记录 tokens/s、step time、allocated 和 reserved；
5. 不要只凭初始化后的静态显存下结论。

### 17.5 推荐的调优顺序

1. 用单卡或 DDP 小模型验证数据和 loss 正确；
2. 套 FSDP2，按 block bottom-up 分组；
3. 检查 global batch、loss reduction 和学习率语义；
4. 启用 BF16 mixed precision；
5. activation OOM 时加 activation checkpointing；
6. 用 profiler 找暴露的 all-gather/reduce-scatter；
7. 再调 wrap、reshard、prefetch；
8. 跨节点通信成为瓶颈时评估 HSDP 或 TP + FSDP；
9. 容量仍不够且能接受吞吐损失时再考虑 CPU offload。

---

## 18. 排错手册

### 18.1 OOM

先按发生阶段判断，而不是盲目减 batch：

| 发生位置 | 先检查什么 |
|---|---|
| 初始化/加载 | 是否每个 rank 都创建或读取完整模型；是否可用 meta + DCP |
| forward 开头 | 是否只 shard root；单个通信组是否太大 |
| forward 中后段 | activation 是否主导；序列长度与 attention 内存 |
| backward | activation + full 参数 + gradient buffer 是否叠峰；是否预取过多 |
| optimizer 第一步 | optimizer 是否在 shard 前创建；状态是否此时才懒初始化 |
| 保存 checkpoint | 是否在每个 rank materialize 完整 state dict |

一个特别有效的诊断是临时缩短序列长度：

- 显存大幅下降：激活更可能是主因；
- 变化不大：参数组、梯度、优化器状态或通信 buffer 更可疑。

### 18.2 训练卡死

分布式“卡住”最常见的根因不是 NCCL 本身坏了，而是各 rank 进入 collective 的顺序不一致：

- 一个 rank 因异常提前退出；
- 各 rank DataLoader 长度不同；
- 只有 rank 0 做了 FSDP forward 或完整 state dict；
- 数据相关分支让不同 rank 执行不同子模块；
- 有的 rank 跳过 backward，而其他 rank 仍进入梯度 collective；
- 只有某些 rank 跳过 optimizer step——这通常不会当场卡在 collective，但会让参数分片不一致，使后续结果错误；
- 多机网络接口、端口、防火墙或 GPU 映射不一致。

若 forward 或 backward 中途抛出异常，而程序捕获异常后还想继续下一批，FSDP2 当前迭代的内部状态已经不再可靠。所有 rank 应协调放弃该批及其累积梯度，并在 root FSDP 模块调用：

```python
model.reset_iter_state()
```

然后再开始一个完整的新迭代。不能让部分 rank 恢复、部分 rank 继续等待旧 collective。

调试时可先打开：

```bash
TORCH_CPP_LOG_LEVEL=INFO \
TORCH_DISTRIBUTED_DEBUG=DETAIL \
NCCL_DEBUG=INFO \
torchrun --standalone --nproc-per-node=2 fsdp_minimal.py
```

`TORCH_DISTRIBUTED_DEBUG=DETAIL` 会增加开销，只应用于排错。若怀疑 collective 类型或消息大小不一致，可再设置 `NCCL_DEBUG_SUBSYS=COLL`。

### 18.3 Loss 与 DDP 对不上

按这个顺序核对：

1. 初始参数是否一致；
2. 各 rank 数据是否正确分片，且 global batch 相同；
3. loss 是 sum 还是 mean，梯度累积时有没有多除或少除；
4. 学习率是否因 global batch 改变而无意变化；
5. mixed precision 的 `param_dtype` 与 `reduce_dtype`；
6. 梯度裁剪算的是全局 norm 还是本地 shard norm；
7. tied weights、冻结参数、unused parameters 是否被框架正确处理；
8. optimizer 与 scheduler 的 step 顺序是否一致；
9. checkpoint 恢复是否遗漏 optimizer、scheduler、RNG 或 sampler 状态。

不要要求浮点结果逐 bit 相同。更有意义的是在固定数据与 seed 下比较前几步 loss、梯度范数、参数更新差异是否处在合理误差内，并验证长期收敛曲线。

### 18.4 显存省了，但训练反而很慢

可能是正常的容量换速度，也可能是：

- 模型太小，本来就更适合 DDP；
- root-only 导致大块阻塞 all-gather；
- wrap 太细，collective 启动延迟过多；
- 本地 batch 太小，没有足够计算掩盖通信；
- 跨节点带宽不足，纯 FSDP shard group 过大；
- activation checkpointing 重算成本太高；
- CPU offload 被 PCIe 或 CPU optimizer 限制；
- DataLoader 或 Python 本身是瓶颈；
- 日志中的频繁 `loss.item()`、同步计时把异步流水强制同步。

---

## 19. FSDP1 与 FSDP2 对照

很多 Hugging Face、旧版 PyTorch 教程和训练框架仍会出现下面的 FSDP1 代码：

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = FSDP(
    model,
    auto_wrap_policy=...,
    sharding_strategy=...,
    mixed_precision=...,
    use_orig_params=True,
)
```

看到它不用慌。核心原理仍然是参数 all-gather 和梯度 reduce-scatter，只是 API 与参数表示不同。

| FSDP1 | FSDP2 | 含义 |
|---|---|---|
| `FSDP(module)` wrapper | `fully_shard(module)` 原地组合 | 应用 FSDP |
| `auto_wrap_policy` | 用户按模块 bottom-up 调 `fully_shard` | 定义通信组边界 |
| `FlatParameter` | per-parameter `DTensor` | 分片参数表示 |
| `ShardingStrategy.FULL_SHARD` | `reshard_after_forward=True` | forward 后重新分片 |
| `ShardingStrategy.SHARD_GRAD_OP` | `reshard_after_forward=False` | 保留 full 参数到 backward |
| `HYBRID_SHARD` | 二维 `DeviceMesh` + reshard | HSDP |
| `MixedPrecision` | `MixedPrecisionPolicy` | 混合精度策略 |
| `no_sync()` | `set_requires_gradient_sync(False)` | 梯度累积时关闭同步 |
| `CPUOffload` | `CPUOffloadPolicy` | CPU 卸载 |
| `use_orig_params` | 始终保留原参数语义 | FSDP2 不再暴露 flat param 给用户 |
| FSDP state-dict 类型/context | DTensor + DCP state-dict API | checkpoint |

FSDP1 的 `NO_SHARD` 接近 DDP；`FULL_SHARD` 对应 ZeRO-3 类完整分片；`SHARD_GRAD_OP` 在计算间隙分片参数，但 forward 后暂时保留完整参数，以更多显存换少一次 all-gather。

阅读旧代码时重点找四件事，而不是背所有构造参数：

1. 哪些模块是 FSDP unit？
2. forward 后是否 reshard？
3. mixed precision 的参数、归约和 buffer dtype 是什么？
4. checkpoint 保存的是 full、sharded 还是 local state dict？

---

## 20. 常见误解

### 误解一：FSDP 是模型并行，所以每张 GPU 算不同层

不准确。FSDP 属于 sharded data parallel。各 rank 处理不同数据，并在当前 FSDP 单元计算前临时拿到完整参数。每张卡永久负责不同层更接近朴素模型切分或 Pipeline Parallel。

### 误解二：8 张 GPU 会把训练显存严格降到八分之一

不会。参数、梯度和优化器状态的稳定分片部分接近除以 8；激活、当前完整参数组、通信 buffer、算子 workspace 与 allocator overhead 不会一起严格除以 8。

### 误解三：套一个 `fully_shard(model)` 就完成了

语义上可能能跑，性能和峰值显存却常常很差。必须设计合理的子模块通信组，并 bottom-up 应用。

### 误解四：增加 GPU 就能解决任意大层

FSDP 计算当前单元时仍要临时还原它。单层本身放不下时需要把算子内部也切开的 TP，或重新设计模型/精度。

### 误解五：FSDP 会自动切数据

不会。仍需 `DistributedSampler`、正确的 iterable dataset sharding，或由训练框架显式处理。

### 误解六：只有 rank 0 保存 checkpoint 最安全

只有 rank 0 **写最终文件**可以，但构造 FSDP 完整 state dict 的 collective 必须由所有相关 rank 一起执行。训练恢复更推荐所有 rank 参与 DCP 分片保存。

### 误解七：显存下降说明 wrap 配置正确

还要看峰值发生位置、tokens/s、通信暴露比例和收敛是否对齐。一个配置可以省显存，却因为 root-only、小 collective 或跨节点 all-gather 而极慢。

### 误解八：FSDP 与 activation checkpointing 是二选一

它们主要解决不同部分的显存，经常组合使用：FSDP 分摊模型状态，activation checkpointing 减少激活。

---

## 21. 什么时候应该选 FSDP？

| 场景 | 首选思路 |
|---|---|
| 模型和目标 batch 能轻松放进单卡，只想提吞吐 | DDP，通常更简单且可能更快 |
| 模型状态放不进单卡，但最大单层可以放下 | FSDP2 |
| 参数能放下，但长序列激活 OOM | 先看 activation checkpointing、attention 实现、batch/sequence |
| 单个大层或算子都放不下 | TP，常与 FSDP 组合 |
| 跨节点纯 FSDP 通信太重，机内互联很快 | HSDP 或 TP + FSDP |
| GPU 显存仍不足，能接受明显吞吐损失 | CPU offload |
| 超大模型且需要深度切分 | 评估 PP + TP + FSDP 的多维并行 |

一个实用决策路径是：

```text
模型 + 训练状态是否能放进单卡？
├─ 能：优先 DDP
└─ 不能：最大单个层/算子能否在单卡计算？
   ├─ 能：FSDP2，按 block 分组
   │  └─ 激活仍 OOM：加 activation checkpointing / 优化 attention
   └─ 不能：引入 TP
      └─ 规模继续扩大：再组合 FSDP/HSDP 与 PP
```

---

## 22. 用四道题检查自己是否真的懂了

### 问题一

FSDP 稳定状态下每个 rank 只有参数分片，为什么还能运行普通 `Linear`？

答案：进入该 FSDP 单元前，hook 用 all-gather 临时还原完整参数；`Linear` 看到的仍是普通完整 tensor。

### 问题二

为什么 backward 后不做完整梯度 all-reduce？

答案：每个 rank 的 optimizer 只更新本地参数分片，只需要聚合后属于自己的梯度分片；reduce-scatter 正好完成“聚合 + 留一片”。

### 问题三

为什么 Transformer 通常按 block shard，而不是整个模型一个组？

答案：按 block 可把峰值限制在少数完整 block，并让下一 block 的 all-gather 与当前 block 计算重叠；整个模型一组会一次性 materialize 全部参数，并暴露一个大阻塞 collective。

### 问题四

为什么 FSDP 后仍可能因长序列 OOM？

答案：FSDP 主要分摊模型状态，不自动分片 forward 为 backward 保存的激活；序列长度、batch 和 attention 中间量仍可能主导显存。

如果这四题都能脱离原文讲清楚，FSDP 的主干心智模型已经建立起来了。

---

## 23. 参考资料

- [PyTorch：Getting Started with Fully Sharded Data Parallel (FSDP2)](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [PyTorch：`torch.distributed.fsdp.fully_shard` API](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)
- [PyTorch：Distributed Overview](https://docs.pytorch.org/tutorials/beginner/dist_overview.html)
- [PyTorch：Distributed communication 与调试](https://docs.pytorch.org/docs/stable/distributed.html)
- [PyTorch：Distributed Checkpoint API](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html)
- [PyTorch：Getting Started with Distributed Checkpoint](https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html)
- [PyTorch：Getting Started with DeviceMesh](https://docs.pytorch.org/tutorials/recipes/distributed_device_mesh.html)
- [PyTorch：`torchrun` 文档](https://docs.pytorch.org/docs/stable/elastic/run)
- [PyTorch：FSDP1 API（阅读旧代码用）](https://docs.pytorch.org/docs/stable/fsdp.html)
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- [PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel](https://arxiv.org/abs/2304.11277)

---

## 24. 一句话总结

> **FSDP 不是让每张 GPU 永远只计算模型的一小片，而是让每张 GPU 平时只保存模型状态的一小片：算当前 FSDP 单元前 all-gather 完整参数，按 reshard 策略释放临时完整参数，反向后 reduce-scatter 梯度，再由各 rank 只更新自己的参数与优化器状态分片。理解“分片稳定态、临时完整态、通信组边界”这三件事，就理解了 FSDP。**
