---
title: vLLM 的 Chunked Prefill：长输入怎样与解码请求一起调度
description: 从 prefill、decode 和 KV cache 讲起，解释 vLLM 怎样分块处理长输入、与解码请求组成混合 batch，以及延迟、吞吐和显存之间的取舍。
date: 2026-10-20
tags:
  - LLM
  - vLLM
  - Inference
  - Scheduling
  - KV-Cache
---

# vLLM 的 Chunked Prefill：长输入怎样与解码请求一起调度

假设一台推理服务器正在给几十个人回复消息，文字陆续出现在屏幕上。这时，又有人提交了一篇很长的文档，让模型帮忙总结。

模型需要先处理这篇文档，才能开始回答。如果服务器一次把整篇长输入都交给 GPU，正在输出的那些回复可能突然停顿；如果一直让正在生成的请求先跑，新来的长文档又可能迟迟轮不到。

Chunked prefill，也就是**分块预填充**，给调度器增加了一种选择：这轮只处理长输入的一部分，同时让其他请求继续生成；下一轮再接着处理剩余部分。

模型仍要处理完整输入，分块改变的是这段计算在各轮之间的安排。

> 本文依据 [vLLM v0.32.0](https://github.com/vllm-project/vllm/releases/tag/v0.32.0) 源码及 2026 年 10 月 20 日的 stable 文档，讨论采用因果注意力的普通 decoder-only 文本生成。多模态、混合架构和推测解码还需考虑其他约束。文中的调度与显存数字用于说明机制，没有进行 GPU 性能实测。

## 1. 从 token 到一次完整生成

### Token 是模型处理文本的单位

文本先经过 tokenizer，转换成模型接收的一串 token ID。一个 token 可能对应一个词、一个字、词的一部分，或某种特殊标记，所以“6000 token”不能直接当成“6000 个汉字”。

把输入写成：

$$
x_1,x_2,\ldots,x_P,
$$

这里 $P$ 是输入的 token 数。聊天服务还会把 system message、历史对话、角色标记等组织进模型输入，因此真正的 prompt 长度要按模型收到的 token 序列计算。

模型随后逐个生成回答中的 token：

$$
y_1,y_2,\ldots,y_G.
$$

自回归的意思是：生成 $y_2$ 时，要以输入和已经生成的 $y_1$ 为条件；生成 $y_3$ 时，还要加上 $y_2$。未来 token 尚未确定，通常不能把整个答案直接并行算出来。

### Prefill：先处理已经知道的输入

输入的所有 token 都已经给定。模型可以在因果注意力约束下，并行计算多个输入位置的表示，建立后续生成需要的状态。这段过程叫 prefill，中文常译作预填充。

处理完整个 prompt 后，模型用最后一个输入位置的输出预测第一个回答 token $y_1$，再通过采样或贪心选择得到 $y_1$。

因此，**首 token 通常来自最后一次 prefill 的结果**，无需再做一轮普通 decode。分块 prefill 要等到最后一块处理完，才会用 prompt 末尾位置的预测生成正式回答。

### Decode：沿着回答继续往后生成

得到 $y_1$ 以后，下一轮把它作为新的输入位置计算，预测 $y_2$；之后再用 $y_2$ 预测 $y_3$。

```text
prefill：处理 x1 ... xP  → 预测并选出 y1
 decode：处理 y1       → 预测并选出 y2
 decode：处理 y2       → 预测并选出 y3
 ...
```

这里说“一轮 decode 处理一个 token”，指普通自回归路径中**每个请求**新增一个计算位置。服务器可以同时处理几十个请求，让它们每个都前进一步；推测解码等机制则会改变这个简单计数。

Prefill 和 decode 是同一个模型在两种输入条件下的计算过程，可参照 [PagedAttention 论文第 2 节](https://arxiv.org/html/2309.06180v1#S2)。后文会介绍 vLLM V1 调度器怎样记录和推进这两类计算。

## 2. KV cache 保存了什么

Transformer 的注意力层会为 token 计算 Query、Key、Value，通常简写为 Q、K、V。当前 token 的 Q 与可见位置的 K 计算相关性，再用得到的权重对 V 求和，从上下文中提取信息。

对于一个普通因果注意力头，位置 $i$ 的计算可以写成：

$$
o_i=\sum_{j=1}^{i}\alpha_{ij}v_j,
\qquad
\alpha_{ij}=
\frac{\exp(q_i k_j^\top/\sqrt{d})}
{\sum_{m=1}^{i}\exp(q_i k_m^\top/\sqrt{d})}.
$$

$d$ 是这个注意力头的维度。求和只到 $i$，表示位置 $i$ 能看见自己和前面的位置，不能看见未来。

如果每生成一个新 token，都从头重算整段输入和已有输出，就会重复做大量工作。在因果注意力下，追加 token 不会改变前面位置能看见的内容，因此可以保留各层已经算好的 K/V，供后续位置继续使用。

这就是 KV cache。它保存的是**各层注意力所需的历史 K/V 张量**，不是文章摘要，也不是一份“模型已经理解了什么”的自然语言笔记。Q 通常只服务于当前位置的注意力计算，不像 K/V 那样作为历史供未来位置反复读取。

```text
已有上下文的 KV cache ─┐
                      ├─ 当前 token 的注意力计算 → 当前层输出
当前 token 的 Q/K/V ──┘
                └────── 新 K/V 追加到这一层的 cache
```

KV cache 也带来代价：在普通全注意力模型里，上下文越长，通常要保留的 K/V 越多；后来的 token 还要读取越来越长的历史。缓存省去了旧位置的重复计算，读取历史 K/V 的开销仍然存在。

## 3. 长 prompt 为什么会打断别人

先回到开头的服务场景。GPU 上同时有两类工作：

- 一批请求已经进入 decode，每个请求这轮只需处理一个新位置，希望尽快拿到下一个输出；
- 一个新请求需要 prefill 几千甚至几万 token，才能给用户第一个输出。

Prefill 一次有较多新 token，可把线性层等计算组织成较大的矩阵运算；decode 每个请求新增位置少，往往需要为较少的新计算读取模型权重和历史 KV。在常见负载下，前者更偏计算密集，后者更容易受显存带宽约束。实际瓶颈仍取决于模型、batch、上下文和硬件。

把请求按轮次动态加入或移出 batch，叫 continuous batching，或 iteration-level batching。它解决了“某个请求结束后，能不能马上补进新请求”的问题。如果新请求的整段 prefill 仍要一次算完，正在生成的请求就可能要等很久，才能等到下一轮。

即使把 decode token 和完整长 prefill 放进同一个 batch，decode 的结果也可能要等这一大轮计算结束才能交付。长 prefill 对输出间隔的影响，仍然存在。

Chunked prefill 把长输入拆到多轮处理，便于在各轮中安排 decode 请求，例如：

```text
不分块的一种安排：
[decode] [整个长 prompt 的 prefill................] [decode]

分块后的一种安排：
[decode + prefill 第 1 块]
[decode + prefill 第 2 块]
[decode + prefill 第 3 块]
```

每轮只加入一部分 prefill，decode 请求就有更多机会在轮次之间返回输出。图中方框的长度不代表耗时。

### 混合 batch 为什么还能提高效率

Prefill 和 decode 使用同一个模型。混合 batch 可以把它们这轮新增的位置合在一起，执行 QKV 投影、FFN 中的线性层等运算。更多 token 共用同一组模型权重，通常能提高矩阵运算效率，降低每个 token 分摊的权重读取开销。Attention 仍根据各请求自己的上下文长度和因果 mask 计算，请求之间不会互相读取上下文。

因此，混合 batch 的收益来自一起执行模型运算。分块则限制了一轮中加入多少 prefill，减少长输入对 decode 的拖延。混合 batch 比纯 decode batch 多做了工作，一轮的耗时仍可能增加；关键在于这段额外时间能处理多少输入，以及是否满足输出延迟要求。

[Sarathi 原论文第 4 节](https://arxiv.org/pdf/2308.16369)详细讨论了分块和混合 batch；后续的 [Sarathi-Serve 第 4 节](https://arxiv.org/html/2403.02310v1#S4)进一步研究了在线服务中的吞吐与延迟取舍。具体收益仍取决于模型、实现和负载。

## 4. vLLM 怎样决定这一轮算多少

`max_num_batched_tokens` 决定每轮可计算多少新 token。

**这份预算由一轮中的所有请求共享。** 每个请求能分到多少，要看其他请求已经用了多少预算。Attention 读取的历史 KV 已经计算过，不会再次计入新 token 数。

在普通文本生成、没有推测解码等额外机制的简化条件下，可以把约束写成：

$$
\sum_{r\in\text{本轮请求}}n_r\le B,
$$

其中 $B$ 是本轮预算，$n_r$ 是请求 $r$ 本轮新计算的 token 数。decode 请求常有 $n_r=1$；prefill 请求可能有几十、几百或几千。

例如，预算为 2048，已经给 32 个 decode 请求各安排 1 token 后，剩下的预算是：

$$
2048-32=2016.
$$

若一个长 prompt 还有 6000 token 没算完，这轮最多可处理其中的 2016 token，还要看 KV 空间、请求数等条件是否允许。因此，预算设为 2048，并不表示每个请求都按 2048 token 分块。

多个 prefill 也可能竞争同一预算。假如这轮 decode 花了 32，另一个 prefill 花了 500，那么剩给后续请求的只有 1516。预算可以有剩余，各请求分到的 token 数也不一定相等。

### “Decode 优先”要怎样理解

[vLLM 的优化文档](https://docs.vllm.ai/en/stable/configuration/optimization/#chunked-prefill)把这种策略概括为优先安排 decode，再用预算容纳 prefill。对“已有一批 decode，新来一个长 prompt”的场景，这个描述很直观。

在 v0.32.0 的 [`Scheduler.schedule()`](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/core/sched/scheduler.py#L559) 中，调度器先遍历 `running`，再处理 `waiting`。`running` 中也可能有尚未完成 prefill 的请求。调度器记录各请求已经算到哪里、还有多少 token 待计算，再依次分配预算。

所以，“decode 优先”不保证任何 decode 都先于任何未完成的 prefill。队列顺序、优先级、剩余预算、抢占和模型限制都会影响调度结果。分块让两类工作能够穿插执行，各请求具体获得多少计算机会，仍由调度器决定。

## 5. 一个例子：6000 token 分三轮处理

假设调度器每轮先服务 32 个 decode 请求，再把剩余预算分给长请求 L：

- 每轮 token 预算为 2048；
- 已有 32 个 decode 请求，它们在下面三轮都没有结束，每轮各新计算 1 token；
- 长请求 L 的 prompt 长度是 6000 token，在第一轮前到达；
- KV 空间充足，没有 prefix cache 命中、推测解码、抢占或额外分块上限，请求数和模型长度限制也不触发。

| 轮次 | Decode token | L 本轮 prefill token | L 剩余 prompt | L 能否输出首 token |
| ---- | -----------: | -------------------: | ------------: | ------------------ |
| 1    |           32 |                 2016 |          3984 | 还不能             |
| 2    |           32 |                 2016 |          1968 | 还不能             |
| 3    |           32 |                 1968 |             0 | 本轮完成后可以     |

第三轮，L 只剩 1968 个输入位置，所以这一轮共计算 $32+1968=2000$ 个 token。若没有其他请求，就留下 48 个 token 的未用预算。

L 的全部 prompt 已处理完，最后位置的结果可以用于采样首 token。若下一轮没有请求结束或新请求加入，那么 32 个老请求加上 L，共有 33 个普通 decode 请求，各新增一个计算位置；L 此时处理刚刚生成的 $y_1$，预测 $y_2$。

如果把这 6000 个 prefill token 一次做完，即使 32 个 decode 请求也放在同一轮，它们仍可能要等整个大 batch 完成才获得输出。分块后，它们在每个较小的混合轮次结束时都有机会返回结果；L 则需要跨三轮完成 prefill，多出一些调度开销。

**Token 预算并不等于时间预算。** 第二块虽然和第一块同为 2016 token，却需要访问更长的历史 KV；不同请求的上下文长度也不同。模型结构、attention kernel、GPU、并行通信与运行时开销，都会影响一轮到底用多久。

## 6. 后一块怎样使用前一块的上下文

把下面 8 个 token 分成三块：

```text
第 1 块：x1 x2 x3
第 2 块：x4 x5 x6
第 3 块：x7 x8
```

处理第一块时，每层生成位置 1–3 的 K/V，并保留到 cache。处理第二块时，模型为位置 4–6 生成新的 Q/K/V：新的 Q 可以读取第一块留下的 K/V，也可以读取当前块中因果 mask 允许的位置。

用 `✓` 表示当前 Q 可以关注对应的 K：

| 当前 Q | K1  | K2  | K3  | K4  | K5  | K6  | K7  | K8  |
| ------ | --- | --- | --- | --- | --- | --- | --- | --- |
| Q4     | ✓   | ✓   | ✓   | ✓   |     |     |     |     |
| Q5     | ✓   | ✓   | ✓   | ✓   | ✓   |     |     |     |
| Q6     | ✓   | ✓   | ✓   | ✓   | ✓   | ✓   |     |     |
| Q7     | ✓   | ✓   | ✓   | ✓   | ✓   | ✓   | ✓   |     |
| Q8     | ✓   | ✓   | ✓   | ✓   | ✓   | ✓   | ✓   | ✓   |

第二块的 `Q4` 不能看见 `K5`、`K6`，哪怕它们属于同一个计算块。到了第三块，`Q7` 则能看见前两块的全部历史，以及自己的 `K7`。

**块边界只决定这轮计算哪些新位置，每个位置能看见的上下文仍由因果 mask 决定。** 后一块能够读取前面各块的 K/V，attention 不局限于当前块内部。原始 [Sarathi 论文第 4.2 节与 Figure 6](https://arxiv.org/pdf/2308.16369)就用跨块 attention mask 解释了这种等价性；vLLM 的 [FlashAttention 后端](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/attention/backends/flash_attn.py)也将新 query、KV cache、序列长度和块映射等信息传给注意力实现。

### 每块都要经过整个模型

分块切分的是 token 序列。第一块仍要依次通过模型所有层，并在每一层保存对应的 K/V；第二块也通过所有层，在第 $\ell$ 层使用第 $\ell$ 层的历史缓存。

各块都经过完整模型，而 pipeline parallelism 是把模型的不同层放到不同设备上，两者切分的对象不同。

在模型、位置编码和注意力规则相同的条件下，先算前一块，再用它的 K/V 计算后一块，与一次处理完整序列在数学上等价。旧位置不依赖未来，所以追加新块时无需重算旧位置。

数学上等价，实际浮点结果却未必逐 bit 相同。Batch 形状、kernel 和归约顺序变化都可能产生数值差异，进而影响后续采样，因此两种安排生成的文本未必逐 token 一致。

如果模型采用 sliding-window attention，保留和读取的历史还受窗口大小限制。这是模型或缓存策略的规则，分块本身不会改变注意力窗口。

## 7. TTFT、ITL 和吞吐分别在衡量什么

用户看到的“快”，至少包含两种体验：多久开始回答，以及回答开始后是否流畅。

| 指标                      | 含义                                                  | 在这个例子里关心什么          |
| ------------------------- | ----------------------------------------------------- | ----------------------------- |
| TTFT，Time To First Token | 从请求发出到收到首个输出 token 的时间，需注明测量边界 | 长请求 L 多久开始回答         |
| ITL，Inter-Token Latency  | 相邻输出 token 之间的时间间隔                         | 老请求有没有突然停顿          |
| 端到端延迟                | 从发出请求到完成整个回答的时间                        | 一次任务多久结束              |
| 输出吞吐                  | 单位时间完成的输出 token 数                           | 服务器总体生成能力            |
| 请求吞吐                  | 单位时间完成的请求数                                  | 给定输入/输出分布下的处理能力 |

有些报告统计输入加输出的 total token/s，有些只统计 output token/s。比较吞吐前要先看清是否包含输入 token：长 prompt 占比越高，两种统计结果的差别可能越大。

对于客户端测得的 TTFT，可以做一个粗略拆分：

$$
T_{\mathrm{TTFT}}\approx
T_{\mathrm{transport}}+
T_{\mathrm{queue}}+
T_{\mathrm{prefill}}+
T_{\mathrm{delivery}}.
$$

四项依次对应传输与预处理、排队及各块间等待、prefill 执行和首 token 交付。分块既会改变 prefill 的执行时间，也会改变请求的等待时间。高负载下，吞吐提高还可能缩短队列，因此多分几轮并不一定让 TTFT 变差。

### 小预算通常能缩短输出间隔

在其他条件相近时，减小预算可以限制每轮的 prefill 工作量，让本轮 decode 更早完成，通常有助于减轻长输入到达时的输出停顿。

代价是长 prompt 需要更多轮，可能有更多调度和 kernel 启动开销；新 token 数太少时，大矩阵运算的效率也可能下降，旧 KV 还会被后续块反复读取。这会拖慢 prefill，影响 TTFT 或吞吐。

### 大预算通常能让长输入更快处理完

增大预算，能让长输入每轮处理更多 token，也更容易形成较大的 batch。但如果较大的 prefill 与 decode 混在一轮，老请求的输出间隔就可能变长。

预算和性能并非简单的线性关系。提高吞吐可能缩短排队，改善 TTFT；预算过大也可能增加工作区占用或引起抢占，反而让延迟和吞吐一起变差。官方[调优文档](https://docs.vllm.ai/en/stable/configuration/optimization/#performance-tuning-with-chunked-prefill)给出了这些方向性建议，具体数值仍应在目标负载下测量。

Sarathi-Serve 也沿着这一思路选择预算：先满足输出间隔的要求，再尽量提高吞吐。

### 平均 TPOT 可能掩盖偶发停顿

有些报告还会给 TPOT，通常按每个请求计算：

$$
\mathrm{TPOT}=\frac{T_{\mathrm{end}}-T_{\mathrm{first}}}{G-1},\qquad G>1.
$$

其中 $T_{\mathrm{end}}$ 和 $T_{\mathrm{first}}$ 分别是完成与收到首 token 的时间，$G$ 为输出 token 数。它是一个请求在首 token 之后的平均输出耗时。一次明显停顿可能被其余快速输出稀释，因此还需要看各次输出间隔的 ITL P95/P99。服务端的 token 时间与客户端流式接收时间也有区别：网络、缓冲或一次返回多个 token 都会影响客户端观测。比较报告前，要确认它们测量的是同一件事。

## 8. 分块能省多少显存

讨论分块的显存收益，需要区分当轮计算使用的激活、临时工作区，以及为后续计算保留的 KV cache。此外，模型权重、CUDA graph 和其他运行时缓冲区也会占用显存。

分块减少了每轮新计算的位置数，往往能降低相关激活或工作区的显存峰值。能省多少取决于 attention 后端和内存分配方式，不能简单把总显存除以块数。

而对于常规全注意力模型，**最终需要保留的 KV 容量不会因为分成四块就降到四分之一**。算完第四块后，前面各块的缓存通常仍需保留，供后续生成读取。

### 估算 KV cache 的大小

对常规 MHA/GQA，忽略分页开销、元数据、并行分片等因素，可以估算：

$$
M_{\mathrm{KV}}\approx
2\times L\times H_{\mathrm{KV}}\times D\times T\times s.
$$

其中：

- $2$ 表示 K 和 V 两份；
- $L$ 是模型层数；
- $H_{\mathrm{KV}}$ 是每层 KV head 数，GQA 中应使用 KV heads，而非 query heads；
- $D$ 是每个 head 的维度；
- $T$ 是当前缓存的 token 数；
- $s$ 是每个元素的字节数，例如 BF16 为 2。

假设一个模型有 32 层、8 个 KV heads、head dimension 为 128，KV 用 BF16，则每个 token 的 KV 约为：

$$
2\times32\times8\times128\times2
=131072\ \mathrm{bytes}=128\ \mathrm{KiB}.
$$

8192 个缓存 token 就约为 1 GiB。如果将输入均分为四块，每完成一块，请求使用的 KV cache 增加约 256 MiB，最终仍为 1 GiB；继续生成还会增加缓存。vLLM 可以预先分配一个大的 KV 缓存池，因此 `nvidia-smi` 显示的占用不一定随分块逐步增加。**缓存池分配了多少显存，与请求实际用了多少缓存块，是两回事。**

MLA、滑动窗口或混合模型的缓存结构不同，不能直接套用这个公式。PagedAttention 的内存管理讨论可参照[原论文第 3、4 节](https://arxiv.org/html/2309.06180v1#S3)。

### 有 token 预算，还要有 KV 空间

新请求能否开始计算，还取决于可用 KV 空间。v0.32.0 默认会检查完整输入所需的 KV 容量，即使第一块放得下，完整输入放不下时仍可能需要等待。这项检查由 `scheduler_reserve_full_isl` 控制，具体分配方式见附录 A。

所以遇到一个请求迟迟不开始，除了看 token 预算，还要看可用 KV 块、运行请求数和等待队列。运行中 KV 空间不足还可能触发抢占与重计算。如果总缓存需求已经超过容量，就需要减少并发或增加可用显存，仅把 chunk 调小未必能解决。

## 9. 几个经常被混在一起的机制

| 机制                | 主要改变什么                     | 与 chunked prefill 的关系                           |
| ------------------- | -------------------------------- | --------------------------------------------------- |
| Continuous batching | 在生成轮次之间动态接纳和移除请求 | 提供动态 batch；分块进一步缩小长 prefill 的调度粒度 |
| Chunked prefill     | 一次 prompt 的计算怎样分布到多轮 | 控制每轮新增 prefill 工作，与其他请求穿插           |
| Prefix caching      | 已算过的相同前缀能否直接复用     | 能跳过命中部分的 prefill，剩余部分仍可分块          |
| PagedAttention      | KV 怎样按块存放、寻址与共享      | 为缓存管理提供基础；内存页不等于一次调度的 chunk    |
| Prefill/decode 分离 | 两阶段由哪些实例或设备执行       | 把阶段放到不同资源池，通常需要传输 KV               |

### Prefix caching 复用的是相同前缀

假设两个请求都以前面同一份文档开头，第二个请求可以复用已缓存的相同 token 前缀，跳过这部分计算。若 6000 token 输入中有 4000 命中，就无需从零计算全部 6000；能复用多长的前缀，还取决于块边界、缓存是否仍然保留，以及实现规则。

Chunked prefill 无需命中其他请求的缓存。同一请求处理完第一块后，下一块继续读取自己的历史 K/V；prefix caching 则让不同请求能够复用同一个前缀。参见 [Automatic Prefix Caching 文档](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/)。

### PagedAttention 的 block 不是 prefill chunk

PagedAttention 将逻辑 KV 块映射到物理缓存块，改善动态分配和共享。prefill chunk 是请求在这一轮要计算的一段 token，一次 chunk 可以跨越多个 KV block；某些后端还要求 chunk 的边界对齐。两者分块的对象和目的不同。

### 分块与 prefill/decode 分离可以同时存在

分离部署把 prefill 和 decode 放到不同实例上，分别选择资源和并行配置，再通过连接器传输 KV。这样可以减少两阶段对同一设备的争用，但也增加了 KV 传输、请求路由，以及两端处理能力匹配的成本。Prefill 实例内部仍然可以分块处理输入。官方[分离式 prefill 文档](https://docs.vllm.ai/en/stable/features/disagg_prefill/)介绍了如何借此分别调节 TTFT 与 ITL；实际吞吐是否提高，还要看部署条件。

## 10. 怎样选择预算

选择预算前，先确定业务能接受多长的首 token 等待，以及多大的输出间隔，再在满足要求的配置中比较吞吐。对于长短输入混合的在线服务，尤其要观察长输入到来时，已有请求的输出是否突然停顿。

如果减小预算后 ITL 改善、长请求 TTFT 变差，就需要在两类请求的等待时间之间做取舍：已有请求更快拿到了下一个 token，长请求却更晚才能开始回答。若吞吐同时下降、队列持续增长，新请求的等待还会继续增加。若出现频繁抢占，则应先检查 KV 容量和并发数。

分块让一段长计算可以分多轮安排。预算决定每轮能做多少新工作，调度策略决定先服务哪些请求，KV 容量则决定能保留多少上下文。理解这三者，再去调整参数，才容易判断一次改动为什么有效。

实际调参时，可以查阅附录 A 的参数说明，并按附录 B 的流程比较不同预算。

## 附录 A：v0.32.0 的相关参数

V1 在模型和后端支持时默认启用 chunked prefill。默认 token 预算取决于硬件、离线 `LLM` 或 API 服务场景，以及其他配置，应以启动日志和最终生效的配置为准。默认值的选择见 [`arg_utils.py`](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/engine/arg_utils.py#L2787)。

| 参数                           | 作用与注意事项                                                      |
| ------------------------------ | ------------------------------------------------------------------- |
| `enable_chunked_prefill`       | 是否允许分块 prefill；V1 在支持时默认启用                           |
| `max_num_batched_tokens`       | 单轮共享的新 token 预算，不能用来表示 KV 总容量或单请求的固定块大小 |
| `max_num_seqs`                 | 单轮可处理的序列数上限，与 token 预算一起限制 batch 大小            |
| `max_model_len`                | 单请求输入加输出的长度上限，分块不能绕过这一限制                    |
| `gpu_memory_utilization`       | 引擎使用的 GPU 显存比例，会影响留给 KV 的空间                       |
| `long_prefill_token_threshold` | 限制较长待计算段在一轮中处理的 token 数；默认 0 表示不额外设限      |
| `scheduler_reserve_full_isl`   | 接纳新请求时检查完整输入的 KV 需求；默认 `True`                     |

### 完整输入的 KV 容量检查

默认启用 `scheduler_reserve_full_isl` 时，接纳新请求前会检查完整输入所需的 KV 空间。通过检查后，仍按本次计算需要分配缓存块。这项检查不意味着一次算完整个输入，也不保证未来所有输出的 KV 都能放下。Prefix cache 命中等情况也会影响所需块数。相关实现见 [`KVCacheManager.allocate_slots()`](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/core/kv_cache_manager.py#L478)。

### 每轮 prefill 上限的例外

`long_prefill_token_threshold` 默认不额外限制每轮的 prefill token 数，请求仍可按总预算分块。当 `running`、`waiting` 和 `kv_holding_waiting` 合计仅有一个请求时，不应用这一上限。这里统计的是队列中的请求数，包含暂时无法执行的等待请求。开启 `long_prefill_token_threshold_adaptive` 后，实际使用的上限还可能变化。

参数定义见 [`SchedulerConfig`](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/config/scheduler.py)，请求计数见 [`get_request_counts()`](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/core/sched/scheduler.py#L2410-L2412)。旧版本中的 `max_num_partial_prefills`、`max_long_partial_prefills` 等参数，需要结合对应版本理解。

V1 还为推测解码等场景提供了区分调度 token 数与输入处理容量的配置。开启这类功能后，每个请求一轮所需的 token 数会变化，前面的“一请求一 decode token”示例不再适用。

## 附录 B：比较不同预算的实验方法

实验可以围绕一个具体问题展开：**在业务的长短输入混合流量下，哪个预算既能维持输出流畅，又能让新请求及时开始？**

以下命令和数值用于搭建实验，未在本文环境中执行。模型路径、上下文长度和显存配置需要按实际环境调整，预算也需通过测量选择。

### 启动服务

```bash
vllm serve /path/to/model \
  --enable-chunked-prefill \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 64 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.9 \
  --no-enable-prefix-caching
```

这里关闭 prefix caching，是为了单独观察分块和预算的影响。生产环境是否开启，应按业务负载测试。普通自回归生成还要求 token 预算不小于 `max_num_seqs`；如果关闭 chunking 做对照，需确认模型支持，并满足 `max_num_batched_tokens >= max_model_len` 的配置校验。

### 固定其他条件，比较不同的 token 预算

固定模型及 revision、tokenizer、精度、KV dtype、GPU 型号和数量、并行方式、vLLM 版本、`max_model_len`、`max_num_seqs`、采样配置与输出长度。保持 prefix caching 的命中条件，以及推测解码、LoRA、多模态等开关一致。

可以先比较 1024、2048、4096、8192 这些预算。每次修改后记录最终生效的配置，完成预热，再重复测量，避免将编译、CUDA graph 捕获或首次加载计入结果。

分别准备短输入、长输入和长短混合负载，例如以 256 与 8192 token 输入、256 token 输出为起点，再换成真实业务分布。逐步提高请求到达率，并在不同并发上限下测量。请求到达率控制每秒送来多少请求，并发上限限制同时进行的请求数，两者会形成不同的排队情况。

### 观察长请求到来时的输出停顿

先让一批输出较长的请求进入 decode，再加入一个或一批长 prompt。记录各请求的发送时间、首 token 时间、各次输出时间和结束时间，并标记长输入到达的时刻。

比较不同预算下，已有请求的 ITL 是否在这一时刻出现尖峰，长请求等待首 token 花了多久，以及系统最终完成了多少请求。流式输出片段未必与 token 一一对应，因此还要记录实际 token 数，区分客户端接收间隔与模型生成间隔。

### 用官方 benchmark 记录基准结果

在已启动的本地服务上，可以用以下命令收集一组长输入负载的结果，再补充前面的混合流量实验：

```bash
vllm bench serve \
  --backend vllm \
  --model /path/to/model \
  --dataset-name random \
  --random-input-len 8192 \
  --random-output-len 256 \
  --num-prompts 300 \
  --num-warmups 10 \
  --request-rate 2 \
  --max-concurrency 32 \
  --ignore-eos \
  --percentile-metrics ttft,tpot,itl \
  --metric-percentiles 50,95,99 \
  --save-result \
  --save-detailed
```

`--ignore-eos` 用于尽量固定输出长度，测量后仍应核对实际输入和输出长度，并补一组真实业务请求。参数含义见 [`vllm bench serve` 官方文档](https://docs.vllm.ai/en/stable/cli/bench/serve/)。

客户端并发限制可能让请求在发送前排队。v0.32.0 的 benchmark 将这部分时间单独记录为 `client_queue_time`，不计入 TTFT。它的 ITL 来自客户端收到的流式输出片段间隔，分位数由成功请求的全部间隔汇总计算，输出较长的请求会贡献更多样本。具体实现见[请求发送](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/benchmarks/serve.py#L905)、[间隔记录](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/benchmarks/lib/endpoint_request_func.py#L209)和[指标汇总](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/benchmarks/serve.py#L564)。

每组结果应记录预算、到达率和并发上限，并按短请求、长请求分别统计：

- 完成请求数、错误率；
- TTFT 和 ITL 的 P50、P95、P99；
- 输出 token/s；
- 等待请求数、KV 使用量和抢占次数。

同时查看[服务端指标](https://docs.vllm.ai/en/stable/usage/metrics/)，判断时间主要花在排队、prefill 还是 decode 上。检查失败和未完成请求，避免它们被排除后让平均延迟显得更低。P99 需要足够多的样本，几十个请求不足以判断结果是否稳定。

## 参考资料与源码

- [vLLM v0.32.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.32.0)：本文固定的实现版本。
- [Optimization and Tuning](https://docs.vllm.ai/en/stable/configuration/optimization/)：Chunked Prefill、性能取舍和抢占说明；stable 会随发布更新。
- [`Scheduler.schedule()`，v0.32.0](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/core/sched/scheduler.py#L559)：请求进度、running/waiting 遍历与预算扣减。
- [`SchedulerConfig`，v0.32.0](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/config/scheduler.py)：token 预算、每轮 prefill 上限、接纳检查和配置校验。
- [`arg_utils.py`，v0.32.0](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/engine/arg_utils.py)：默认配置如何根据场景与硬件选择。
- [`KVCacheManager`，v0.32.0](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/core/kv_cache_manager.py)：缓存块分配与完整输入容量检查。
- [SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills](https://arxiv.org/abs/2308.16369)：第 4 节解释分块、跨块 mask 与混合 batch。
- [Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve](https://arxiv.org/html/2403.02310v1)：在线调度、token budget 与延迟约束。
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/html/2309.06180v1)：KV 内存管理、分页与 vLLM 的基础设计。
