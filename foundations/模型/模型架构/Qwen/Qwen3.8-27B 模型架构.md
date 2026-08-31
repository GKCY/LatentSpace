# Qwen3.8-27B 架构拆解：48 层线性注意力 + 16 层全局注意力

> 本文资料核对日期为 2026 年 8 月 22 日，研究对象是 2026 年 8 月 14 日发布的官方后训练权重 `Qwen/Qwen3.8-27B`。截至本文写作时，Qwen 尚未发布一份专属于 3.8-27B 的技术报告，因此架构数字以官方 Model Card、`config.json`、权重索引和 Hugging Face Transformers 实现为准。涉及参数量与显存的数字会明确标注为“推算”。

Qwen3.8-27B 最容易被误解成一台普通的“27B Transformer”。它其实同时集成了四类设计：27B 级稠密语言模型、Gated DeltaNet 与全局注意力混合的 Decoder、原生图文视频输入，以及为 speculative decoding 准备的 MTP 辅助头。

它最值得关注的地方也不是某个孤立的新算子，而是一组相互制衡的取舍：让 48 层线性注意力承担低成本的流式记忆，再让 16 层全局注意力周期性地恢复精确检索；视觉 token 与文本 token 共用同一个 Decoder；原生上下文做到 262,144 token，再通过 YaRN 外推到 100 万。

先给结论：**Qwen3.8-27B 不是纯线性模型，也不是 MoE，更不是一个把视觉编码器临时接到文本模型上的“看图插件”。它是一台稠密、原生多模态、混合注意力的自回归模型。**

---

## 1. 一张表看懂 Qwen3.8-27B

| 项目             | 公开配置                                                                         |
| -------------- | ---------------------------------------------------------------------------- |
| 模型类型           | 带视觉编码器的 Causal Language Model                                                |
| 参数形态           | Dense，不是 MoE                                                                 |
| 语言模型宽度         | 5,120                                                                        |
| Decoder 层数     | 64                                                                           |
| 层布局            | `16 × [3 × Gated DeltaNet + 1 × Gated Attention]`                            |
| Gated DeltaNet | 48 层；16 个 Q/K head、48 个 V head；head dim 128                                  |
| 全局注意力          | 16 层；24 个 Q head、4 个 KV head；head dim 256                                    |
| FFN            | 每层都有一个稠密 SwiGLU，intermediate size 17,408                                     |
| 词表/输出行数        | 248,320，已 padding；输入与输出权重不共享                                                 |
| 位置编码           | partial multimodal RoPE；仅旋转全局注意力每个 256 维 Q/K head 的前 64 维；DeltaNet 层不使用 RoPE |
| 原生上下文          | 262,144 token                                                                |
| 扩展上下文          | 通过 YaRN 可到 1,000,000 token                                                   |
| 视觉塔            | 27 层 ViT；hidden 1,152；16 heads；FFN 4,304                                     |
| 视觉 patch       | 时间维 2 帧，空间维 `16 × 16`；随后做 `2 × 2` spatial merge                              |
| MTP            | 1 层辅助预测模块，经过多步 MTP 训练                                                        |
| 权重格式           | BF16，另有独立的官方 FP8 版本                                                          |
| License        | Apache-2.0                                                                   |

这里的“语言模型宽度 5,120”就是配置中的 `hidden_size=5120`。更准确地说，它是每个 token 进入模型后对应的 hidden state 维度：如果 batch size 是 $B$、序列长度是 $N$，那么 embedding 输出以及每个 Decoder block 的主残差流形状都是 `[B, N, 5120]`。这不是上下文长度，也不是词表大小；token 本身在进入 embedding 前只是一个整数 ID。

模型内部的注意力宽度也不必等于 hidden size。例如全局注意力的 Query 内部宽度是 $24 \times 256=6,144$，随后再由输出投影映射回 5,120 维，才能继续进入下一段残差连接。

一个值得先记住的细节是：配置中的 `model_type` 仍然叫 `qwen3_5`，加载类仍是 `Qwen3_5ForConditionalGeneration`。这不是模型传错了。复用实现类与 Qwen 官方所述“3.8 建立在 Qwen3.5 的架构基础上”相一致：至少从公开网络定义看，3.8 是同一骨架上的权重与训练迭代，而不是另起炉灶。

## 2. 从输入到输出：整条数据路径

把模型压缩成一张结构图，大致是这样：

```text
图像 / 视频
    │
    ├─ Conv3D Patchify: 2 × 16 × 16
    ├─ 27-layer Vision Transformer, hidden=1152
    └─ 2 × 2 Spatial Merge + Project to 5120 ─────────────┐
                                                          │
文本 ─ Byte-level BPE ─ Token Embedding, hidden=5120 ────┤
                                                          ▼
                                  交错的文本 / 图像 / 视频 embedding
                                                          │
                         ┌────────────────────────────────┴───────────┐
                         │ 重复 16 次                                  │
                         │                                            │
                         │  Gated DeltaNet → Dense SwiGLU FFN   × 3   │
                         │  Gated Global Attention → Dense FFN  × 1   │
                         │  └─ Q/K: partial T/H/W M-RoPE              │
                         └────────────────────────────────────────────┘
                                                          │
                                               Final RMSNorm
                                                          │
                                      Untied LM Head: 248,320 rows
                                                          │
                                                   next token

训练 / 特定推理后端可额外使用：1-layer MTP draft head
```

视觉编码器不会把一条独立的 cross-attention 支路挂在每个语言层上。它先把图像和视频变成 5,120 维视觉 embedding，再替换输入序列中的视觉占位 token；此后视觉与文本进入同一套 64 层自回归 Decoder。配置还将 `deepstack_visual_indexes` 设为空，因此这里没有 Qwen3-VL 式的多层 DeepStack 视觉注入。

这正是“原生多模态”更准确的含义：**仍然有专门的视觉前端，但视觉表示很早就进入统一 token 流，并与文本一起接受后续语言建模。**

## 3. 核心骨架：为什么是“三层线性 + 一层全局”

64 个 Decoder block 不是同一种模块重复 64 次，而是下面这个四层周期重复 16 次：

```text
Layer  0: Gated DeltaNet  + FFN
Layer  1: Gated DeltaNet  + FFN
Layer  2: Gated DeltaNet  + FFN
Layer  3: Global Attention + FFN
                ...
Layer 60: Gated DeltaNet  + FFN
Layer 61: Gated DeltaNet  + FFN
Layer 62: Gated DeltaNet  + FFN
Layer 63: Global Attention + FFN
```

每个 block 都采用 pre-norm 和两段残差，抽象成公式就是：

$$
h' = h + \operatorname{Mixer}(\operatorname{RMSNorm}(h))
$$

$$
h_{out} = h' + \operatorname{SwiGLU}(\operatorname{RMSNorm}(h'))
$$

其中 `Mixer` 根据层号选择 Gated DeltaNet 或全局 Gated Attention。RMSNorm 的 `eps` 为 $10^{-6}$；语言主干使用 zero-centered RMSNorm，即参数在零点初始化、实际缩放围绕 1 展开。语言主干的 Attention 与 FFN 线性投影均不使用 bias，attention dropout 也为 0。

### 3.1 Gated DeltaNet：把历史压进固定大小的状态

标准 attention 要计算当前 query 与历史 key 的两两关系。序列长度是 $N$ 时，全局注意力的 prefill 计算量随 $N^2$ 增长；FlashAttention 等实现可以避免显式物化完整的 $N \times N$ 矩阵，却不能消除二次计算。decode 时还要持续保存历史 K/V。

Gated DeltaNet 换了一种思路：为每个 head 维护一个相对于上下文长度固定大小的 recurrent state，把过去的 key-value 关系压缩进状态矩阵。下面采用 $S_t \in \mathbb{R}^{d_v \times d_k}$、$k_t,q_t \in \mathbb{R}^{d_k}$、$v_t,o_t \in \mathbb{R}^{d_v}$ 的转置约定，用省略实现细节的示意写法表示：

$$
\widetilde S_t = \alpha_t S_{t-1}
$$

$$
\widehat v_t = \widetilde S_t k_t
$$

$$
S_t = \widetilde S_t + \beta_t (v_t - \widehat v_t) k_t^\top,
\qquad o_t = S_t q_t
$$

Transformers 实现内部保存的是等价的 $K \times V$ 方向状态；转置和乘法方向不同，但下面的直觉不变：

- $\alpha_t$ 是遗忘门，决定旧记忆保留多少；
- 状态先根据 $k_t$ 读出自己“以为”对应的值 $\widehat v_t$；
- Delta Rule 不盲目追加 $v_t$，而只写入真实值与预测值之间的误差；
- $\beta_t$ 控制这次修正的强度，最后再由 $q_t$ 读取状态。

Qwen3.8-27B 的每个 Gated DeltaNet 层有 16 个 Q/K head 和 48 个 V head，head dimension 都是 128；每个 Q/K head 与 3 个 V head 配对，形成 1:3 的映射。在进入 delta rule 之前，Q/K/V 还会通过 kernel size 为 4 的 depthwise causal convolution，补一段很短的局部时序归纳偏置；Q/K 做 L2 normalization，`a` 与 `b` 投影分别生成 decay 和 update gate，额外的 `z` 投影控制输出。

这 48 层的优势是：单 token recurrent decode 不必让状态随上下文长度增长，prefill 也可以使用 chunkwise 算法并行计算。代价则是，固定状态毕竟是对历史的压缩，不如完整 attention 擅长逐 token 的精确回看。

关于线性注意力、Delta Rule、遗忘门与 chunkwise parallel 的完整推导，可继续阅读 [[Gated DeltaNet]]。

### 3.2 每四层一次全局注意力：给压缩记忆加“校准点”

Qwen 没有把整个模型都换成线性注意力，而是在每个四层周期的第 4 层保留一次真正的 causal global attention。这里不是 sliding window，也不是 local attention：16 层都可以看见此前完整上下文。

全局层使用 GQA：

- 24 个 Query heads；
- 4 个 Key/Value heads；
- 每个 head 256 维；
- 每个 KV head 被 6 个 Query heads 共享。

注意 $5,120 / 24$ 并不等于 256。这里 Q 投影后的注意力内部宽度是 $24 \times 256 = 6,144$，可以大于 residual stream 的 5,120 维；K、V 投影宽度则各为 $4 \times 256 = 1,024$。

它也不是最朴素的 GQA。Q 投影一次产生 query 与同宽的 output gate，attention 结果先逐元素乘上 `sigmoid(gate)`，再通过输出投影回到 5,120 维。Q 和 K 还各自经过 head-wise RMSNorm。Qwen 在 Qwen3-Next 的公开实验中将 output gating 解释为缓解 attention 低秩、Attention Sink 与 massive activation 问题的稳定性设计。

于是 3:1 的混合结构形成了一个很务实的分工：

- 75% 的层用固定状态做低成本记忆更新；
- 25% 的层付出全局注意力成本，保留精确检索能力；
- 模型没有消灭二次复杂度，只是把昂贵的全局注意力层数降到了四分之一。

因此，把 Qwen3.8-27B 简写成“严格 $O(N)$ 的线性模型”是不准确的。在固定模型宽度和层数时，token mixing 部分随序列长度变化的 prefill 主项可以写成 $\Theta(48N + 16N^2)=\Theta(N^2)$；混合结构降低了二次计算的常数和 KV cache，却没有改变整网的渐进复杂度。

## 4. 全局注意力的位置编码：256 维 head 为什么只旋转 64 维

全局注意力的 head dimension 是 256，但 RoPE 只作用在前 25%，即 64 维。剩余 192 维不旋转。Qwen3-Next 的公开结论是，partial RoPE 比全维旋转更利于长序列外推。

这 64 个旋转维又组成 32 个正余弦频率对，并按 `[11, 11, 10]` 切给三个多模态坐标轴：

| 分区 | 表达的坐标 |
| --- | --- |
| 11 个频率对 | temporal，时间 |
| 11 个频率对 | height，高度 |
| 10 个频率对 | width，宽度 |

三组频率交错排列，`rope_theta` 为 10,000,000。对纯文本 token，位置仍然退化成普通的一维顺序；对图像和视频 token，同一套 attention 则能感知时间、高度和宽度。这也是视觉 token 可以与文本 token 共用语言 Decoder 的关键接口之一。

## 5. FFN 才是参数大户，线性注意力并不等于“小参数量”

每一个 Decoder block 后面都有稠密 SwiGLU：

$$
\operatorname{FFN}(x)
= W_{down}\left(\operatorname{SiLU}(W_{gate}x) \odot W_{up}x\right)
$$

输入宽度 5,120，中间宽度 17,408，三个矩阵都不带 bias。单层 FFN 参数量为：

$$
3 \times 5,120 \times 17,408 = 267,386,880
$$

64 层加起来约 17.11B，已经占整个公开 checkpoint 的六成以上。

根据官方配置、实现和 `model.safetensors.index.json` 可以把参数账本进一步拆开。下表是按张量形状推算的结果，不是 Qwen 官方给出的分项统计：

| 组件 | 约参数量 | 观察 |
| --- | ---: | --- |
| 64 个 Dense SwiGLU FFN | 17.113B | 占比最大，约占 checkpoint 的 61.6% |
| 48 个 Gated DeltaNet mixer | 5.562B | 单层约 115.9M，并不比全局层更省参数 |
| Input embedding + LM head | 2.543B | 两个 `248,320 × 5,120` 矩阵，且不共享 |
| 16 个 Gated Global Attention mixer | 1.678B | 单层约 104.9M |
| Decoder block 与 final RMSNorm | 0.00066B | `(2 × 64 + 1) × 5,120 = 660,480` |
| 视觉塔与 merger | 0.461B | 27 层 ViT 加 5,120 维投影 |
| MTP 辅助模块 | 0.425B | 一层 draft head |
| **合计** | **约 27.781B** | 与权重索引中 55,562,855,904 字节的 BF16 权重对齐 |

官方 Model Card 只在 `Language Model` 项下标注 27B，并没有进一步定义模型名是否计入 MTP 与视觉塔。按张量拆分推测，语言模型主路径（不含视觉塔与 MTP）约 26.896B，可取整称为 27B；加上 MTP 后约 27.321B，再加视觉塔后，整个官方 BF16 权重包约 27.781B。Hugging Face 页面将它显示成约 28B 也不矛盾。

这个账本还揭示了两个容易忽略的事实：

第一，线性注意力主要节省的是长序列 token mixing 与缓存成本，并不是为了减少 mixer 参数。第二，24.8 万行的大词表加上 untied LM head 本身就占约 2.54B 参数，已经接近一台小语言模型。

## 6. 视觉塔：图像与视频怎样变成语言 token

视觉前端沿用了 Qwen3-VL 的 ViT 设计，但关闭了 DeepStack。公开配置是：

- 27 层 Vision Transformer；
- hidden size 1,152；
- 16 个 attention heads，因此视觉 head dimension 为 72；
- FFN intermediate size 4,304，激活采用 tanh 近似形式的 GELU；
- 2,304 个 learned position embeddings，可看作一张 `48 × 48` 的基础位置网格，并在运行时插值到实际分辨率。

输入首先通过 kernel 和 stride 都为 `[2, 16, 16]` 的 Conv3D。也就是说，原始 patch 在时间轴覆盖 2 帧、空间上覆盖 `16 × 16` 像素。ViT 编码后，merger 再把相邻 `2 × 2` 的空间 patch 拼成 4,608 维，经过两层投影映射到语言模型的 5,120 维，同时把视觉 token 数压缩为原来的四分之一。

之后，视觉 embedding 直接替换图像或视频占位 token 所在位置，模型为整条序列计算三维位置 ID，再交给同一套混合 Decoder；这些位置 ID 会在全局注意力层内用于旋转 Q/K。这里没有每层 cross-attention，也没有单独的视觉回答头；最终仍由统一 LM head 逐 token 生成文本。

## 7. 262K 是原生长度，1M 是 YaRN 外推

公开配置中的 `max_position_embeddings` 是 262,144。官方给出的 100 万 token 方案，是在部署时把 RoPE 改成静态 YaRN：

```json
{
  "rope_type": "yarn",
  "factor": 4.0,
  "original_max_position_embeddings": 262144,
  "partial_rotary_factor": 0.25,
  "mrope_interleaved": true,
  "mrope_section": [11, 11, 10],
  "rope_theta": 10000000
}
```

这两个说法不能混在一起：**262K 是原生上下文，1M 是位置编码缩放后的可扩展长度。** 官方还特别提醒，主流推理框架目前使用 static YaRN，固定缩放因子可能伤害短上下文效果；如果目标上下文约为 524K，更合适的是 `factor=2`，而不是一直开着 `factor=4`。

支持长上下文并不意味着长上下文推理显存低。先看未启用 MTP 时的主生成路径：只计算 16 个全局层的原始 BF16 KV cache，在 batch size 为 1 时，每个 token 需要：

$$
16\ \text{layers}
\times 2\ (K,V)
\times 4\ \text{KV heads}
\times 256\ \text{dims}
\times 2\ \text{bytes}
= 65,536\ \text{bytes}
$$

也就是每 token 64 KiB。于是：

- 262,144 token 的全局层 KV cache 理论值约为 16 GiB；
- 1,000,000 token 的理论值约为 61 GiB。

`factor=4` 的数学跨度其实是 $4 \times 262,144=1,048,576$；如果把它用满，16 层 KV 恰为 64 GiB，而官方给出的 1,000,000 token 上限对应约 61.04 GiB。

这里尚未包含权重、激活、分页与 allocator 开销，也没有乘 batch size；视觉 token 与文本 token 共用窗口，上述数值按两者合计的总 token 数计算。Gated DeltaNet 的矩阵状态在 batch size 为 1 时共有 $48 \times 48 \times 128 \times 128=37,748,736$ 个标量；若按配置中的 FP32 口径估算，约为 144 MiB，且不随序列长度增长，此外还有较小的卷积状态。

如果推理后端启用 MTP draft head，它的一层全局注意力还可能额外需要 4 KiB/token 的 cache，即 262K 时约 1 GiB、100 万时约 3.81 GiB；能否复用缓存取决于后端。作为对照，如果主模型的 64 层全都采用同规格全局 GQA，262K 时仅 KV 就会达到约 64 GiB；当前混合架构把主路径中随长度增长的 KV 部分削掉了 75%。不过，100 万 token 仍然不是一张消费卡上“改个配置就免费得到”的能力。

## 8. MTP 与 thinking：哪些是网络结构，哪些是后训练行为

权重包中还有一个单层 MTP（Multi-Token Prediction）模块，包含融合层、一层 attention、一层 FFN 和 norm；配置为 `mtp_num_hidden_layers=1`，并与主模型共享 embedding 体系。Qwen3.8-27B Model Card 只明确说它经过了多步 MTP 训练；在祖先架构 Qwen3-Next 的说明中，Qwen 将这套机制用于生成高接受率的 speculative decoding draft token。

但 checkpoint 带有 MTP 权重，不等于普通 `generate()` 会自动一次吐出多个 token。标准自回归路径仍然逐 token 解码，只有显式支持并启用该 draft head 的推理后端，才可能把它转化成加速；加速比还取决于接受率、batch、上下文和 kernel，不能从“有 MTP”直接推出一个固定倍数。

关于多预测距离的目标函数、并行 head 与顺序 module 的区别，以及 MTP 如何通过 speculative decoding 转化为推理加速，可继续阅读 [[MTP]]。

同样，默认 `thinking mode`、`reasoning_effort=xhigh/medium/low` 和 `preserve_thinking` 主要是后训练与 chat template/API 层面的能力，不是 Decoder 内多出三档“推理层”。`preserve_thinking` 会把历史 `reasoning block` 保留在多轮上下文中，有助于任务连续性和 prefix/KV 复用，但也会真实占用上下文窗口。

## 9. Qwen3.8 相比 Qwen3.5 / Qwen3.6，架构到底改了什么

把 Qwen3.5-27B、Qwen3.6-27B 与 Qwen3.8-27B 的官方配置并排比较，会得到一个可能有点反直觉的结论：

- 都是 64 层、hidden size 5,120、FFN 17,408；
- 都是 48 层 Gated DeltaNet + 16 层全局 Gated Attention；
- 三者的全局注意力都是 24 个 Q head、4 个 KV head，head dim 为 256；
- 视觉塔也都是 27 层、hidden size 1,152；
- 原生上下文同为 262,144，MTP 同为一层。

换句话说，从公开结构超参数看，**Qwen3.8-27B 并不是靠换骨架升级的。** 官方 Model Card 在其评测设定下给出的结果——例如 SWE-bench Pro 从 Qwen3.6-27B 的 53.5 提升到 61.7，OSWorld-Verified 从 63.9 提升到 84.3——说明能力增益应主要来自训练数据、训练与后训练配方，而非公开结构超参数的变化。由于官方尚未披露 Qwen3.8-27B 的完整训练配方，这只是根据相同配置做出的推断，具体归因仍无法确定。

这也解释了为什么 `config.json` 和 Transformers 实现继续使用 `qwen3_5` 架构名：对部署框架而言，它需要支持的是同一套算子组合；对用户而言，3.8 的差异主要体现在模型学到了什么、如何思考和如何执行任务，而不在 Python 类名。

## 10. 部署时真正要关心的四件事

### 10.1 不要只看“27B”估显存

官方 BF16 权重文件合计约 55.56 GB（十进制，约 51.75 GiB），此外还有 KV cache、Gated DeltaNet 状态和运行时 workspace。量化能显著降低权重占用，却不会自动消除长上下文 cache。

### 10.2 推理框架必须真正支持 Gated DeltaNet kernel

Transformers 文档明确提示：缺少 `causal_conv1d` 和 FLA 等快速 kernel 时，会回退到更慢、更耗内存的 PyTorch 实现。能“加载成功”与能“高效运行”是两回事，尤其在长 prompt 的 prefill 阶段。

### 10.3 1M 模式应按请求隔离

static YaRN 可能影响短文本，且 1M cache 成本巨大。更合理的服务设计是把常规 262K 与超长 YaRN 实例分开，或者按典型长度选择 factor，而不是把所有请求都固定跑在 factor 4。

### 10.4 Dense 更简单，但不等于更小

27B dense 没有 expert routing、负载均衡和跨设备 all-to-all 的复杂性，部署行为通常比同总参数 MoE 更直接；代价是每个 token 都要经过全部 64 个稠密 FFN。它适合的是“可预测的单模型部署”，不是以极低 active parameters 取胜的吞吐路线。

## 11. 五个常见误读

1. **“Qwen3.8-27B 是 MoE。”** 错。27B 是 dense；同代 2.4T-A95B 才是 MoE。
2. **“它是纯文本模型，视觉只是 API 功能。”** 错。公开权重包含约 0.46B 的视觉塔与 merger，支持图像和视频。
3. **“全模型都是线性注意力，所以严格 $O(N)$。”** 错。48 层是 Gated DeltaNet，16 层仍是全局 attention。
4. **“原生支持 1M。”** 不准确。原生是 262K，1M 需要 YaRN 外推。
5. **“MTP 权重存在，所以任何框架都能自动加速。”** 错。是否使用 MTP draft head 取决于推理后端。

## 12. 总结

Qwen3.8-27B 的架构可以浓缩成一句话：

> 它用 48 层 Gated DeltaNet 压低长序列的计算和缓存成本，用 16 层周期性全局注意力守住精确回看能力，再把图像、视频和文本统一送入一个 5,120 维、64 层的稠密 Decoder。

从架构研究的角度，它最有意思的地方不是“3.8 发明了什么全新 block”，而是这套从 Qwen3-Next、Qwen3.5 延续下来的混合骨架经受住了多轮模型迭代：模型能力可以大幅变化，核心结构却几乎不动。这提醒我们，今天大模型版本之间的进步，越来越不能只靠 `config.json` 解释；架构决定能力与效率的边界，数据和后训练则决定模型最终能在该边界内达到怎样的水平。

## 参考资料

1. [Qwen3.8-27B 官方 Model Card](https://huggingface.co/Qwen/Qwen3.8-27B)
2. [Qwen3.8-27B `config.json`](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json)
3. [Qwen3.8-27B 权重索引](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/model.safetensors.index.json)
4. [Qwen3.8 官方 GitHub 仓库](https://github.com/QwenLM/Qwen3.8)
5. [Hugging Face Transformers：Qwen3.5 文档](https://huggingface.co/docs/transformers/model_doc/qwen3_5)
6. [Transformers：Qwen3.5 模型实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modular_qwen3_5.py)
7. [Transformers：Qwen3-Next 模型实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/modeling_qwen3_next.py)
8. [Transformers：Qwen3-VL 视觉实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py)
9. [Qwen3-Next：Towards Ultimate Training & Inference Efficiency](https://qwen.ai/blog?id=qwen3-next)
10. [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464)
11. [Qwen3.5-27B `config.json`](https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/config.json)
12. [Qwen3.6-27B `config.json`](https://huggingface.co/Qwen/Qwen3.6-27B/blob/main/config.json)
13. [Qwen3.8-27B `tokenizer_config.json`](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/tokenizer_config.json)
14. [Transformers：Qwen2 tokenizer 实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/tokenization_qwen2.py)
