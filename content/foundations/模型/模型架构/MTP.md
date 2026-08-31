# 从 Next-Token Prediction 到 MTP：让模型同时学习多个未来 Token

> 本文讨论大语言模型中的 MTP（Multi-Token Prediction，多 Token 预测）。文章面向知道 token、hidden state、Transformer 和交叉熵，但尚未系统了解 MTP 的读者；从 Next-Token Prediction 出发，依次解释通用目标函数、Meta 的并行预测头、DeepSeek 的顺序 MTP 模块、训练实现与 speculative decoding。资料核对日期为 2026 年 8 月 27 日。

一台普通自回归语言模型在位置 $t$ 只接受一个问题：

> 已知 $x_{\le t}$，下一个 token $x_{t+1}$ 是什么？

MTP 给同一个 hidden state 增加了几道更远的问题：

> 再下一个 $x_{t+2}$ 呢？第三个 $x_{t+3}$ 呢？

这看起来只是把 label 多平移几次，实际却同时影响两件事：

1. **训练**：主干必须让当前表示包含足够信息，以支持多个预测距离上的判断；
2. **推理**：额外模块可以充当模型内置的 drafter，一次提出多个候选 token，再由主模型并行验证。

先给出全文最重要的边界：

> **MTP 首先是一种多预测距离的训练机制；它可以为 speculative decoding 提供 draft，但不等于模型可以不经验证地一次提交多个 token。**

---

## 1. 先把三个问题分开

围绕 MTP 的讨论经常把训练目标、网络结构和解码算法混成一件事。它们其实是三个层次：

| 层次 | 要回答的问题 | MTP 中的典型做法 |
| --- | --- | --- |
| 训练目标 | 每个位置监督哪些未来 token？ | 同时对 $x_{t+1},\ldots,x_{t+n}$ 计算 loss |
| 网络结构 | 用什么模块产生这些分布？ | 并行独立 heads，或顺序连接的 MTP modules |
| 解码算法 | 候选 token 怎样成为最终输出？ | draft 后由主模型 verify，接受合法前缀 |

因此，“某模型使用 MTP”至少可能有三种含义：

- 它在预训练时使用过多 token 辅助 loss，但发布时删除了辅助模块；
- checkpoint 保留了 MTP 权重，普通推理仍只使用主 LM head；
- 推理后端显式加载 MTP 模块，并用它执行 speculative decoding。

三者的模型能力、显存占用和生成速度都不同。看到配置里有 `mtp_num_hidden_layers` 或 `num_nextn_predict_layers`，只能确认配置声明了 MTP 层数与预期结构；tensor 是否实际存在仍需核对 checkpoint，更不能直接推出当前请求已经被加速。

## 2. Next-Token Prediction 到底只预测了什么

给定 token 序列：

$$
x_1,x_2,\ldots,x_T,
$$

因果 Transformer 在位置 $t$ 产生 hidden state：

$$
h_t=f_\theta(x_{\le t}).
$$

普通 Next-Token Prediction（NTP）通过 LM head 得到下一个 token 的分布：

$$
p_t^{(1)}
=
\operatorname{softmax}(W_u h_t),
$$

并最小化交叉熵：

$$
\mathcal L_{\mathrm{NTP}}
=
-\frac{1}{T-1}
\sum_{t=1}^{T-1}
\log p_t^{(1)}[x_{t+1}].
$$

这里有一个容易被“next token”这个名字遮住的事实：**训练并不是一次前向只计算一个位置。** 在 causal mask 下，一整段序列的所有 $h_t$ 可以并行算出，因此长度为 $T$ 的样本已经提供了约 $T$ 个 next-token loss。

“只预测一个 token”准确地说是：

> 对于同一个位置的上下文表示 $h_t$，监督目标只有预测距离为 1 的 $x_{t+1}$。

### 2.1 一个具体的 target shift

假设训练序列是：

今天 → 天气 → 很 → 好 → 。

普通 NTP 的对齐方式是：

```text
hidden state     h(今天)  h(天气)  h(很)  h(好)
target             天气      很      好      。
预测距离              1       1       1       1
```

如果改成 3-token prediction，同一个位置会收到三组目标：

| 上下文截止位置 | 距离 1 | 距离 2 | 距离 3 |
| --- | --- | --- | --- |
| 今天 | 天气 | 很 | 好 |
| 天气 | 很 | 好 | 。 |
| 很 | 好 | 。 | 无有效目标 |
| 好 | 。 | 无有效目标 | 无有效目标 |

序列尾部没有足够远的未来 token，因此不同预测距离拥有不同数量的有效 label；实现时必须切片或 mask，不能让 padding、越界位置混入 loss。

## 3. 先用并行 future heads 写出 MTP 目标

MTP 并不限定所有架构采用同一种条件分解。本节先以“同一个共享位置表示直接连接多个 future heads”的并行结构统一 horizon 记号；DeepSeek 式顺序 module 还会读取中间真实 token，将在 4.2 节单独定义。

本文先用 $n$ 表示“总共预测多少个未来位置”，其中距离 1 就是普通 next token。对每个预测距离 $k\in\{1,\ldots,n\}$，并行 head 输出：

$$
p_t^{(k)}[v]
=
P_\theta(X_{t+k}=v\mid x_{\le t}),
\qquad v\in\mathcal V.
$$

先定义第 $k$ 个 horizon 的有效位置集合：

$$
\mathcal I_k
=
\left\{
t\in\{1,\ldots,T-k\}
\mid m_{t,k}=1
\right\},
$$

其中 $m_{t,k}$ 还会屏蔽 padding 或不允许跨越的样本边界。这样不会在数学上引用越界的 $x_{t+k}$。下式假设纳入目标的 $\mathcal I_k$ 非空；空 horizon 在实现中应跳过。带 horizon 权重的目标可以写成：

$$
\mathcal L_k
=
-
\frac{1}{|\mathcal I_k|}
\sum_{t\in\mathcal I_k}
\log p_t^{(k)}[x_{t+k}],
$$

$$
\boxed{
\mathcal L_{\mathrm{MTP}}
=
\frac{
\sum_{k=1}^{n}\lambda_k\mathcal L_k
}{
\sum_{k=1}^{n}\lambda_k
}
}.
$$

其中：

- $m_{t,k}\in\{0,1\}$ 表示位置 $t$ 是否拥有合法的第 $k$ 个未来 target；
- $\lambda_k\ge0$ 控制不同预测距离的权重，并要求 $\sum_k\lambda_k>0$；
- $p_t^{(1)}$ 通常就是正式的 next-token 分布；
- 距离越远，条件熵通常越高，预测也更难。

这套写法不是唯一约定。有的论文直接把所有 head loss 相加，有的先对预测深度求平均，再乘一个辅助系数。比较论文或配置时，必须确认它说的 $n$、$D$ 和 loss normalization 分别是什么。

### 3.1 $n$ 与 $D$ 的常见口径差异

Meta 的 MTP 论文将 $n$ 定义为**包含 next token 在内的总预测数**：

$$
x_{t+1},x_{t+2},\ldots,x_{t+n}.
$$

DeepSeek-V3 则将 $D$ 定义为**主模型之外的额外预测深度**。主模型先预测 $x_{t+1}$，第 $1$ 个 MTP module 再预测 $x_{t+2}$。因此：

$$
\text{DeepSeek 总预测数}=D+1.
$$

所以 DeepSeek-V3 的 $D=1$ 不是“总共只预测一个 token”，而是“主模型预测 next token，再额外预测一个 token”，总预测范围为 2。

### 3.2 主干梯度为什么会改变

设第 $k$ 个 head 为 $g_k$，其 logits 是：

$$
z_t^{(k)}=g_k(h_t).
$$

以并行 heads 为例，共享主干在位置 $t$ 接收到的梯度是多个预测距离贡献的和：

$$
\frac{\partial\mathcal L_{\mathrm{MTP}}}{\partial h_t}
=
\frac{1}{\sum_{j=1}^{n}\lambda_j}
\sum_{k=1}^{n}
\lambda_k
\frac{\partial\mathcal L_k}{\partial h_t}.
$$

这正是 MTP 能影响主模型能力的机制：辅助 head 即使最终被删除，它们在训练期间已经改变了共享 trunk 的优化方向。

但“更多梯度”不自动等于“更好的梯度”。如果远期目标太难、权重过大，多个 horizon 可能互相干扰；因此预测距离和 loss 权重都是需要实验选择的超参数。

## 4. 两条主流架构路线

“同时预测多个未来 token”并没有规定 head 必须怎样连接。最有代表性的两类实现，是 Meta 论文中的并行独立 heads，以及 DeepSeek-V3 中保留完整因果链的顺序 modules。

### 4.1 Meta：共享主干 + 并行独立预测头

《Better & Faster Large Language Models via Multi-token Prediction》采用下面的结构：

```text
                         ┌─ head 1 ─ shared unembedding ─ x[t+1]
tokens ─ shared trunk ─ h├─ head 2 ─ shared unembedding ─ x[t+2]
                         ├─ head 3 ─ shared unembedding ─ x[t+3]
                         └─ head n ─ shared unembedding ─ x[t+n]
```

共享 Transformer trunk $f_s$ 先产生上下文表示，再由 $n$ 个独立 Transformer head $f_{h_k}$ 和共享 unembedding $f_u$ 输出：

$$
p_t^{(k)}
=
\operatorname{softmax}
\left(
f_u(f_{h_k}(f_s(x_{\le t})))
\right).
$$

这组 heads 是并行的：第 3 个 head 直接从同一个上下文表示预测 $x_{t+3}$，不会先读取第 1、2 个 head 预测了什么。

因此，它学习的是一组条件边缘分布：

$$
P(x_{t+k}\mid x_{\le t}),
\qquad k=1,\ldots,n,
$$

而不是标准自回归联合分解：

$$
P(x_{t+1:t+n}\mid x_{\le t})
=
\prod_{k=1}^{n}
P(x_{t+k}\mid x_{\le t+k-1}).
$$

并行 heads 提出的各位置 argmax 可能组成一条不够连贯的候选序列；这不是训练 bug，而是条件信息不同造成的结构性限制。用于加速时，主模型的 verify 阶段会过滤不一致的后缀。

还要注意论文中“没有训练时间和显存开销”的实验口径。作者为了固定总参数量和计算预算，在增加 $n-1$ 个预测层时，从共享 trunk 移除了相同数量的层。它并不意味着给任意现成模型直接追加多个完整 heads 都是免费的。

显存方面，论文通过逐 head 前向/反向避免同时保存所有词表 logits。论文附录中的实际实现仍报告了小幅到约 22% 的训练时间开销，且模型越大相对开销通常越小。

### 4.2 DeepSeek：顺序 MTP modules 保留因果链

DeepSeek-V3 认为独立 heads 没有显式保留不同预测深度之间的因果依赖，因此改用 $D$ 个顺序模块：

```text
主模型：
x[≤i] ─ backbone ─ h_i^0 ─ main head ─ predict x[i+1]

第 1 个 MTP module：
h_i^0 + Emb(x[i+1]) ─ fusion ─ Transformer block ─ h_i^1 ─ predict x[i+2]

第 2 个 MTP module：
h_i^1 + Emb(x[i+2]) ─ fusion ─ Transformer block ─ h_i^2 ─ predict x[i+3]
```

对第 $k$ 个预测深度，先把上一深度的表示 $h_i^{k-1}$ 与真实 token $x_{i+k}$ 的 embedding 分别归一化、拼接，再投影回 hidden size：

$$
h_i^{\prime k}
=
M_k
\left[
\operatorname{RMSNorm}(h_i^{k-1});
\operatorname{RMSNorm}(\operatorname{Emb}(x_{i+k}))
\right].
$$

随后经过该深度自己的 Transformer block：

$$
h_{1:T-k}^{k}
=
\operatorname{TRM}_k(h_{1:T-k}^{\prime k}),
$$

最后用与主模型共享的输出 head 预测再往后一个 token：

$$
P_{i+k+1}^{k}
=
\operatorname{OutHead}(h_i^k).
$$

这里最容易发生 off-by-one 错误。以 $k=1$ 为例：

- 输入上一深度的 $h_i^0$；
- 同时输入 $x_{i+1}$ 的 embedding；
- 输出目标是 $x_{i+2}$。

训练时的 $x_{i+1}$ 来自 ground truth，这仍属于 teacher forcing。推理时它会换成刚刚生成的候选 token，因此模块具备沿生成链继续向前 draft 的接口。

DeepSeek 对每个额外深度计算交叉熵。下面沿用技术报告的下标约定：令 $j=i+k+1$，输入长度记为 $T$，右移后的语言建模 target 记到 $x_{T+1}$，并统一用 $T$ 归一化：

$$
\mathcal L_{\mathrm{MTP}}^k
=
-\frac{1}{T}
\sum_{j=2+k}^{T+1}
\log P_j^k[x_j],
$$

再将不同深度平均并乘辅助权重：

$$
\mathcal L_{\mathrm{aux}}
=
\frac{\lambda}{D}
\sum_{k=1}^{D}
\mathcal L_{\mathrm{MTP}}^k.
$$

完整训练还包括主模型的 next-token loss。DeepSeek-V3 设置 $D=1$，并把 $\lambda$ 从前 10T token 的 0.3 降到后 4.8T token 的 0.1。这个 schedule 可以理解为一种常见取舍：早期用较强的远期监督塑造表示，后期降低辅助目标对正式 next-token 目标的干扰；技术报告本身没有单独证明这种解释。

### 4.3 Qwen：单模块也可以进行多步训练

Qwen3-Next 延续了原生 MTP 思路，并特别强调 multi-step training。官方公开结论是，这种训练维持了训练与推理的一致性，并提高真实多步 speculative decoding 的接受率；直觉上，它需要让模块适应多步使用产生的输入分布。公开材料没有披露完整训练计算图，因此不能进一步断言它具体采用了哪一种 scheduled sampling 或递归展开配方。后续 Qwen3.5—Qwen3.8 系列继续使用这一设计。

以 [[Qwen3.8-27B 模型架构]] 为例，公开配置保留 1 层 MTP 辅助模块。这里的“1 层”是权重结构，不必然等于推理时最多只能 draft 1 步：推理后端可以递归使用一个模块，但多步效果取决于训练配方、缓存实现和后端是否支持。反过来，即使 checkpoint 带有这层权重，普通 `generate()` 仍可以完全不调用它。

### 4.4 三种口径放在一起

| 项目 | Meta 并行 MTP | DeepSeek 顺序 MTP | Qwen 原生 MTP |
| --- | --- | --- | --- |
| future head 关系 | 各 head 从共享 trunk 独立预测 | 按深度顺序连接 | 顺序 draft，并强调多步训练 |
| 第 $k$ 步是否读取前一步 token | 否 | 是；训练时读 ground truth embedding | 多步训练中尽量贴近推理使用 |
| embedding / LM head | 共享 unembedding | 与主模型共享 embedding 和 output head | 通常与主模型共享体系 |
| 主要目标 | 改善预训练，也可 self-speculate | 首先改善主模型，也可 speculate | 同时改善 backbone 与 draft 接受率 |
| 普通推理能否不调用模块 | 可以 | 可以 | 可以，但会失去 MTP draft |

## 5. MTP 为什么可能改善模型能力

“同时看得更远，所以更会规划”是一个好直觉，但不是严格定理。当前证据主要来自受控实验和大型模型消融；更稳妥的理解可以分成四层。

### 5.1 同一个表示获得更密集的监督

在并行 heads 中，NTP 让 $h_t$ 只对 $x_{t+1}$ 负责，MTP 让它还必须直接支持 $x_{t+2},x_{t+3},\ldots$ 的预测；在顺序 modules 中，额外 loss 则穿过 module chain 回传到主模型表示。对于长度为 $T$ 的序列，监督项从约 $T$ 个增加到约：

$$
nT-\frac{n(n+1)}{2},
$$

这里忽略了 BOS/EOS 和 padding，并假设每个距离都覆盖到序列尾部。数据本身没有变多，但每个样本产生了更多“上下文—未来目标”配对。

这会迫使共享表示保留两类信息：

- 对紧邻 token 有用的局部句法与搭配；
- 对更远 continuation 有用的语义、结构和计划。

### 5.2 更重视会改变后续走向的 choice point

并非每个 token 决策都同等重要。代码中的函数名、分支方向、循环变量，文章中的论点转折，都可能决定后面一长段内容；空格或常见标点往往只是局部容易预测的过渡。

Meta 论文将前者称作 choice point。若一个难预测的选择会影响随后多个 token，那么 MTP 会在多个 horizon 上重复感受到这个选择的后果。论文的简化计数表明，$n$-token prediction 对 choice point 及其相关后续项赋予约：

$$
\frac{n(n+1)}{2}
$$

的累计权重，而普通、彼此无关的局部转移约为 $n$。这不是训练代码中显式写入的 class weight，而是多个 shifted loss 叠加形成的隐式效果。

从信息论角度，令 $X$ 和 $Y$ 分别为接下来两个 token，省略共同上下文 $C$。有：

$$
H(X)=H(X\mid Y)+I(X;Y),
$$

$$
H(X)+H(Y)
=
H(X\mid Y)+2I(X;Y)+H(Y\mid X).
$$

$H(Y\mid X)$ 对应“已经看见 $X$ 后再预测 $Y$”的难度，它会在相邻位置的 next-token loss 中再次出现。按论文用于解释相对权重的比较暂时扣除这项后，2-token prediction 会把互信息 $I(X;Y)$ 的系数从 1 提到 2。这个分解提供的是直觉，不是 MTP 必然改善所有任务的证明。

### 5.3 它缓解局部 teacher forcing 偏好，但没有消除训练—推理差异

teacher forcing 训练时总能看到真实历史；自由生成时，模型只能继续读取自己刚生成的 token，错误可能累积。只优化一步预测容易让模型依赖短程线索，MTP 则要求当前表示提前解释更远未来，因此可能减少这种局部短视。

但它没有彻底消除 exposure bias：

- 并行 heads 训练时直接面对真实的远期 label，不读取自己的中间预测；
- DeepSeek 顺序模块训练时读取真实 $x_{i+k}$ 的 embedding，推理时读取候选 $\hat x_{i+k}$；
- 一旦某个 draft token 错误，后续模块面对的 hidden state 分布仍可能偏离训练分布。

这也是 Qwen 强调 multi-step training 与训练—推理一致性的原因。最终的主模型验证仍然不可省略。

### 5.4 收益取决于规模、数据和任务

更远的 token 天生更不确定。自然语言中一句话可以有许多同样合理的续写，代码和形式化推理的后续约束通常更强。因此 MTP 的收益常在代码、算法和结构化生成上更明显。

预测距离也不是越远越好：

- 小模型可能没有容量同时完成多个困难目标；
- 太远的 target 会带来高方差或相互冲突的梯度；
- tokenizer 粒度越细，同样的 $n$ 覆盖的真实语义距离越短；
- 数据分布不同，最佳 $n$ 也会不同。

MTP 更像一种可扩展的辅助监督配方，而不是对任何模型都单调有效的定律。

## 6. 训练实现：真正困难的是对齐和显存

从数学上看，MTP 只是多做几次 target shift；工程上最常见的错误却正是 shift、mask、因果性和词表 logits 的峰值显存。

### 6.1 一个只展示对齐逻辑的 PyTorch 实现

下面函数假设 `logits_by_horizon[k - 1]` 的形状都是 `[B, T, V]`，其中位置 $t$ 预测 $x_{t+k}$。它只演示 loss 对齐，不代表高性能训练实现：

```python
import torch
import torch.nn.functional as F


def multi_token_loss(
    logits_by_horizon,     # 长度 n；每个元素形状 [B, T, V]
    input_ids,             # [B, T]
    token_mask=None,       # [B, T]，1 表示真实 token
    horizon_weights=None,  # 长度 n
):
    """计算包含 next-token head 在内的平均 MTP loss。"""
    batch_size, seq_len = input_ids.shape
    num_horizons = len(logits_by_horizon)

    if token_mask is None:
        token_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        token_mask = token_mask.bool()

    if horizon_weights is None:
        horizon_weights = [1.0] * num_horizons
    if len(horizon_weights) != num_horizons:
        raise ValueError("horizon_weights 与 logits 数量必须一致")

    weighted_loss = input_ids.new_zeros((), dtype=torch.float32)
    total_weight = 0.0

    for horizon, (logits, weight) in enumerate(
        zip(logits_by_horizon, horizon_weights),
        start=1,
    ):
        if horizon >= seq_len:
            continue

        # h_t 对齐 x_{t+horizon}
        pred = logits[:, : seq_len - horizon, :]
        target = input_ids[:, horizon:]

        # 输入位置和目标位置都必须有效。
        valid = (
            token_mask[:, : seq_len - horizon]
            & token_mask[:, horizon:]
        )
        if not valid.any().item():
            continue

        token_nll = F.cross_entropy(
            pred.float().reshape(-1, pred.size(-1)),
            target.reshape(-1),
            reduction="none",
        ).view(batch_size, seq_len - horizon)

        valid_count = valid.sum().clamp_min(1)
        loss_h = (token_nll * valid).sum() / valid_count
        weighted_loss = weighted_loss + float(weight) * loss_h
        total_weight += float(weight)

    if total_weight == 0.0:
        raise ValueError("没有合法的 horizon target")
    return weighted_loss / total_weight
```

生产代码还必须处理：

- packed samples 之间能否互相注意，以及 target 是否允许跨样本边界；
- BOS、EOS、padding 和被截断文档的 loss mask；
- tensor parallel 下共享 LM head 的通信；
- mixed precision、vocab parallel cross entropy 与 sequence parallel；
- 不同 horizon 的样本数不同，究竟按 token、按 horizon 还是按样本归一化。

如果 pack 中的样本被要求彼此隔离，仅检查 padding mask 还不够；还要确认输入位置 $t$ 与目标位置 $t+k$ 属于同一个 sample，否则 MTP 会跨边界偷学一条不存在的 continuation。

### 6.2 因果 mask 不能因为“预测未来”而放宽

对并行 direct head 而言，MTP 的目标在未来，但输入仍只能看到当前及过去。对任意词表项 $v$：

$$
p_t^{(k)}[v]
=
P_\theta(X_{t+k}=v\mid x_{\le t}).
$$

如果预测 head 是 Transformer layer，它也必须使用与位置 $t$ 对齐的 causal mask。让 head 在训练时 attention 到 $x_{t+1}$ 之后的信息，会造成 label leakage；loss 会很好看，推理时却没有这些输入。

DeepSeek 式模块读取 $x_{i+k}$ 并不违反因果性，因为它预测的是更后面的 $x_{i+k+1}$。在真实生成链中，模型准备预测 $x_{i+k+1}$ 时，$x_{i+k}$ 本来就已经生成。关键是位置和预测目标必须一起向前移动。

### 6.3 为什么 logits 会成为显存瓶颈

假设：

- batch size $B=1$；
- sequence length $T=4096$；
- vocabulary $V=131{,}072$；
- 预测 $n=4$ 个未来位置；
- logits 使用 BF16。

仅同时物化 logits 就需要：

$$
1\times4096\times131{,}072\times4\times2\ \text{bytes}
=
4\ \text{GiB}.
$$

这还没有计算 softmax 中间量、梯度和其他激活。朴素地把所有结果堆成 $[B,T,n,V]$ 很容易让词表维成为显存峰值。

Meta 的做法是在共享 trunk 前向后，依次执行每个 head 的 forward 和 backward，把对 trunk hidden state 的梯度累积起来；当前 head 的 logits 释放后再计算下一个。理想情况下，峰值从：

$$
O(nV+d)
$$

降到：

$$
O(V+d),
$$

其中 $d$ 是 hidden size。真实分布式实现还要协调参数通信、autograd graph 和梯度累积，因此不能只在 Python 中套一个 `for` 循环就假定没有额外开销。

### 6.4 四个值得单独监控的训练指标

只看总 loss 会掩盖很多问题。至少应分别记录：

1. 正式 next-token loss / perplexity；
2. 每个 horizon 的 cross entropy 或准确率；
3. 不同 head 对共享 trunk 的梯度范数及夹角；
4. 在真实多步 draft 路径上的接受长度，而不只是 teacher-forced head accuracy。

第 4 项尤其重要。一个模块在输入真实前序 token 时预测得很好，不代表它在读取自己生成的候选 token 后仍能稳定向前滚动。

## 7. MTP 怎样变成 speculative decoding

标准自回归 decode 每轮只确认一个 token：

```text
prefix ─ 主模型 forward ─ 确认 1 个 token ─ 主模型 forward ─ 再确认 1 个 token
```

瓶颈不只来自 FLOPs。batch 较小时，每一步都要把大模型权重从高带宽显存读入计算单元，却只为一个位置服务，decode 往往是 memory-bound。speculative decoding 的思路是用便宜模块先猜，再让昂贵主模型一次检查一段：

```text
                         ┌─ draft y1, y2, ..., yK
prefix ─ MTP proposer ──┤
                         └─ 主模型一次并行 verify
                                      │
                            接受最长合法前缀
                                      │
                          从第一个拒绝位置继续
```

主模型之所以能“一次验证多个位置”，是因为候选序列已经作为输入给出。causal mask 让验证位置 $j$ 只能看到 prefix 和候选的前 $j-1$ 个 token，于是一次 block forward 可以并行计算：

$$
P(y_j\mid x_{\le t},y_{<j}),
\qquad j=1,\ldots,K.
$$

这和训练时并行计算整段 next-token loss 是同一个并行性来源：输入 token 已知时，多个位置可以同时评分；真正困难的是生成时未来 token 原本未知，drafter 先提供了可验证的猜测。

### 7.1 Greedy decoding：逐位置比对 argmax

对 greedy decoding，可以使用下面的严格流程：

1. MTP 提出候选 $\hat y_1,\ldots,\hat y_K$；
2. 主模型在对应候选前缀下并行算出每个位置的 argmax；
3. 从左到右比较，接受第一个不一致位置之前的所有 token；
4. 在不一致位置提交主模型自己的 argmax；
5. 在本文后续采用的标准算法中，如果全部一致，再使用 verify forward 最后一个位置给出的分布产生一个 bonus token。

之所以只能接受**最长前缀**，是因为一旦 $\hat y_j$ 被拒绝，后面候选都是在错误前缀上产生或验证的，不能跳过错误位置继续接受。

只要 verifier 使用与基线相同的 logit processors、生成约束、数值精度和 argmax tie-breaking，这个流程可以产生与逐 token greedy decode 相同的输出。若直接提交未经主模型检查的多个 head argmax，生成结果已经改变，不能再称为无损 speculative decoding。

### 7.2 随机采样：无损来自接受—纠正协议

温度采样、top-k 或 top-p 场景不能只比较 argmax。经典 speculative sampling 设 drafter 分布为 $q$、目标模型分布为 $p$，候选 $x\sim q$ 的接受概率是：

$$
a(x)
=
\min\left(1,\frac{p(x)}{q(x)}\right).
$$

如果拒绝，则从归一化后的残差分布采样：

$$
p_{\mathrm{res}}(x)
=
\frac{[p(x)-q(x)]_+}
{\sum_y[p(y)-q(y)]_+}.
$$

如果整段 $K$ 个候选都被接受，标准算法还会从主模型在下一个位置的分布中采样一个 bonus token。这套拒绝采样、纠正与 bonus 规则保证最终样本仍来自目标分布 $p$。

给定某个 prefix $c$，该位置的条件接受概率为：

$$
\beta(c)
=
\sum_x\min(p(x\mid c),q(x\mid c))
=
1-\operatorname{TV}
\left(
p(\cdot\mid c),q(\cdot\mid c)
\right).
$$

跨实际生成 prefix 的平均接受率才记为：

$$
\alpha
=
\mathbb E_c[\beta(c)].
$$

因此，draft 与 target 分布越接近，总变差距离越小，平均接受率通常越高。

这里不能把公式机械套在任何一组“未来 logits”上。并行 MTP heads 给出的是基于同一旧 prefix 的多个边缘分布；若要做严格随机采样，需要推理算法为候选树或每一步定义合法的条件 proposal，并在经过 temperature、top-k、top-p 等变换后执行匹配的拒绝纠正。**无损性来自完整 decoder 协议，不来自 checkpoint 上写了 MTP。**

### 7.3 一次 verify 平均能前进多少 token

下面采用标准 speculative decoding 的记账约定：若中途拒绝，就提交一个 verifier 纠正 token；若 $K$ 个 draft 全部接受，就再提交一个 bonus token；暂时忽略 EOS 和最大生成长度在 block 中间截断的边界。令 $A\in\{0,\ldots,K\}$ 表示连续被接受的 draft token 数，则一次 verify 的提交量为 $A+1$：

$$
\mathbb E[N_{\mathrm{commit}}]
=
1+\mathbb E[A]
=
1+\sum_{j=1}^{K}P(A\ge j).
$$

如果粗略假设每一深度的条件接受率都等于 $\alpha$，则：

$$
\mathbb E[N_{\mathrm{commit}}]
\approx
1+\alpha+\alpha^2+\cdots+\alpha^K
=
\frac{1-\alpha^{K+1}}{1-\alpha}.
$$

当 $\alpha=1$ 时，上式按极限取值为 $K+1$。

例如 $K=3,\alpha=0.85$ 时，理想平均前进量约为：

$$
1+0.85+0.85^2+0.85^3
\approx
3.19\ \text{tokens}.
$$

这不代表速度一定是 $3.19\times$。设普通一步 decode 成本为 $C_{\mathrm{AR}}$，一次 draft、verify 和调度成本分别为 $C_{\mathrm{draft}}(K)$、$C_{\mathrm{verify}}(K)$ 与 $C_{\mathrm{misc}}$，更接近真实的粗略模型是：

$$
S
\approx
\frac{
\mathbb E[N_{\mathrm{commit}}]C_{\mathrm{AR}}
}{
C_{\mathrm{draft}}(K)
+C_{\mathrm{verify}}(K)
+C_{\mathrm{misc}}
}.
$$

验证 $K$ 个位置的计算量高于普通单 token decode，只是它能更好地并行和摊薄权重读取。随着 $K$ 增大：

- 可接受的 token 数先增加；
- 深层候选的正确率通常下降；
- drafter、KV cache、attention 和调度成本继续上升。

所以速度通常会在某个 draft depth 达到峰值，而不是候选越多越快。

### 7.4 哪些场景更容易加速

MTP speculative decoding 更适合：

- 低到中等 QPS、单步 decode 受显存带宽限制；
- draft 模块远小于主模型；
- 代码、模板化文本等 continuation 较确定的任务；
- 较低温度、较高候选接受率；
- 后端有融合良好的 draft、verify、cache 与 rejection sampling kernel。

它可能收益很小甚至变慢的场景包括：

- 高并发已经让主模型 GEMM 充分 compute-bound；
- 在不少模型与任务中，高温采样会降低 proposal 接受率，但温度与 $p,q$ 距离没有普遍的单调关系，必须按实际采样配置测试；
- draft 太深、MTP module 本身很重；
- 不同请求接受长度差异大，batch 出现分歧和 padding；
- 框架只能加载权重，却没有高效 MTP kernel。

MTP 主要优化 decode 阶段，通常不会让长 prompt 的 prefill 自动变快；额外模块甚至可能增加少量状态准备和缓存成本。

## 8. 训练收益与推理加速是两条独立路径

同一个 MTP checkpoint 可以有三种使用模式：

| 模式 | MTP module | 输出路径 | 主模型能力 | 推理成本 |
| --- | --- | --- | --- | --- |
| MTP 训练 | 保留并参与反向传播 | 多个 horizon 都算 loss | 主干被辅助目标塑形 | 增加训练计算与参数 |
| 普通自回归推理 | 删除或不调用 | 只用正式 next-token head | 保留训练后的主干能力 | 与普通主模型基本相同 |
| MTP speculative 推理 | 加载并调用 | draft → verify → accept | 目标分布由 verifier 决定 | 多一些计算，少一些串行步 |

这张表解释了两个看似矛盾的说法为什么都成立：

- “MTP module 推理时可以丢弃”：对，只保留训练收益时可以；
- “MTP 可以加速推理”：也对，但必须保留模块并接入 speculative decoder。

同样，speculative decoding 本身不会让答案更聪明。若验证算法严格保持主模型分布，它只改变计算顺序；模型能力提升来自更早的 MTP 训练对 backbone 的影响。

## 9. 论文与真实模型给出了什么证据

### 9.1 Meta：收益随规模出现，代码任务最明显

Meta/FAIR 在 300M 到 13B 的 decoder-only 代码模型上做了等参数、等计算预算比较。13B 模型中，4-token prediction 相对 NTP 的结果包括：

| Benchmark | NTP $n=1$ | MTP $n=4$ | 变化 |
| --- | ---: | ---: | ---: |
| MBPP pass@1 | 26.0 | 30.5 | +4.5 个百分点，约 +17.3% 相对提升 |
| HumanEval pass@1 | 14.1 | 15.8 | +1.7 个百分点，约 +12.1% 相对提升 |

论文摘要中的“多解决 12% / 17% 问题”指相对提升，不是 12 或 17 个百分点。

实验同时给出了重要反例：

- 300M、600M 等小模型使用 MTP 往往比 NTP 更差，优势随模型规模增大才出现；
- 7B、200B-token 的代码实验中，HumanEval 与 MBPP 的最佳值是 $n=4$，APPS/Intro 则更偏好 $n=6$；
- 7B 自然语言多选评测中，$n=2$ 大致持平，$n=4$ 有所退化；
- 摘要生成有所改善，但 GSM8K 的相对顺序会随训练 token 数变化；
- 直接把已有 Llama 2 checkpoint 切换到 MTP 微调没有得到稳定收益。

因此，$n=4$ 是这组 tokenizer、数据和模型规模上的经验结果，不是通用常数。

在 self-speculative decoding 实验中，7B 的 4-token 模型在代码上对 3 个额外建议平均接受 2.5 个，报告约 $3.0\times$ 加速；文本任务约为 $2.7\times$。这些数字来自论文的指定实现和硬件，不应直接当作部署承诺。

### 9.2 DeepSeek-V3：一个额外深度也能改善多数指标

DeepSeek 在两个 MoE 规模上比较 baseline 与 1-depth MTP，并在评测时直接丢弃 MTP module，因此两组普通推理成本相同。部分结果如下：

| 模型规模 | Benchmark | Baseline | w/ MTP |
| --- | --- | ---: | ---: |
| 15.7B total / 2.4B active | HumanEval pass@1 | 20.7 | 26.8 |
| 15.7B total / 2.4B active | GSM8K | 25.4 | 31.4 |
| 228.7B total / 20.9B active | HumanEval pass@1 | 44.5 | 53.7 |
| 228.7B total / 20.9B active | DROP | 68.5 | 70.6 |
| 228.7B total / 20.9B active | MMLU | 67.5 | 66.6 |

多数指标改善不意味着所有指标都改善：大模型 MMLU 从 67.5 降到 66.6，小模型 NaturalQuestions 也从 22.7 降到 22.3。更准确的结论是“这套配方在该消融中整体有利”，而不是 MTP 对每项能力单调提升。

DeepSeek-V3 正式配置的 $D=1$。技术报告称额外 token 在不同生成主题上的接受率为 85%—90%，结合 speculative decoding 得到约 $1.8\times$ TPS。这里的 acceptance rate 是“draft 是否与 verifier 一致”，不是事实正确率或 benchmark accuracy。

公开权重把 MTP module 一并发布：官方仓库将下载规模概括为 685B，其中主模型为 671B；权重说明进一步区分出 MTP 的 11.5B 独有参数以及与主模型共享的 embedding 和 output head。这说明“推理可删除”不等于“训练和存储免费”。

### 9.3 Qwen：重点从单步准确率转向多步接受率

Qwen3-Next 官方说明把 MTP 的两个目标并列：增强 backbone，以及产生高接受率的 speculative token。它进一步加入 multi-step training，以减小只在 teacher-forced 单步输入上训练、却在推理时递归多步使用的分布差异。

这条设计随后延续到 Qwen3.5—Qwen3.8。已核对的 Qwen3.5-9B、Qwen3.8-27B 等官方 checkpoint 只声明一层 MTP module，但官方同时描述了多步训练。因此“模块层数”“训练预测步数”和“服务时 draft depth”是三个需要分别查看的量，不能由其中一个推断另外两个。

## 10. MTP 与相关方法是什么关系

MTP 不是凭空出现的。2018 年的 Blockwise Parallel Decoding 已经使用多个未来预测器加速 greedy decoding；2020 年 ProphetNet 在 Seq2Seq 预训练中使用 future n-gram prediction；2023 年的 speculative decoding 则给出了保持目标分布不变的通用采样协议。

现代几类方法的边界如下：

| 方法 | draft 从哪里来 | 是否改变 backbone 训练 | 是否显式建模前序候选 | 主要目的 |
| --- | --- | --- | --- | --- |
| 原生 MTP | 主模型自带 future heads / modules | 通常是，从预训练开始联合优化 | 并行版否；顺序版是 | 改善训练，也可加速 |
| Medusa-1 | 主模型上的多个 decoding heads | backbone 冻结 | 多头并行，并构造候选树 | 后训练推理加速 |
| Medusa-2 | 多头与 backbone 联合训练 | 是，需要保能力配方 | 多头并行 + tree verify | 更高接受率与加速 |
| EAGLE-1 / 2 | 轻量 draft 模型预测接近顶层的 feature | 否；目标 LLM 冻结，只训练 drafter | 是，递归读取提前一位的 token | 推理加速 |
| EAGLE-3 | 融合目标模型多层 feature 后直接预测 token | 否；目标 LLM 冻结，只训练 drafter | 是，自回归地产生 draft | 推理加速 |
| 外置 draft model | 一台独立小语言模型 | 否 | 是，标准自回归 | 通用 speculative decoding |
| 非自回归生成 | 模型直接并行提交多个位置 | 需要专门训练 | 依方法而定 | 减少或消除 AR 步骤 |

### 10.1 MTP 与 Medusa

两者最容易被混淆，因为都可以画成“共享主模型 + 多个未来 heads”。关键差异是训练时机和目标：

- Meta 式 MTP 从预训练开始用多个 future loss 塑造共享 trunk；
- Medusa-1 在已经训练好的 LLM 上冻结 backbone，只训练解码 heads；
- Medusa-2 会联合调整 backbone，但重点仍是候选树与推理接受率。

结构相似不代表训练得到的表示和使用场景相同。

### 10.2 MTP 与 EAGLE

EAGLE-1 / 2 不直接把远期 token 当作互相独立的分类问题，而是预测目标模型接近顶层的 feature，并把提前一位的真实或已采样 token 作为输入，递归产生后续 draft。DeepSeek 报告明确说，其“保持完整因果链”的原则类似 EAGLE，但 DeepSeek 首要用途是改善预训练，而 EAGLE 首要用途是 speculative sampling。

EAGLE-3 已经放弃 feature-prediction 训练目标：它在训练时融合目标模型低、中、高层的 feature，并直接优化 token prediction。它仍是在冻结 target LLM 后训练专用 drafter，因此不能用 EAGLE-1 的训练目标概括整个 EAGLE 系列。

### 10.3 MTP 不是非自回归语言模型

非自回归模型试图直接产生并提交多个位置；MTP speculative decoding 仍以自回归主模型为最终裁判。一次 verify 可以确认多个 token，只是因为候选前缀已经给出，并不代表 token 间的因果依赖消失了。

## 11. 训练与部署时怎样做选择

### 11.1 先明确想优化哪一个目标

如果目标只是改善主模型能力：

- 可以在训练结束后删除 MTP heads；
- 重点评估正式 next-token head 和下游生成质量；
- 不必为推理框架的 MTP 支持付出额外复杂度。

如果目标是 decode 加速：

- checkpoint 必须保留可用的 MTP 权重；
- 训练应报告多步 rollout 下的接受长度；
- 服务后端必须实现 draft、verify、拒绝纠正和 cache 管理；
- benchmark 必须覆盖实际 batch、上下文、采样参数和硬件。

### 11.2 $n$、$D$ 和 $\lambda$ 应联合调节

一套稳妥的 ablation 至少包括：

- $n=1$ 的 NTP baseline；
- 2—4 个较短 horizon，而不是一开始就预测很远；
- uniform 与随距离衰减的 $\lambda_k$；
- 固定总参数 / FLOPs 与直接追加模块两种公平口径；
- 从头预训练与从已有 checkpoint 切换目标的区别；
- 不同 tokenizer、代码 / 自然语言数据的分项结果。

距离 $k$ 个 token 并不是固定的语义长度。byte tokenizer 的 8 步可能只覆盖一个短单词，32K BPE 的 8 步则可能跨越大半句话，因此最佳 horizon 不能脱离 tokenizer 讨论。

### 11.3 部署 benchmark 不应只报一个“接受率”

至少同时报告：

| 指标 | 它回答的问题 |
| --- | --- |
| 每深度 conditional acceptance | 已接受前缀成立时，第 $k$ 个 draft 还能否被接受？ |
| 平均接受长度 | 每次 verify 实际省掉多少串行步？ |
| inter-token latency | 单请求逐 token 延迟是否下降？ |
| tokens/s 与 QPS | 并发吞吐是否真的提高？ |
| draft / verify 时间拆分 | 时间花在 MTP、主模型还是调度？ |
| 额外显存 | MTP 权重、KV cache 和 workspace 需要多少资源？ |
| 输出一致性 | greedy 是否逐 token 相同，采样分布是否通过校验？ |

“第二 token 接受率 90%”和“平均接受 2.5 个 token”不是同一个统计量；比较系统前必须统一分母和深度定义。

### 11.4 框架支持是运行时能力，不是模型属性

截至资料核对日期，vLLM 等后端已经为部分模型家族提供 `mtp` speculative method。其命令形态类似：

```bash
vllm serve <model> \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

这只是说明配置入口，不能保证任意带 MTP 字段的 checkpoint 都能运行。模型类、权重命名、专用 attention / MoE kernel、量化格式和框架版本都必须匹配。若不启用 MTP 配置，普通 `generate()` 走标准 next-token 路径；若显式为不受支持的模型启用 `method=mtp`，后端通常会报错，不能依赖静默回退。

配置字段也没有完全统一：

- DeepSeek 系列常见 `num_nextn_predict_layers`；
- Qwen3.5 / Qwen3.8 的 `text_config` 中常见 `mtp_num_hidden_layers`；
- `num_speculative_tokens` 往往是服务时希望 draft 的步数。

前两个描述 checkpoint 结构，最后一个描述运行策略，它们不必相等。

## 12. 十个常见误解

1. **“MTP 让 Transformer 训练时从一次预测一个位置变成一次预测整段。”** 不准确。NTP 训练本来就会在 causal mask 下并行计算整段位置；MTP 改变的是同一位置的预测 horizon 数。
2. **“MTP 把训练数据扩大了 $n$ 倍。”** 数据 token 没有增加，只是同一序列产生了更多 shifted supervision。
3. **“所有 MTP 都是多个独立 LM heads。”** Meta 是并行独立 heads；DeepSeek 和 Qwen 的原生模块保留了更强的顺序依赖。
4. **“Meta 发明了多 token 预测。”** 不是。Blockwise Parallel Decoding、ProphetNet 等工作更早使用了相近思想；Meta 的贡献在于把它系统扩展到大规模 decoder-only 预训练。
5. **“DeepSeek 的 $D=1$ 表示只预测 next token。”** 它表示主 next token 之外再预测一个 token，总范围为 2。
6. **“checkpoint 有 MTP head，普通 `generate()` 就会自动更快。”** 只有显式支持并启用它的推理后端才会走 draft–verify 路径。
7. **“多个 head 的 argmax 可以直接一次输出。”** 并行 heads 的候选可能不一致；跳过 verifier 会改变生成结果或分布。
8. **“MTP speculative decoding 天然无损。”** 无损来自严格 greedy 验证，或随机采样中的接受—残差纠正规则；近似接受策略可能主动改变分布。
9. **“MTP 一定提高能力，也一定提高速度。”** 两者都依赖规模、数据、预测距离、接受率、batch、硬件和 kernel，论文中也存在退化项。
10. **“推理时删掉 MTP module，就浪费了它的训练。”** 辅助梯度已经改变 backbone；删除模块只放弃 draft 能力，不会抹去主干学到的表示。

## 13. 总结

理解 MTP，可以抓住三条主线。

第一，以并行 future heads 为例，训练目标从单一 horizon：

$$
P(x_{t+1}\mid x_{\le t})
$$

扩展到多个 horizon：

$$
P(x_{t+k}\mid x_{\le t}),
\qquad k=1,\ldots,n.
$$

同一个 hidden state 因此接收到局部与远期监督，可能更重视会影响后续走向的关键信息。

第二，MTP 没有唯一结构。Meta 用共享 trunk 上的并行独立 heads；DeepSeek 用读取中间 token embedding 的顺序 modules；Qwen 又进一步通过多步训练提高递归 draft 时的训练—推理一致性。讨论“MTP 的公式”时必须先说明是哪一种计算图。

第三，训练与推理是两件事：

> 训练阶段，MTP 用多个未来目标塑造 backbone；推理阶段，MTP 只负责提出便宜候选，最终 token 仍由主模型通过 speculative decoding 验证和纠正。

所以，对 MTP 最准确的概括不是“一次生成多个 token”，而是：

> **训练时学会多看几步，推理时先猜几步、再一次验完。**

## 参考资料

1. [Better & Faster Large Language Models via Multi-token Prediction（ICML 2024）](https://proceedings.mlr.press/v235/gloeckle24a.html)
2. [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
3. [DeepSeek-V3 官方仓库](https://github.com/deepseek-ai/DeepSeek-V3)
4. [DeepSeek-V3 权重结构说明](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/README_WEIGHTS.md)
5. [Qwen3-Next：Towards Ultimate Training & Inference Efficiency](https://qwen.ai/blog?id=qwen3-next)
6. [Qwen3.8-27B 官方 Model Card](https://huggingface.co/Qwen/Qwen3.8-27B)
7. [ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training](https://aclanthology.org/2020.findings-emnlp.217/)
8. [Blockwise Parallel Decoding for Deep Autoregressive Models](https://arxiv.org/abs/1811.03115)
9. [Fast Inference from Transformers via Speculative Decoding](https://proceedings.mlr.press/v202/leviathan23a.html)
10. [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774)
11. [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077)
12. [EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840)
13. [Megatron Core：Multi-Token Prediction](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/multi_token_prediction.html)
14. [vLLM：Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode/)
