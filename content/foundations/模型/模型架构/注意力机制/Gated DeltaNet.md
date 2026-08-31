# 从 KV Cache 到“会擦写的白板”：Gated DeltaNet 入门

> 本文面向第一次接触线性注意力的读者，从标准注意力的 KV Cache 开始，依次解释固定状态、Delta Rule、遗忘门、分块并行，以及 Qwen3.8 为什么采用 Gated DeltaNet 与完整注意力混合的架构。阅读本文只需要知道向量点积和矩阵乘法。资料核对日期为 2026 年 8 月 24 日；通用机制以 ICLR 2025 论文和官方实现为准，Qwen3.8 配置以官方 Model Card、`config.json` 与 Transformers 实现为准。

假设模型先读到：

> 张三的办公室是 101。

过了一会儿，它又读到：

> 张三搬到了 307。

一个合格的记忆系统不应该把 101 和 307 简单相加，而应该用 307 覆盖 101。如果接下来文章主题完全改变，它还应该逐渐淡化上一段文本中的无关信息。

这对应了序列模型的三个基本问题：

1. 如何记住历史？
2. 如何修改已经写入的记忆？
3. 如何忘掉不再需要的内容？

标准注意力、普通线性注意力、DeltaNet 和 Gated DeltaNet，正是在用不同方式回答这三个问题。

先给出全文最重要的直觉：

> 标准注意力像保存每张原始便签的档案柜；线性注意力像一块固定大小的白板；Delta Rule 让白板能够先读旧答案、再按误差修改；Gate 则给白板增加了一个整体淡化旋钮。

---

## 1. 标准注意力：保存所有原始便签

Transformer 的注意力机制会为每个 token 生成三个向量：

- Query：我现在想找什么？
- Key：这条历史信息的索引是什么？
- Value：这条历史信息的内容是什么？

对于第 $t$ 个 token，标准因果注意力可以写成：

$$
o_t=\sum_{i\le t}a_{ti}v_i
$$

其中：

$$
a_{ti}
=
\operatorname{softmax}_i
\left(
\frac{q_t^\top k_i}{\sqrt d}
\right)
$$

当前查询 $q_t$ 会和所有历史 key 比较，再根据相似度对对应的 value 加权求和。

可以把它想象成一个档案柜：

- 每个历史 token 都有一张原始便签；
- key 是便签标签；
- value 是便签内容；
- query 会在所有历史便签中查找相关信息。

这种方式的优势是保留了每条历史记录，可以直接回看某个具体 token。代价也很明显。

### 1.1 训练与 Prefill

长度为 $L$ 的序列会形成一个 $L\times L$ 的注意力关系，核心计算量大约为：

$$
O(L^2d)
$$

FlashAttention 可以显著减少中间结果的显存读写，但不会改变完整注意力关于序列长度的二次计算复杂度。

### 1.2 自回归解码

生成新 token 时，为避免重复计算，模型需要保存所有历史 token 的 K 和 V，也就是 KV Cache。因此：

- KV Cache 随 $L$ 线性增长；
- 每生成一个 token，都要读取长度为 $L$ 的历史；
- 单步注意力计算量相对于历史长度是 $O(L)$。

上下文越长，档案柜就越大，每次查找需要翻阅的内容也越多。

## 2. 线性注意力：把历史压进固定大小的状态

线性注意力选择了另一条路线：

> 不保存每张便签，而是把历史不断压缩进一个固定大小的状态矩阵。

为了理解这个过程，先考虑一种可以分解的相似度。省略归一化项后：

$$
o_t
=
\sum_{i\le t}
(q_t^\top k_i)v_i
$$

利用矩阵乘法的结合律，可以改写成：

$$
o_t
=
\left(
\sum_{i\le t}v_i k_i^\top
\right)q_t
$$

本文统一采用下面这组转置后的记号：

$$
q_t,k_t\in\mathbb R^{d_k},
\qquad
v_t,o_t\in\mathbb R^{d_v},
\qquad
S_t\in\mathbb R^{d_k\times d_v}
$$

定义状态矩阵：

$$
S_t=\sum_{i\le t}k_i v_i^\top
$$

那么状态更新和读取就是：

$$
S_t=S_{t-1}+k_tv_t^\top
$$

$$
o_t=S_t^\top q_t
$$

$S_t$ 可以理解为一个从 key 空间到 value 空间的联想记忆：

$$
k\longrightarrow v
$$

写入一条关联使用外积 $k_tv_t^\top$，读取则使用矩阵乘法 $S_t^\top q_t$。这块“白板”的大小只取决于 $d_k$ 和 $d_v$，不取决于已经读了多少 token。

因此：

- 状态大小是 $O(d_kd_v)$；
- 单步更新与读取成本是 $O(d_kd_v)$；
- 这些成本相对于上下文长度 $L$ 都是 $O(1)$；
- 整段序列在维度固定时，对 $L$ 呈线性增长。

这里的“线性”指复杂度随序列长度近似线性，并不是说整个模块是线性函数。真实层仍然包含非线性激活、门控、归一化和卷积。

还需要强调一点：Softmax 注意力不能直接通过交换乘法顺序变成线性注意力。通常必须先换成可分解的相似度，或者像 Gated DeltaNet 一样直接定义递归状态更新。Gated DeltaNet 属于后者，而不只是一个 Softmax 近似技巧。

## 3. 固定状态的代价：信息会叠加和碰撞

普通线性注意力的写入方式是：

$$
S_t=S_{t-1}+k_tv_t^\top
$$

问题在于，它只会增加新内容，不知道怎样覆盖旧内容。

假设同一个 key 先后对应两个 value：

$$
k\rightarrow v_{\text{old}}
$$

$$
k\rightarrow v_{\text{new}}
$$

两次普通加法写入后：

$$
S
=
kv_{\text{old}}^\top
+
kv_{\text{new}}^\top
$$

再次用 $k$ 查询时，得到的会近似于：

$$
v_{\text{old}}+v_{\text{new}}
$$

模型不是擦掉“101”再写“307”，而是在同一位置继续叠墨。

此外，真实模型中的 key 通常不是彼此正交的。如果两个 key 很相似，它们写入状态的方向也会重叠，查询其中一个时就可能混入另一个的 value。

因此，固定状态不是一座无限容量、无损的数据库。它是用压缩换取效率，容量有限，信息碰撞不可避免。这正是 Delta Rule 要解决的问题。

## 4. Delta Rule：不要写入答案，而要写入误差

Delta Rule 的核心思想是：

> 写入新内容之前，先看看状态现在认为答案是什么，然后只写入预测误差。

对于当前 key $k_t$，先从旧状态中读取：

$$
\widehat v_t=S_{t-1}^\top k_t
$$

$\widehat v_t$ 是状态对当前 key 的旧预测。然后计算误差：

$$
e_t=v_t-\widehat v_t
$$

最后把误差沿当前 key 的方向写回：

$$
S_t
=
S_{t-1}
+
\beta_t k_t e_t^\top
$$

展开就是：

$$
\boxed{
S_t
=
S_{t-1}
+
\beta_t k_t
\left(
v_t-S_{t-1}^\top k_t
\right)^\top
}
$$

其中 $\beta_t\in(0,1)$ 控制本次修改的强度。这就是 DeltaNet 中的 Delta：它指新 value 与旧预测之间的差值。

### 4.1 为什么它能够覆盖旧关联

假设 $k_t$ 已经经过 L2 归一化，即：

$$
\lVert k_t\rVert_2=1
$$

更新后再次查询相同的 key，可以得到：

$$
S_t^\top k_t
=
(1-\beta_t)\widehat v_t+\beta_t v_t
$$

因此：

- $\beta_t=0$：完全不更新；
- $\beta_t=0.5$：旧答案和新答案各占一半；
- $\beta_t=1$：当前 key 对应的旧答案被替换为新答案。

普通线性注意力写入的是“新答案”，DeltaNet 写入的是“新答案与旧预测之间的差”。

### 4.2 在线梯度下降视角

把 $S$ 看成一个小型线性模型，其目标是让：

$$
S^\top k_t\approx v_t
$$

定义平方误差：

$$
\mathcal L_t(S)
=
\frac12
\left\lVert
S^\top k_t-v_t
\right\rVert^2
$$

对 $S$ 做一步梯度下降：

$$
S_t
=
S_{t-1}
-
\beta_t\nabla_S\mathcal L_t(S_{t-1})
$$

正好得到 Delta Rule。因此 DeltaNet 也可以被理解为一种前向过程中的在线学习：

- $S$ 是快速变化的权重；
- 每读入一个 token，就产生一个训练样本 $(k_t,v_t)$；
- 模型在前向传播过程中，对内部状态执行一步梯度下降；
- $\beta_t$ 相当于数据依赖的学习率。

这也是它被归入 fast-weight memory 一类方法的原因。

但 Delta Rule 不是哈希表式的精确覆盖。如果另一个 key 与当前 $k_t$ 不正交，它仍可能受到这次更新影响。Delta Rule 改善了定向修改和关联覆盖，但没有消除固定容量带来的所有碰撞。

## 5. DeltaNet 还缺少什么：主动遗忘

Delta Rule 擅长修改当前 key 所在的方向。但如果文本主题发生整体切换，例如从一篇法律文档突然切换到一段 Python 代码，旧状态中的大量关联可能都已失去价值。

纯 DeltaNet 只有在新 key 与旧 key 重叠时，才会修改相应方向。它缺少一个快速淡化整块旧状态的机制。

一种简单的解决方法是加入衰减：

$$
S_t=\alpha_tS_{t-1}+k_tv_t^\top
$$

其中：

$$
0<\alpha_t<1
$$

- $\alpha_t\approx1$：保留大部分旧状态；
- $\alpha_t\approx0$：快速清空旧状态。

如果说 Delta Rule 是一块局部橡皮擦，那么 $\alpha_t$ 就是整块白板的透明度旋钮。

但只有衰减也不够。它能淡化旧状态，却仍然采用简单的加法写入，不能准确覆盖同一个 key 的旧答案。

于是，一个自然的问题出现了：能否同时使用整体遗忘和定向改写？这就是 Gated DeltaNet。

## 6. Gated DeltaNet：先遗忘，再按误差改写

Gated DeltaNet 的核心更新最好拆成两步理解。

第一步，整体衰减：

$$
\widetilde S_t
=
\alpha_tS_{t-1}
$$

第二步，使用衰减后的状态预测并写入误差：

$$
\widehat v_t
=
\widetilde S_t^\top k_t
$$

$$
S_t
=
\widetilde S_t
+
\beta_t k_t
\left(v_t-\widehat v_t\right)^\top
$$

合在一起：

$$
\boxed{
S_t
=
\alpha_tS_{t-1}
+
\beta_t k_t
\left(
v_t-\alpha_tS_{t-1}^\top k_t
\right)^\top
}
$$

展开后得到：

$$
\boxed{
S_t
=
\alpha_t
\left(
I-\beta_tk_tk_t^\top
\right)S_{t-1}
+
\beta_tk_tv_t^\top
}
$$

这与原论文中的公式互为转置，只是矩阵方向约定不同。两个门的分工是：

- $\alpha_t$：旧状态保留率，负责整体遗忘；
- $\beta_t$：误差写入强度，负责定向更新。

一些特殊情况也很直观：

- $\alpha_t=1$：退化为纯 DeltaNet；
- $\beta_t=0$：只衰减，不写入当前关联；
- $\alpha_t\rightarrow0$：旧状态几乎清空，但当前 token 的新关联仍能写入；
- $\alpha_t=1,\beta_t=1$：在归一化 key 的方向上进行完整覆盖。

### 6.1 一个容易写错的公式

下面这个公式经常出现，但它与论文更新并不等价：

$$
\alpha_tS_{t-1}
+
\beta_tk_t
\left(
v_t-S_{t-1}^\top k_t
\right)^\top
$$

它衰减了状态，却使用未衰减的状态计算误差。正确顺序是：

1. 先得到 $\widetilde S_t=\alpha_tS_{t-1}$；
2. 再用 $\widetilde S_t$ 预测；
3. 最后写入预测误差。

## 7. 一个二维数值例子

设 key 空间只有两个正交方向：

$$
k_A=
\begin{bmatrix}
1\\
0
\end{bmatrix},
\qquad
k_B=
\begin{bmatrix}
0\\
1
\end{bmatrix}
$$

value 是一个标量。把当前状态简写成：

$$
S=[20,5]
$$

它表示：

$$
A\rightarrow20,
\qquad
B\rightarrow5
$$

现在收到一条新信息：

$$
A\rightarrow7
$$

比较几种更新方法：

| 更新方式 | 更新后的状态 | 查询 A | 查询 B |
| --- | ---: | ---: | ---: |
| 普通加法 | $[27,5]$ | 27 | 5 |
| Delta，$\beta=1$ | $[7,5]$ | 7 | 5 |
| 衰减再加法，$\alpha=0.1$ | $[9,0.5]$ | 9 | 0.5 |
| Gated Delta，$\alpha=0.1,\beta=1$ | $[7,0.5]$ | 7 | 0.5 |

最后一行的计算过程是：

先衰减：

$$
\widetilde S=0.1[20,5]=[2,0.5]
$$

当前对 A 的预测是 2，因此误差为：

$$
7-2=5
$$

将误差写回 A 的方向：

$$
S=[2,0.5]+[5,0]=[7,0.5]
$$

这正是我们想要的效果：

- A 被准确更新为 7；
- 不相关的旧关联 B 被衰减；
- 新值不会和旧值简单叠加。

## 8. 最小实现：五步看懂核心机制

下面的 PyTorch 风格代码只用于解释递推语义，不适合作为高性能训练实现：

```python
def gated_delta_step(q, k, v, log_alpha, beta, state):
    """
    q, k:  [..., d_key]
    v:     [..., d_value]
    state: [..., d_key, d_value]
    """

    q = l2_normalize(q) / sqrt(q.shape[-1])
    k = l2_normalize(k)

    # 1. 全局遗忘
    alpha = exp(log_alpha)              # 0 < alpha <= 1
    state = alpha[..., None, None] * state

    # 2. 当前 key 在记忆中已经对应什么 value
    predicted_v = einsum("...kv,...k->...v", state, k)

    # 3. 计算并缩放预测误差
    delta = beta[..., None] * (v - predicted_v)

    # 4. 沿当前 key 的方向写回误差
    state = state + einsum("...k,...v->...kv", k, delta)

    # 5. 使用 query 读取更新后的状态
    output = einsum("...kv,...k->...v", state, q)

    return output, state
```

如果去掉工程细节，Gated DeltaNet 的核心可以浓缩成四句话：

```text
旧状态先衰减
用 key 读取旧答案
把新旧答案的误差写回
用 query 读取更新后的状态
```

## 9. 真实 Gated DeltaNet 层还有哪些组件

完整的 Gated DeltaNet 层不只有递推公式。典型处理流程是：

```text
输入 x
  │
  ├─ 线性投影得到 Q、K、V
  │    └─ depthwise causal Conv1D → SiLU
  │
  ├─ 线性投影得到 α、β
  │
  ├─ Q/K 做 L2 normalization
  │
  ├─ Gated Delta Rule
  │
  ├─ RMSNorm + 输出门 z
  │
  └─ 输出投影
```

原论文的基础模块具有以下设计：

- Q、K、V 由线性投影生成；
- Q、K、V 会经过短程 causal convolution 和 SiLU；
- Q、K 做 L2 归一化，以提高训练稳定性；
- $\alpha$、$\beta$ 由输入投影产生；
- 读取结果还会经过归一化和输出门控。

### 9.1 为什么需要短程卷积

递归状态擅长保存和修改压缩后的历史，但模型还需要识别局部模式，例如：

- 相邻字符组成的单词；
- 局部语法结构；
- 连续代码符号；
- 最近几个 token 的顺序变化。

短程因果卷积为 Q、K、V 注入局部上下文，同时不会看到未来 token。

### 9.2 为什么要归一化 Q 和 K

Delta Rule 的覆盖性质依赖 key 的尺度。如果 $\lVert k_t\rVert_2=1$，$\beta_t$ 就可以被清楚地解释为当前关联的更新比例。未经控制的 key 范数可能让状态更新过强或过弱。

### 9.3 不要混淆三种 Gate

除了记忆衰减门 $\alpha$ 和写入门 $\beta$，真实实现通常还会生成输出门 $z$：

$$
o_t
=
\operatorname{SiLU}(z_t)
\odot
\operatorname{Norm}(S_t^\top q_t)
$$

三者作用不同：

- $\alpha$ 控制旧记忆；
- $\beta$ 控制本次写入；
- $z$ 控制层输出，不参与状态更新。

它们也不同于完整注意力层中的 output gating。

## 10. 推理是递归的，训练为什么还能并行

看到：

$$
S_t=f(S_{t-1},x_t)
$$

很容易产生一个疑问：既然当前状态依赖上一个状态，训练时岂不是必须逐 token 串行执行？

从数学语义上看，它确实是递归的。但训练时可以采用等价的分块并行形式。

### 10.1 解码：逐 token 递推

生成阶段每次只来一个新 token，直接使用递推最自然：

```text
读取旧状态 → 更新状态 → 输出结果
```

这时只需要保存：

- 固定大小的矩阵状态；
- 短程卷积最近几个位置的状态。

每一步的成本不随历史长度增加。

### 10.2 训练与 Prefill：Chunkwise Parallel

训练时会把序列划分成若干块，例如每块 64 个 token：

```text
块 1 → 状态 → 块 2 → 状态 → 块 3
```

块与块之间传递状态，块内部则将大量计算重新组织成矩阵乘法。

Delta 更新中的状态转移包含：

$$
I-\beta_tk_tk_t^\top
$$

它属于单位矩阵加低秩修正。DeltaNet 论文利用广义 Householder 矩阵乘积的紧凑 WY/UT 表示，将多步递推转换为 GPU 更擅长的大规模矩阵乘法；Gated DeltaNet 在此基础上继续处理 $\alpha_t$ 带来的衰减。

因此：

> 递归形式解释模型记住了什么，分块并行形式解决 GPU 怎么高效计算。

两种形式的数学结果相同，只是执行顺序不同。

## 11. Gated DeltaNet 到底节省了什么

下面只比较注意力或 token-mixing 核心，不包含 MLP、输入输出投影等共同成本。

| 对比项 | 完整注意力 | Gated DeltaNet |
| --- | --- | --- |
| 历史表示 | 每个 token 的 K/V | 固定状态矩阵 |
| 状态大小 | $O(Ld)$ | $O(d_kd_v)$ |
| Prefill 序列复杂度 | $O(L^2d)$ | $O(Ld_kd_v)$ |
| 单步解码对 $L$ 的依赖 | $O(L)$ | $O(1)$ |
| 精确回看历史 | 强 | 受压缩容量限制 |
| GPU 训练 | 天然矩阵并行 | 需要专用分块算法 |

这里的 $O(1)$ 是指“不随上下文长度增长”，并不代表一次状态更新没有计算成本。每个 Gated DeltaNet head 仍需维护和更新一个 $d_k\times d_v$ 的矩阵。

此外，“Gated DeltaNet 层具有固定状态”也不等于“采用混合架构的整个模型没有 KV Cache”。如果模型仍包含完整注意力层，这些层的 KV Cache 依然随上下文长度增长。

## 12. 为什么还要混入完整注意力

固定状态是一种有损压缩。Gated DeltaNet 擅长：

- 流式处理长序列；
- 持续更新工作记忆；
- 修改已经写入的关联；
- 用固定大小的状态概括历史；
- 降低长上下文解码成本。

但它不保存可逐 token 回看的原始历史，因此在下面这些任务上可能吃亏：

- 精确复制很久以前的一段文本；
- 在大量相似 key 中定位某一个；
- 对远距离 token 做精确比较；
- 恢复原始上下文中的细粒度细节。

可以继续使用“白板与档案柜”的类比：

- Gated DeltaNet 是工作白板：便宜、固定大小、随时修改；
- 完整注意力是原始档案柜：昂贵，但可以重新查阅具体记录。

Qwen 团队在 Qwen3-Next 中采用 3:1 的混合比例：75% 的层使用 Gated DeltaNet，25% 的层保留完整注意力。官方报告称，该组合比只使用其中一种结构取得了更好的效果与效率平衡。

## 13. Qwen3.8 中的 Gated DeltaNet

Qwen3.8-27B 延续了这套 3:1 布局，共有 64 个语言模型层：

$$
16\times
\left[
3\times(\text{Gated DeltaNet}\rightarrow\text{FFN})
+
1\times(\text{Gated Attention}\rightarrow\text{FFN})
\right]
$$

也就是：

- 48 层 Gated DeltaNet；
- 16 层完整 Gated Attention；
- 每连续 3 层 GDN 后插入 1 层完整注意力。

其 GDN 配置为：

- 16 个 Q/K heads；
- 48 个 V heads；
- 每个 Q/K head 服务 3 个 V heads；
- key/value head dimension 均为 128；
- depthwise causal Conv1D 的 kernel size 为 4。

Qwen 实现中的门控参数为：

$$
\beta_t=\sigma(b_t)
$$

$$
g_t
=
-\exp(A_{\log})
\operatorname{softplus}(a_t+dt_{\text{bias}})
$$

$$
\alpha_t=\exp(g_t)
$$

因为 $g_t\le0$，所以：

$$
0<\alpha_t\le1
$$

这保证了 $\alpha_t$ 表示衰减，而不是不受控制地放大旧状态。实现还会：

- 让 Q、K、V 先经过 kernel size 为 4 的 depthwise causal convolution 和 SiLU；
- 对 Q、K 做 L2 normalization；
- 将 16 个 Q/K heads 各复制 3 次，对齐 48 个 V heads；
- 用独立的 $z$ 投影控制 GDN 输出；
- 在长序列路径中使用 chunkwise kernel，在单 token decode 中使用 recurrent kernel。

Qwen3.8 的 GDN 层本身不使用 RoPE，顺序信息主要来自因果递推和短卷积；周期性的完整注意力层则使用位置编码。

需要特别注意：Qwen3.8 整个模型的缓存并不是 $O(1)$。48 个 GDN 层使用固定状态，但剩余 16 个完整注意力层仍然需要随上下文增长的 KV Cache。混合架构减少了昂贵层的数量，并没有让整网变成严格的纯线性模型。

更多模型级结构与显存分析可继续阅读 [[Qwen3.8-27B 模型架构]]。

## 14. 常见误区

1. **“线性注意力就是交换 Softmax 乘法顺序。”** 不是。Softmax 权重依赖当前 query，不能直接重新加括号。必须先改变相似度形式，或直接定义递归状态更新。
2. **“固定状态可以无损保存无限历史。”** 不能。它是压缩记忆，容量有限，相似 key 会产生串扰。
3. **“Delta Rule 消除了所有碰撞。”** 它改善了覆盖与定向更新，但非正交 key 仍会互相影响。
4. **“$\alpha$ 可以精确删除某个 key。”** 原始 Gated DeltaNet 中，$\alpha$ 是每个 token、每个 head 的标量，会缩放该 head 的整个状态。定向修改主要来自 Delta 项。
5. **“$\alpha$、$\beta$ 和输出门 $z$ 是同一个 Gate。”** 三者分别控制衰减、误差写入和层输出，作用位置不同。
6. **“递归模型只能串行训练。”** 递归是语义形式；训练与 Prefill 可以使用等价的分块并行算法。
7. **“线性注意力意味着整个网络都是线性的。”** “线性”主要描述序列长度复杂度，真实层仍包含卷积、SiLU、归一化和门控。
8. **“使用 GDN 后完全不需要 KV Cache。”** 纯 GDN 层不需要随长度增长的 K/V，但混合模型中的完整注意力层仍然需要。
9. **“$\beta=1$ 总能无条件精确覆盖。”** 精确覆盖结论依赖 key 归一化，而且与当前 key 不正交的其他方向仍可能受到影响。
10. **“Gated DeltaNet 与 Qwen 的 Gated Attention 是同一机制。”** 前者是矩阵状态的递归更新，后者仍是完整注意力，只是在输出路径上增加了门控。

## 15. 总结

理解 Gated DeltaNet，可以抓住三条公式。

普通线性注意力只会追加：

$$
S_t=S_{t-1}+k_tv_t^\top
$$

DeltaNet 先预测，再写入误差：

$$
S_t
=
S_{t-1}
+
\beta_tk_t
\left(
v_t-S_{t-1}^\top k_t
\right)^\top
$$

Gated DeltaNet 先衰减，再写入误差：

$$
\boxed{
S_t
=
\alpha_tS_{t-1}
+
\beta_tk_t
\left(
v_t-\alpha_tS_{t-1}^\top k_t
\right)^\top
}
$$

如果只记住一句话，可以记住：

> 标准注意力保存历史，线性注意力压缩历史，Delta Rule 改写历史，Gate 决定什么时候忘记历史。

## 参考资料

1. [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464)
2. [Gated DeltaNet：ICLR 2025 论文 PDF](https://jankautz.com/publications/GatedDeltaNet_ICLR25.pdf)
3. [Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://arxiv.org/abs/2406.06484)
4. [Gated DeltaNet 官方 PyTorch 实现](https://github.com/NVlabs/GatedDeltaNet)
5. [Qwen3-Next：Towards Ultimate Training & Inference Efficiency](https://qwen.ai/blog?id=qwen3-next)
6. [Qwen3-Next-80B-A3B-Instruct 官方 Model Card](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)
7. [Qwen3.8-27B 官方 Model Card](https://huggingface.co/Qwen/Qwen3.8-27B)
8. [Qwen3.8-27B `config.json`](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json)
9. [Transformers：Qwen3-Next Gated DeltaNet 实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/modeling_qwen3_next.py)
10. [Transformers：Qwen3.5 / Qwen3.8 模型实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modular_qwen3_5.py)
