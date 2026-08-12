# Slime 轨迹管理：从 Message Tree 到可训练 Token Sequence

在普通的单轮 RL 训练中，一条轨迹通常很简单：

```text
prompt → response → reward
```

但 coding agent 的运行过程不是一次性问答，而是一个真实的多轮交互系统。模型会读取代码、调用工具、接收工具结果、继续推理，还可能启动 sub-agent 或触发上下文压缩。一次 coding-agent rollout 更接近：

```text
用户任务
  → assistant 调用工具
  → tool 返回结果
  → assistant 继续推理
  → assistant 修改代码
  → tool 运行测试
  → assistant 完成任务
  → 根据最终 patch 计算 reward
```

Slime 的难点不只是保存这些消息，而是把消息世界中的多轮 agent 执行，转换成 token 世界中严格可验证的 RL 训练数据。它采用了一套两层设计：

1. 用 message tree 维护 agent 的对话分支；
2. 用每轮 SGLang 返回的 token snapshot 构建最终训练序列。

本文基于 `examples/coding_agent_rl`、`slime.agent.adapters` 和 `slime.agent.trajectory` 的当前实现，分析这套机制解决了什么问题，以及它如何工作。

---

## 一、为什么不能直接保存 messages 后重新 tokenize

Coding agent 通过 Anthropic Messages 或 OpenAI Chat Completions 形式与 adapter 通信。每轮请求携带的是结构化消息：

```json
[
  {"role": "system", "content": "..."},
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "...", "tool_calls": [...]},
  {"role": "tool", "content": "..."}
]
```

但 RL 训练优化的不是字符串，而是模型实际采样的 token。对于 PPO、GRPO 等算法，训练 token 还必须与 rollout 时保存的旧策略 logprob 对齐。

因此，一个 token 可以参与训练，至少要满足：

- 它确实由 rollout model 采样；
- 能找到对应的 rollout logprob；
- 训练时使用的上下文与 rollout 时一致；
- 它在完整序列中的位置没有因重新 tokenize 而改变。

如果只保存 messages，训练前再套一遍 chat template、重新 tokenize，得到的 token 不一定与 rollout 时一致。原因包括：

- tokenizer 的 decode→encode 不保证 token 边界完全可逆；
- tool call 会在文本与结构化 JSON 之间转换；
- JSON 空格、字段顺序和转义形式可能变化；
- reasoning、text、tool-use block 可能被 parser 重新组织；
- agent CLI 可能规范化 assistant message；
- chat template 可能重写特殊标记；
- auto-compaction 会用摘要替换旧消息历史。

这就是所谓的 string-in、token-out 问题：环境以字符串和 message 工作，但训练必须保留 token provenance。

Slime 的原则是：

> 可读的 message 用于理解和路由轨迹；真正用于训练的序列必须直接来自 rollout 阶段保存的 token IDs。

---

## 二、Slime 的树到底存什么

Slime 的 tree 不是纯字符串树，也不是纯 token 树，而是一种混合结构。

### 1. MessageNode：负责树的拓扑

每个 session ID 对应一棵 message tree。节点 `MessageNode` 保存：

```python
role
message
parent
children
metadata
turn
turn_index
response_trained
```

其中 `message` 是完整的结构化 message dict。树的路径匹配使用：

```python
child.role == msg.get("role") and child.message == msg
```

也就是说，tree topology 是 message-based 的。完整 message dict 不相等，就可能产生新分支。

这套拓扑能够自然表达：

- 不同 user/system prompt；
- 同一 assistant 后出现不同 tool result；
- 主 agent 和 sub-agent 的不同消息历史；
- agent 自动压缩上下文后的新历史；
- assistant message 被客户端改写后的分歧。

### 2. TurnRecord：负责训练 token

如果某个 assistant 节点是本轮模型真实生成的，它还会挂载一个 `TurnRecord`：

```python
TurnRecord(
    prompt_ids=[...],
    output_ids=[...],
    output_log_probs=[...],
    finish_reason="...",
    ill_formed=False,
)
```

因此可以把整棵树理解为：

```text
Message Tree
├─ message dict：决定节点如何匹配、树如何分叉
└─ TurnRecord：记录 generated assistant 的真实 token snapshot
```

只有 `turn is not None` 的 assistant node 才具备可靠的 sampled-token provenance。

---

## 三、一轮请求如何进入树

Adapter 每处理一轮 agent 请求，会依次执行：

1. 将 Anthropic/OpenAI 请求翻译成统一消息；
2. 套用当前模型的 chat template，生成 `prompt_ids`；
3. 调用 SGLang `/generate`；
4. 要求 SGLang 返回 token logprob；
5. 保存实际的 `output_ids` 与 `output_log_probs`；
6. 解析模型输出，构造返回给 agent 的 assistant message；
7. 响应成功发送给 agent 后，调用 `record_turn()`。

这里的先后顺序很重要：Slime 先把响应 flush 给客户端，再记录这轮轨迹。如果客户端在生成期间断开，这轮响应并没有真正进入 agent 的后续状态，因此也不应被记入训练树。

`record_turn()` 的核心流程是：

```text
查找最深的 message 匹配点
  → 尝试处理 assistant rewrite
  → 挂载剩余 prompt messages
  → 添加本轮 generated assistant leaf
```

---

## 四、Message Tree 如何分叉

假设当前树是：

```text
root
└─ system
   └─ user
      └─ assistant: call tool
```

后续出现两个不同工具结果：

```text
root
└─ system
   └─ user
      └─ assistant: call tool
         ├─ tool result A
         │  └─ assistant A
         └─ tool result B
            └─ assistant B
```

这类分叉是真实的 agent 分支，应当保留。

Sub-agent 和 auto-compaction 也不需要专门的节点类型。只要它们送入 adapter 的消息历史与已有路径不同，message equality 就会让它们挂载到新的分支。

值得注意的是，message tree 只回答“这些 turn 属于哪条对话路径”，并不保证同一条路径上的 token 可以无损拼起来。后者要到导出阶段根据 token ID 再判断。

---

## 五、Assistant Rewrite：避免把格式变化误判成真实分支

Agent CLI 有时会把上一轮 assistant 消息稍微修改后再放进下一轮 prompt。例如模型实际生成：

```text
assistant: 调用 Read(foo.py)
```

下一轮客户端回放时变成：

```text
assistant: 调用 Read(foo.py)␠
```

也可能是 tool-call JSON 的空格、字段结构或 block 表示发生变化。由于 Slime 使用完整 message dict 相等来匹配路径，这两个 message 不相等。

如果按普通逻辑处理，树会变成：

```text
user
├─ assistant: 调用 Read(foo.py)       # 原始 generated leaf
└─ assistant: 调用 Read(foo.py)␠      # 新建 routing-only branch
   └─ tool result
      └─ assistant: 后续生成
```

但这往往不是真实分支，只是同一条消息被客户端重新渲染。原始 assistant 会变成一个孤立 leaf，额外导出一个训练 Sample。

为避免这种假分叉，Slime 对短 assistant response 提供 rewrite merge。满足以下条件时：

- 当前不匹配的消息是 assistant；
- 挂载点下恰好只有一个 assistant child；
- 该 child 是 leaf；
- 该 child 是模型真实生成的，即 `turn is not None`；
- 原 response token 数小于阈值；
- rewrite merge 没有被关闭；

Slime 会原地修改该节点：

```python
rewritten_node.message = new_message
rewritten_node.turn = None
rewritten_node.turn_index = None
```

结果是：

```text
user
└─ assistant: 调用 Read(foo.py)␠      # routing-only
   └─ tool result
      └─ assistant: 后续生成           # generated
```

这里存在一个明确取舍：

- 原始 assistant 有真实 `output_ids`，但它已不在 live path 中；
- 重写后的 assistant 位于 live path，却不是 SGLang 本轮采样的 token；
- 二者都不能作为严格可靠的 RL target。

因此，重写节点只作为上下文保留，最终 `loss_mask=0`；只有后续新的 generated assistant 参与训练。

默认阈值为 1024 token。旧 response 太长时，直接清掉会损失大量有效信号，因此 Slime 不做合并，而是保留为独立分支。阈值设为 0 可以完全关闭 rewrite merge。

这个限制主要来自 RL 的 token provenance 要求，而不是 SFT 的理论限制。纯 SFT 不依赖 rollout logprob，可以自行选择原始消息或重写消息作为 label；但如果直接复用 Slime 当前导出的 RL Sample，重写 assistant 已经被 mask，不会参与 SFT loss。

---

## 六、真正困难的部分：多轮 Token 如何拼接

假设有三轮生成：

```text
Turn 1: P1 → R1
Turn 2: P2 → R2
Turn 3: P3 → R3
```

其中 `P` 是这一轮完整 `prompt_ids`，`R` 是 SGLang 实际采样的 `output_ids`。

理想情况下：

```text
P2 = P1 + R1 + tool_context
P3 = P2 + R2 + tool_context
```

那么最终训练序列可以拼成：

```text
P1 | R1 | tool_context | R2 | tool_context | R3
     11 |      00      | 11 |      00      | 11
```

但下一轮的 prompt 是由消息重新渲染、重新 tokenize 得到的。它可能包含 `R1'` 而不是上一轮真实的 `R1`：

```text
上一轮真实采样：P1 + R1
下一轮实际输入：P1 + R1' + tool_context
```

即便 `R1` 和 `R1'` 解码后的字符串几乎相同，token IDs 也可能不同。

如果训练时强行使用 `R1` 作为 `R2` 的上下文，而 SGLang rollout 时看到的是 `R1'`，那么训练与采样对应的是两个不同 state。PPO ratio、GRPO loss 和 rollout logprob 的语义都会受到破坏。

所以 Slime 不根据字符串判断连续性，而是比较 token ID。

---

## 七、SampleBuilder 的三种处理：CLEAN、REALIGN、FORK

线性化一条 root-to-leaf 路径时，Slime 使用 `_SampleBuilder` 逐轮累计：

```python
tokens
loss_mask
logprobs
last_response_start_idx
leading_prompt_len
```

处理下一轮 `TurnRecord` 时，令：

```text
H = builder 当前累计的 tokens
P = 下一轮 prompt_ids
R = 下一轮 output_ids
L = H 与 P 的公共 token 前缀长度
```

然后根据 divergence 的位置选择三种策略。

### 1. CLEAN：严格前缀延续

如果：

```text
H == P[:len(H)]
```

说明现有序列是新 prompt 的严格前缀。Slime 只追加新 prompt 的尾部，再追加本轮 response：

```text
已有 H | prompt 新增部分 | R
        | loss_mask = 0  | loss_mask = 1
```

真实 rollout logprob 写入 `R` 对应位置；prompt/context 位置填 0。

这是正常工具交互最常见的路径。

### 2. REALIGN：最近 response 内出现漂移

假设：

```text
H: P1 | [a b c d]
P: P1 | [a b X Y] | tool context
```

divergence 发生在最近一轮 response 内。若当前新 response 较短，Slime 可以继续使用同一个 builder，但会从最近 response 的起点整体重建：

```text
P1 | a b X Y | tool context | R2
     0000000 |      000     | 11
```

也就是说：

- 删除 builder 中最近 response 之后的旧尾部；
- 从新 `prompt_ids` 中重新补齐该区域；
- 整段重建内容设置 `loss_mask=0`；
- 再将当前新生成的 `R2` 设置为 `loss_mask=1`。

即使只有最后一个 token 漂移，Slime 也不会继续训练前面仍相同的半截 response。因为一个 assistant turn 一旦发生重建，它就不再是完整、连续、来源单一的 sampled action。整轮降级是更保守的 provenance 策略。

这里有一个实现细节：REALIGN 的阈值判断使用的是当前新 turn 的 `output_ids` 长度，不是 drift token 数，也不是旧 response 长度。

### 3. FORK：开启新的 Sample

以下情况不能安全 REALIGN：

- divergence 落在初始 prompt；
- divergence 落在更早的 response，而不是最近 response；
- 当前 response 达到 fork threshold；
- threshold 为 0。

此时旧 builder 完成，当前 turn 用自己的完整 `prompt_ids + output_ids` 创建新 builder：

```text
Sample A:
P1 | R1 | context | R2
     11 |    00   | 11

Sample B:
P3 | R3
     11
```

在 Sample B 中，完整 `P3` 都是 prompt；只训练新生成的 `R3`。

因此，一条 message-tree leaf 不一定只产生一个 Sample。只要 token 连续性中途断裂，它就可能被拆成多个 segment。

---

## 八、Message 分叉与 Token 分叉是两个独立层次

这两个概念需要严格区分：

| 层次 | 判断依据 | 发生阶段 | 解决的问题 |
|---|---|---|---|
| Message tree 分叉 | role + 完整 message dict | `record_turn()` | turn 属于哪条对话路径 |
| Token segment 分叉 | `prompt_ids` 的 token 前缀关系 | `get_trajectory()` | token 能否作为连续 RL 序列 |

因此：

- 一条 message leaf 可能因 token drift 产生多个 Sample；
- 多条 message leaf 也可能共享相同的 generated assistant prefix；
- auto-compaction 通常先表现为 message branch，随后各 branch 再独立进行 token 拼接。

---

## 九、多 Leaf 共享前缀如何避免重复训练

考虑：

```text
assistant: 发起工具调用 R1
├─ tool A → assistant R2A
└─ tool B → assistant R2B
```

两条 leaf 都包含 `R1`。如果分别导出：

```text
Sample A: R1 mask=1, R2A mask=1
Sample B: R1 mask=1, R2B mask=1
```

那么共享 response `R1` 会被重复训练。

Slime 在 generated assistant node 上维护 `response_trained`：

- DFS 遍历到的第一条 leaf 认领该 response，设置 `loss_mask=1`；
- 后续 leaf 仍保留它作为必要上下文，但设置 `loss_mask=0`；
- 对应 rollout logprob 也填 0。

最终是：

```text
Sample A: R1 mask=1, R2A mask=1
Sample B: R1 mask=0, R2B mask=1
```

这样每个 generated assistant node 在整个 session 中最多训练一次。

具体由哪条 leaf 认领共享前缀，取决于 child 的插入和 DFS 顺序。在并发 sub-agent 场景中归属可能变化，但训练次数仍保持一次。

---

## 十、最终 Sample 的语义

`_SampleBuilder` 内部保存完整 token 序列：

```text
first prompt | first output | later context | later output | ...
```

例如：

```text
tokens:
P1 | R1 | tool1 | R2 | tool2 | R3

internal mask:
00 | 11 |  000  | 11 |  000  | 11
```

导出时第一轮 prompt 不放入 `Sample.loss_mask`，而通过：

```python
response_length = len(tokens) - leading_prompt_len
```

隐式表示。因此最终：

```text
Sample.tokens:
P1 | R1 | tool1 | R2 | tool2 | R3

Sample.response_length:
len(R1 | tool1 | R2 | tool2 | R3)

Sample.loss_mask:
11 |  000  | 11 |  000  | 11
```

这里的 `response_length` 不是“模型实际生成的 token 数”，而是第一轮 prompt 之后需要与训练 logits 对齐的整个尾部长度。

真正参与训练的 token 数是：

```python
sum(sample.loss_mask)
```

Sample 还会携带：

- `rollout_log_probs`：与 `loss_mask` 等长；
- `reward`：最终 patch 评测结果；
- `rollout_id`：同一 agent execution 导出的 sibling samples 共用；
- `metadata`：是否使用工具、是否 ill-formed、是否 solved、实例 ID 等；
- `response`：仅用于阅读的 decode sidecar。

`response` 不会被重新 tokenize 来恢复训练序列。

---

## 十一、Reward、Rollout ID 与训练聚合

当前实现中，同一个 trajectory 导出的每个 Sample 都获得完整 reward，而不是 `reward / K`：

```python
for sample in samples:
    sample.reward = reward
```

README 中“reward 在 K 条分支间平分”的描述与当前实现并不一致。

为了避免一个 rollout 因拆成多个 Sample 而被当作多次独立 rollout，所有 sibling 使用相同的 `rollout_id`。训练数据转换阶段会计算：

```text
rollout_mask_sum =
  同一 rollout 下所有 Sample 的 trainable token 总数
```

训练 loss 最终以整个 rollout 的 trainable token 总数作为分母，形成 per-rollout token-weighted mean。这样即使多个 segment 被动态 batch 分到不同 microbatch，也不会因为 segment 数量增加而简单放大一次 rollout 的 loss。

不过需要留意：默认 GRPO reward normalization 在 fan-out 后可能对扁平化的 Sample 集合计算 mean/std。分支多的 rollout 会重复出现更多次 reward，从而影响 normalization 权重。共享 `rollout_id` 能修正后续 loss reducer，但不能逆转已经完成的 reward normalization。这是当前 fan-out 语义中值得继续审视的边界。

---

## 十二、训练数据如何导出

`examples/coding_agent_rl/generate.py` 在 patch 评测完成后调用：

```python
adapter.finish_session(
    session_id,
    base_sample=base_sample,
    reward=reward,
)
```

`finish_session()` 会：

1. 等待并关闭该 session 的在途请求；
2. 遍历 message tree 的所有 leaf；
3. 将每条路径按 CLEAN/REALIGN/FORK 转为 Sample builders；
4. 过滤完全没有 trainable response 的 builder；
5. 构造 `list[Sample]`；
6. decode 可读的 `response` sidecar；
7. 清除 session tree。

### Rollout dump

示例 launcher 配置：

```bash
--save-debug-rollout-data \
  "${RUN_ROOT}/rollout_dumps/rollout_{rollout_id}.pt"
```

保存格式大致为：

```python
{
    "rollout_id": rollout_id,
    "samples": [sample.to_dict(), ...],
}
```

这是最适合离线检查轨迹的产物，包含 token IDs、loss mask、rollout logprob、reward、metadata 和 response sidecar。

### 送入训练器的数据

Sample flatten 后会转换为列式 train data：

```text
tokens
response_lengths
loss_masks
rollout_log_probs
rewards
raw_reward
sample_indices
rollout_ids
rollout_mask_sums
truncated
source_names
```

随后 Slime 按唯一 `rollout_id` 切 training step，再依据 token 长度做 DP 与 microbatch packing，并通过 Ray object store 传给 Megatron。

如果希望保存已经 tensorize、已经按 rank 分片的训练输入，可以额外配置：

```bash
--save-debug-train-data \
  "${RUN_ROOT}/train_data/rollout_{rollout_id}_rank_{rank}.pt"
```

---

## 十三、如何理解 Slime 的整体策略

Slime 的轨迹管理可以概括为四句话：

1. **Message 决定路径。**

   用结构化 message equality 建树，保留主 agent、sub-agent、工具结果和 compaction 产生的分支。

2. **Token 决定能否连续训练。**

   不相信字符串看起来一致，而是直接比较 SGLang 保存的 token IDs。

3. **Provenance 不确定就 mask 或 fork。**

   最近 response 的轻微漂移采用 REALIGN，将整轮降级为上下文；更早或更大的漂移直接拆成新 Sample。

4. **一次真实生成最多训练一次。**

   多 leaf 共享的 assistant response 通过 `response_trained` 去重，其他分支只把它作为 `loss_mask=0` 的上下文。

这套设计的核心偏好非常明确：

> 宁可牺牲一部分训练 token、增加一些 Sample，也不把来源无法证明的 token 当作 on-policy rollout action。

对于依赖旧策略 logprob 的 coding-agent RL，这是比“尽量拼成一条漂亮长序列”更重要的正确性约束。

---

## 代码索引

- `examples/coding_agent_rl/generate.py`：一次 SWE rollout 的编排与 session 导出
- `slime/agent/adapters/common.py`：消息翻译、SGLang 调用、TurnRecord 记录
- `slime/agent/trajectory.py`：MessageNode、TrajectoryManager、SampleBuilder
- `slime/ray/rollout.py`：Sample flatten、train data 转换和 rollout dump
- `slime/utils/dp_schedule.py`：按 rollout ID 切 step 和动态 batch
- `slime/backends/megatron_utils/cp_utils.py`：per-rollout loss reducer
- `tests/test_agent/test_trajectory_manager_branching.py`：message/token 分叉测试矩阵
