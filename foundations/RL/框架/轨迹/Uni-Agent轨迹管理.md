# Uni-Agent 轨迹管理：从 Message List 到在线构造的 Token Trajectory

在单轮 RL 训练中，一条轨迹通常可以简单表示为：

```text
prompt → response → reward
```

但 Claude Code、coding agent 和 tool agent 的真实运行过程是多轮、带状态且可能分叉的：

```text
用户任务
  → assistant 生成工具调用
  → tool 返回观察
  → assistant 继续生成
  → 上下文压缩或启动 sub-agent
  → 更多工具调用
  → 完成任务
  → 根据最终结果计算 reward
```

在 message 世界里，这是一组持续增长或发生分叉的结构化消息；在 RL 训练世界里，它必须变成一条满足以下条件的 token 序列：

- 模型生成 token 与 rollout logprob 严格对齐；
- 工具结果和用户输入作为上下文保留，但不参与 policy loss；
- 多轮 assistant 输出都能参与训练；
- 分叉轨迹之间不会相互污染；
- 训练时使用的上下文与 rollout 时模型真正看到的上下文一致。

Uni-Agent 的核心选择是：

> 不在 rollout 结束后从完整 message history 重建轨迹，而是在每次模型调用期间在线维护 token-level trajectory；session finalize 时直接从 chain buffer 取出训练 token。

这个设计避免了对模型输出进行 decode→encode，但也把 chat template 的正确性压力放到了增量编码路径上。本文基于 Uni-Agent 当前的 Gateway、Session 和 Agent Framework 实现，分析它如何从 message list 构造训练序列、如何处理 message 分叉，以及这种在线轨迹设计存在的关键一致性风险。

---

## 一、端到端数据流

Uni-Agent 的 RL 数据流可以概括为：

```text
Parquet 中的 prompt/message list
        ↓
verl DataLoader：保留 raw chat
        ↓
Agent 运行并维护 message history
        ↓
OpenAI/Anthropic 请求携带完整 message list
        ↓
Uni-Agent Gateway 规范化 messages
        ↓
GatewaySession 在线维护 ChainState/TrajectoryBuffer
        ↓
prompt_ids + response_ids + response_mask + logprobs
        ↓
finalize：直接 materialize 为 Trajectory
        ↓
input_ids / attention_mask / position_ids / loss_mask
        ↓
TransferQueue
        ↓
PPO/GRPO trainer
```

### 1. 数据集中的 prompt

以 SWE-Bench 为例，预处理后的 `prompt` 是 OpenAI chat 格式：

```python
prompt = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {
        "role": "user",
        "content": USER_PROMPT.format(
            problem_statement=example["problem_statement"]
        ),
    },
]
```

对应代码位于：

```text
uni_agent/tasks/swe_bench/preprocess.py
```

训练 recipe 使用：

```bash
data.prompt_key=prompt
data.return_raw_chat=True
```

这让 verl DataLoader 保留原始 message list。Agent Framework 从输入 batch 中读取：

```python
raw_prompt = sample_fields["raw_prompt"]
```

不过在当前内置 `run_task` 路径里，任务实际使用的是 `tools_kwargs["task"]["prompt"]`；`raw_prompt` 主要用于遵循 verl runner 协议和保留样本元数据。任务最终执行：

```python
messages = cfg.prompt
agent_result = await agent.run(
    sandbox=sandbox,
    messages=messages,
)
```

### 2. Agent 每轮发送完整 message list

Agent 会维护类似下面的消息历史：

```json
[
  {"role": "system", "content": "..."},
  {"role": "user", "content": "..."},
  {
    "role": "assistant",
    "content": "",
    "tool_calls": [
      {
        "id": "call_123",
        "type": "function",
        "function": {
          "name": "read_file",
          "arguments": {"path": "foo.py"}
        }
      }
    ]
  },
  {
    "role": "tool",
    "tool_call_id": "call_123",
    "content": "..."
  }
]
```

Claude Code 使用 Anthropic Messages API，其他 agent 也可能使用 OpenAI Chat Completions API。两种请求都会在 Gateway adapter 中被转换成统一的内部格式：

```python
InternalGenerationRequest(
    messages=normalized_messages,
    tools=normalized_tools,
    sampling_params=sampling_params,
)
```

值得注意的是：客户端每一轮都发送完整 message list，但 GatewaySession 并不一定对完整 message list 重新执行 chat template。

---

## 二、Uni-Agent 的 token truth 存在哪里

Uni-Agent 在一个 `GatewaySession` 中维护多条 `ChainState`。每条 chain 是一条线性的对话与 token 轨迹：

```python
ChainState(
    chain_id,
    message_history,
    message_tip_hash,
    active_tool_schemas,
    buffer,
    image_data,
    video_data,
    last_assistant_start,
    updated_seq,
)
```

真正被视为训练 token truth 的是 `TrajectoryBuffer`：

```python
TrajectoryBuffer(
    prompt_ids,
    response_ids,
    response_mask,
    response_logprobs,
    routed_experts,
)
```

各字段的语义是：

| 字段 | 含义 |
|---|---|
| `prompt_ids` | 第一次模型调用时完整模板化得到的 prompt token |
| `response_ids` | 历次模型生成 token，加上轮次之间插入的上下文 token |
| `response_mask` | `1` 表示模型实际生成，`0` 表示工具结果、用户消息或模板连接 token |
| `response_logprobs` | 与 `response_ids` 对齐；模型生成位置是真实 logprob，上下文位置填 `0.0` |
| `routed_experts` | 可选的 MoE routing 信息 |

这里有一个容易误解的点：

> `response_ids` 并不是纯 assistant response，而是首轮 prompt 之后的整个时序 token 流。

例如：

```text
prompt_ids:
  system + initial user + assistant-start

response_ids:
  assistant-generation-1       mask=1
  tool-result/chat-glue        mask=0
  assistant-generation-2       mask=1
  user/tool continuation       mask=0
  assistant-generation-3       mask=1
```

训练时到底优化哪些位置，不由 `response_ids` 这个名字决定，而由 `response_mask/loss_mask` 决定。

---

## 三、第一次模型调用：完整模板化

当 Session 没有找到可复用的 chain 时，它会对完整 message list 编码：

```python
prompt_ids = codec.encode_full(
    messages,
    tools=tools,
    image_data=image_data,
    video_data=video_data,
)

buffer = TrajectoryBuffer(prompt_ids=prompt_ids)
```

文本模型的逻辑相当于：

```python
prompt_ids = apply_chat_template(
    tokenizer,
    messages,
    tools=tools,
    add_generation_prompt=True,
)
```

其中 `add_generation_prompt=True` 会在序列尾部添加 assistant 开始生成所需的模板标记。

多模态模型则先渲染字符串模板，再交给 processor：

```python
raw_prompt = apply_chat_template(
    processor,
    messages,
    tools=tools,
    add_generation_prompt=True,
    tokenize=False,
)

model_inputs = processor(
    text=[raw_prompt],
    images=image_data,
    videos=video_data,
    return_tensors="pt",
)

prompt_ids = model_inputs["input_ids"]
```

随后 Gateway 将：

```python
context_ids = prompt_ids + response_ids
```

发给 vLLM/SGLang rollout backend。

---

## 四、模型输出如何进入训练轨迹

Rollout backend 返回的是实际采样 token，而不是仅返回解码后的字符串：

```python
output.token_ids
output.log_probs
output.stop_reason
```

Session 将这些 token 原样追加：

```python
response_ids = list(output.token_ids)
buffer.response_ids.extend(response_ids)
buffer.response_mask.extend([1] * len(response_ids))
buffer.response_logprobs.extend(output.log_probs)
```

因此模型输出部分保留了严格的 sampled-token provenance：

```text
模型实际采样 token IDs
        ↕ 一一对齐
rollout policy logprobs
        ↕ 一一对齐
response_mask = 1
```

这一步避免了下面的危险路径：

```text
模型 token
  → decode 成字符串
  → parser/客户端重组 message
  → 再次 tokenize
```

decode→encode 不保证恢复原始 token 边界，tool-call JSON、reasoning block 和特殊 token 都可能在重编码过程中发生变化。Uni-Agent 直接保存 backend token IDs，是其在线轨迹设计最重要的优点。

---

## 五、后续模型调用：只模板化增量 messages

假设第一轮完成后，chain 中保存：

```text
message_history:
  [system S, user U, assistant A]

token buffer:
  prompt_ids=P0
  response_ids=G0
```

下一轮 Claude Code 会发送完整历史：

```text
[system S, user U, assistant A, tool T]
```

Session 首先匹配已有 chain，然后计算：

```python
incremental_start = len(selected_chain.message_history)
incremental_messages = messages[incremental_start:]
```

本例中：

```text
incremental_messages = [tool T]
```

随后只对增量 messages 调用：

```python
incremental_ids = codec.encode_incremental(incremental_messages)
```

`encode_incremental()` 不是简单的 `tokenizer.encode(tool_text)`。它仍然调用 chat template：

```python
ids = apply_chat_template(
    tokenizer,
    incremental_messages,
    add_generation_prompt=True,
)

return ids[len(system_prompt_ids):]
```

也就是说，其算法是：

```text
增量 messages
  → 单独执行 chat template
  → 得到带固定 system prefix 的 token
  → 按 initialize_system_prompt() 的长度删除 system prefix
  → 得到 incremental_ids
```

这些 token 会追加到 `response_ids`，但标记为不可训练：

```python
buffer.response_ids.extend(incremental_ids)
buffer.response_mask.extend([0] * len(incremental_ids))
buffer.response_logprobs.extend([0.0] * len(incremental_ids))
```

下一轮真正发送给 rollout backend 的上下文是：

```python
context_ids = (
    initial_prompt_ids
    + previous_generated_ids
    + incremental_message_ids
)
```

新的模型生成再次以 `mask=1` 追加。

---

## 六、一个完整的多轮 token 示例

假设：

```text
初始 system + user:       [10, 11, 12]
第一轮 assistant 输出:    [20, 21]
工具返回及模板连接符:      [30, 31, 32]
第二轮 assistant 输出:    [40, 41]
```

最终 chain buffer 是：

```python
prompt_ids = [10, 11, 12]

response_ids = [
    20, 21,           # assistant turn 1
    30, 31, 32,       # tool result / template glue
    40, 41,           # assistant turn 2
]

response_mask = [
    1, 1,
    0, 0, 0,
    1, 1,
]
```

训练前，Agent Framework 只做张量化和拼接：

```python
prompts = torch.tensor(trajectory.prompt_ids)
responses = torch.tensor(trajectory.response_ids)
input_ids = torch.cat([prompts, responses])

attention_mask = torch.ones_like(input_ids)
loss_mask = response_mask
```

最终：

```python
input_ids = [
    10, 11, 12,
    20, 21,
    30, 31, 32,
    40, 41,
]

loss_mask = [
    # prompt 部分不属于 responses，不进入 response loss
    1, 1,
    0, 0, 0,
    1, 1,
]
```

`rm_scores` 通常只在最后一个 response token 上放置 episode reward：

```python
rm_scores = torch.zeros_like(responses)
rm_scores[-1] = reward_score
```

---

## 七、Uni-Agent 如何处理 message 分叉

Uni-Agent 没有维护一棵显式的 message tree，而是在一个 Session 中维护一个扁平的 active chain 列表：

```text
GatewaySession
├── Chain 1: system → user → assistant A → tool → assistant A2
├── Chain 2: system → user → assistant B
└── Chain 3: subagent system → user → assistant C
```

不同 chain 可以拥有相同的 message 前缀，但每条 chain 都保存一份独立的：

```text
message_history
prompt_ids
response_ids
response_mask
response_logprobs
```

它不是共享节点和 token prefix 的树，而是若干条带有公共逻辑前缀的独立线性轨迹。

### 1. 用链式 message hash 识别前缀

对传入完整 messages，Session 计算：

```text
H1 = hash(message_1)
H2 = hash(H1, message_2)
H3 = hash(H2, message_3)
...
```

每条 chain 保存 `message_tip_hash`。若一条 chain 有 `N` 条消息，则比较：

```python
chain.message_tip_hash == incoming_prefix_hashes[N - 1]
```

成立意味着该 chain 的完整 message history 是新请求的消息级前缀。

比较前会做一定的 canonicalization：

- 忽略随机的 `tool_call_id`；
- 忽略 tool call 自身的随机 `id`；
- 规范化 JSON 字符串和 dict 形式的 tool arguments。

因此工具调用 ID 或 JSON 表示差异不会轻易制造假分叉。

### 2. 普通线性延续

已有：

```text
Chain 1: U → A
```

新请求：

```text
U → A → T
```

Chain 1 是精确前缀，因此 Session：

1. 复制 Chain 1 的 token buffer；
2. 对 `T` 增量编码；
3. 调用模型生成 `A2`；
4. 用更新后的状态替换原 Chain 1。

结果仍然只有：

```text
Chain 1: U → A → T → A2
```

### 3. 相同 prompt 多次采样产生 sibling

连续三次发送相同 prompt：

```text
[U] → A1
[U] → A2
[U] → A3
```

当请求恰好停在 assistant 开始生成的边界时，Uni-Agent 将其视为从同一个 prompt 发起的新采样，而不是已有 assistant 的延续：

```text
Chain 1: U → A1
Chain 2: U → A2
Chain 3: U → A3
```

随后收到：

```text
U → A1 → follow-up
```

只有 Chain 1 的完整历史匹配，所以继续 Chain 1。

若多个 sibling 的 assistant message 内容完全相同，多个 chain 都可以匹配。选择优先级大致是：

1. message history 更深；
2. 精确前缀优先于 rollback；
3. 最近更新的 chain；
4. 较新的 `chain_id`。

### 4. 主 agent 与 sub-agent

主 agent 轨迹：

```text
[system "main", user task, assistant A]
```

sub-agent 请求：

```text
[system "researcher", user subtask]
```

因为不存在匹配的消息前缀，Session 会对 sub-agent 的完整 messages 执行 `encode_full()` 并创建新 chain：

```text
Chain 1: main context...
Chain 2: sub-agent context...
```

主 agent 后续带着原 history 返回时，仍然可以重新匹配并继续 Chain 1。

上下文压缩也是类似逻辑。若 Claude Code 用 summary 替换旧历史：

```text
旧 history:
  [system S, user U, assistant A, tool T]

压缩后:
  [system "Summary so far: ...", user U2]
```

新 messages 不再以旧 chain history 为前缀，因此会创建独立 chain。

### 5. 最后一次 assistant rewrite

Uni-Agent 对最后一个 assistant 的改写有特殊 rollback 逻辑。

已有：

```text
U → assistant BAD
```

新请求没有回放 `BAD`，而是从 assistant 之前的边界继续：

```text
U → user "previous response had an error"
```

如果可以唯一定位到旧 chain，Session 会删除最后一次 assistant 对应的：

```python
response_ids
response_mask
response_logprobs
```

然后从记录的 `last_assistant_start` 重新编码新 suffix。

结果不是保留两个分支：

```text
Branch 1: U → BAD
Branch 2: U → error feedback → FIXED
```

而是原地替换为：

```text
U → error feedback → FIXED
```

被 rollback 的模型 token 不进入最终训练数据。

如果多个 chain 都可能是 rollback 目标，无法唯一判断应该改写哪一条，Session 会倾向于创建新 chain，避免覆盖错误 sibling。

### 6. 并发分叉

同一 Session 的 backend generation 可以并发执行。某条 chain 被一个请求选中后，会进入 `reserved_chain_ids`；其他请求不能同时修改它。

并发选择顺序为：

```text
选择未被占用的匹配 sibling
        ↓
所有匹配 sibling 都被占用
        ↓
对完整 messages 重新编码并创建新 chain
```

这样不同生成请求不会同时写入同一个 mutable token buffer。

---

## 八、Session 结束时没有轨迹重建

Session finalize 时不会执行：

```text
完整 message history
  → apply_chat_template
  → tokenize
  → 重新识别 assistant/tool token
  → 重新推导 loss mask
```

它只把每条 active chain 的 buffer 复制成 `Trajectory`：

```python
Trajectory(
    prompt_ids=list(chain.buffer.prompt_ids),
    response_ids=list(chain.buffer.response_ids),
    response_mask=list(chain.buffer.response_mask),
    response_logprobs=list(chain.buffer.response_logprobs),
    reward_info=...,
    num_turns=...,
)
```

随后 Framework 做：

```text
list → torch.Tensor
prompt_ids + response_ids → input_ids
response_mask → loss_mask
计算 position_ids
放置 rm_scores
写入 TransferQueue
```

因此更准确地说：

> Uni-Agent 的 trajectory 是在 rollout 期间在线构造的；finalize 只是 materialize，Framework 只是 tensorize，不存在最终的 message-to-token 轨迹重建过程。

这意味着 chain buffer 一旦在中途构造错误，finalize 阶段不会根据完整 messages 自动纠正。

### 分叉最终如何进入训练

每条 chain 会 materialize 成一条独立 `Trajectory`，并携带同一份 session-level reward info。Framework 支持两种保留策略：

- `trajectory_selection="all"`：所有分支进入后续打分和 TransferQueue；
- `trajectory_selection="longest"`：只保留 trainable token 数最多的分支。

`longest` 大致按以下优先级选择：

```text
response_mask 中 1 的数量
  → response_ids 总长度
  → turn 数量
  → 输出顺序
```

---

## 九、关键风险：增量 chat template 与完整模板化不一定等价

Claude Code 在真实请求层面，每个 turn 都会发送完整 message list。普通无状态模型服务通常会对每次请求的完整 messages 执行 chat template 和 tokenize。

Uni-Agent 虽然收到完整 message list，但命中已有 chain 后，只模板化新增 messages。

设完整模板化/tokenize 函数为：

```text
F(messages, tools)
```

第一轮：

```text
P0 = F(M0, tools)
模型生成 G0
```

第二轮真实完整模板化应该是：

```text
P1 = F(M0 + assistant(G0) + ΔM, tools)
```

Uni-Agent 当前构造的是：

```text
P1' = P0 + G0 + strip_system(F(ΔM, tools=None))
```

当前实现隐含要求：

```text
P1 == P1'
```

但代码没有验证这个 token-level 等式。

### 1. Chat template 一般不保证可拼接

完整模板可能：

- 在 assistant 和 tool 之间插入结束标记；
- 合并连续 user/tool messages；
- 根据完整 history 选择不同格式；
- 根据是否存在 tool call 改变 assistant 边界；
- 将 tools 动态注入 system prompt；
- 对最后一条消息采用特殊模板；
- 对 reasoning/tool-use block 采用不同包装；
- 校验或调整 user/assistant 角色交替。

因此一般不能假设：

```text
F(history + delta)
=
F(history) + strip_prefix(F(delta))
```

### 2. Assistant 结束 token 可能缺失或重复

考虑模板：

```text
<|im_start|>assistant
{assistant_content}<|im_end|>
<|im_start|>tool
{tool_result}<|im_end|>
<|im_start|>assistant
```

第一轮 rollout backend 返回的 `output.token_ids` 是否包含 `<|im_end|>`，取决于 backend 对 stop token 的处理。

完整重模板化会根据结构化 assistant message 明确插入 `<|im_end|>`。增量拼接则假定这一边界已经正确存在于原始生成 token 或增量模板中。

如果两边都不插入，会缺失；如果两边都插入，会重复。

### 3. Tool schema 在增量路径中没有重新传入

首轮：

```python
encode_full(messages, tools=tools)
```

增量轮：

```python
encode_incremental(incremental_messages)
```

`encode_incremental()` 没有 `tools` 参数。Session 会要求复用 chain 的 tool schema 与新请求相等，但“schema 相等”并不证明模板只需在第一轮渲染一次。

### 4. Tokenizer 也不保证字符串级拼接等价

即使模板字符串看起来满足：

```text
full_text = old_text + delta_text
```

也不普遍保证：

```python
tokenize(full_text)
==
tokenize(old_text) + tokenize(delta_text)
```

BPE/tokenizer 可能跨字符串边界改变切分。消息之间有稳定特殊 token 时风险较低，但这仍然需要验证，而不是由 tokenizer 接口保证。

### 5. Anthropic adapter 的转换可能依赖完整列表

Claude Code 的 Anthropic Messages 请求会先被转换成内部 OpenAI-like messages。该过程会：

- 转换 `tool_use` 和 `tool_result`；
- 规范化 assistant content blocks；
- 处理 system 字段；
- 将 mid-list system reminder 折叠进 user message。

Adapter 每次确实处理完整请求，但 Session 匹配 chain 后只 tokenize 新增的内部 messages，因此 adapter 的完整列表语义不等于完整 chat-template 语义。

---

## 十、这是否造成训练与真实使用不一致

需要区分两种推理路径。

### 场景 A：推理也经过 Uni-Agent stateful Gateway

如果 Claude Code 在推理时仍连接同一个 Uni-Agent Gateway，那么：

```text
训练 rollout：增量模板化
推理 rollout：增量模板化
```

两者内部逻辑一致。这里的问题不是训练与该 Gateway 推理不一致，而是这个 Gateway 送给模型的 token 是否等于标准完整模板化结果。

### 场景 B：真实部署使用普通无状态模型服务

如果真实部署中，Claude Code 每轮把完整 message list 发给普通模型服务，服务端执行：

```text
完整 messages
  → 完整 chat template
  → 完整 tokenize
  → prefix cache 仅作为计算优化
```

而训练使用：

```text
首轮完整编码
  + 历史 sampled token
  + 增量 messages 的独立模板结果
```

那么两者不一定一致。

标准 prefix/KV cache 与 Uni-Agent 当前逻辑的区别是：

```text
标准服务：
  先确定完整请求的 token truth
  再检查 token prefix 能否复用 KV cache

当前 Uni-Agent：
  先假设 message 增量能够模板化拼接
  直接构造下一轮 token truth
```

前者只优化计算，不改变语义；后者在可拼接假设不成立时会改变模型实际看到的 token。

---

## 十一、当前测试验证了什么，缺少什么

现有 continuation 测试通常构造：

```python
expected_prompt_ids = (
    initial_prompt_ids
    + generated_tool_call_ids
    + encode_incremental(tool_message)
)
```

然后断言 Gateway 第二轮收到的 `prompt_ids` 与之相同。

这证明：

> Session、Codec 和测试预期都遵循同一套增量算法。

但它没有证明：

```python
encode_full(complete_second_turn_messages, tools)
==
initial_prompt_ids
+ generated_tool_call_ids
+ encode_incremental(tool_message)
```

因此当前测试属于 implementation consistency test，而不是 full-template parity test。

代码中也保留了：

```python
# TODO: check if delta tokenization is better than remove_system_prompt
```

说明 `remove_system_prompt` 只是当前增量策略，并没有建立通用的模板等价性保证。

---

## 十二、更稳妥的设计：完整模板是语义真值，增量只是优化

更可靠的原则应该是：

> 每轮先从完整 message list 计算模型应该看到的 token；只有 token 前缀验证通过，才允许复用已有 chain buffer。

每一轮执行：

```python
full_ids = encode_full(
    messages,
    tools=tools,
    image_data=full_image_data,
    video_data=full_video_data,
)

old_context_ids = (
    chain.buffer.prompt_ids
    + chain.buffer.response_ids
)
```

然后验证：

```python
full_ids[:len(old_context_ids)] == old_context_ids
```

### 情况 1：严格 token 前缀成立

可以安全计算：

```python
incremental_ids = full_ids[len(old_context_ids):]
```

并追加：

```python
response_ids += incremental_ids
response_mask += [0] * len(incremental_ids)
response_logprobs += [0.0] * len(incremental_ids)
```

此时增量复用只是性能优化，完整模板结果仍然是 token truth。

### 情况 2：严格 token 前缀不成立

说明新一轮完整模板修改了旧 token 上下文，例如：

- assistant 边界不同；
- tool schema 渲染发生变化；
- parser 重写 assistant message；
- chat template 根据完整历史改变格式；
- tokenizer 边界发生变化。

这时不能继续把 delta 拼到旧 buffer。

最安全的处理是：

1. 将旧 chain materialize；
2. 从当前 `full_ids` 创建一条新 trajectory；
3. 当前轮新生成 token 作为 `response_ids`，mask 为 1；
4. 不把旧 sampled token 强行映射到新模板下。

不能简单用新的完整模板重写旧 trajectory 后继续沿用旧 logprob，因为旧 assistant token 是在旧上下文下采样的。上下文改变后，旧 token 与旧 logprob 的 on-policy 语义不再可靠。

### 可选的更细粒度检查

除了完整前缀断言，还可以记录每轮：

```python
TurnSnapshot(
    request_messages,
    full_prompt_ids,
    output_ids,
    output_logprobs,
)
```

导出时：

1. 使用相邻 turn 的 `full_prompt_ids` 计算 longest common prefix；
2. 检查上一轮 output 是否原样出现在下一轮 prompt；
3. 只有严格对齐的 sampled token 才继承 `mask=1`；
4. 被客户端重写或重新模板化的位置降为 context-only，或切分成新 trajectory。

这种设计更接近“每轮完整 token snapshot 是事实，跨轮拼接需要证明”，而不是“在线 buffer 是事实，完整模板等价性靠假设”。

---

## 十三、Uni-Agent 与显式 Message Tree 方案的差异

Uni-Agent 当前的设计可以总结为：

```text
Message 层：
  prefix hash + flat active chains

Token 层：
  每条 chain 在线维护一个持续增长的 buffer

导出层：
  直接 materialize buffer，不重建
```

显式 Message Tree + per-turn token snapshot 的另一类设计则是：

```text
Message 层：
  tree 保存完整分叉拓扑

Token 层：
  每次模型调用保存独立 full prompt/output snapshot

导出层：
  沿 tree path 验证相邻 snapshot 后重建训练 sample
```

两者的主要取舍是：

| 维度 | Uni-Agent 在线 chain buffer | Message Tree + turn snapshot |
|---|---|---|
| 运行时结构 | 简单，直接维护训练序列 | 需要维护 tree 和 turn records |
| sampled token 保存 | 直接保留 backend token | 每轮 snapshot 保留 |
| 多轮拼接 | rollout 时完成 | 导出时验证并完成 |
| finalize 成本 | 很低 | 较高 |
| 模板一致性 | 依赖增量可拼接假设 | 可用 full prompt snapshot 验证 |
| rewrite/compaction | rollback 或新 chain | tree 分叉后按 token provenance 处理 |
| 错误恢复 | buffer 构造错误会直接进入训练 | 导出阶段仍可检测不一致 |

在线构造本身并不是错误。只要每次追加之前验证完整 token prefix，Uni-Agent 的方案可以同时保留低 finalize 成本和 token 正确性。真正危险的是：

> 把 message 前缀相等误当成 chat-template token 前缀相等。

---

## 十四、结论

Uni-Agent 的轨迹管理有三个核心特征。

第一，它把 GatewaySession 当作 token trajectory owner。模型生成 token 直接来自 rollout backend，工具观察和用户续接通过增量 chat template 插入，并用 `response_mask` 区分是否参与训练。

第二，它用扁平的多 `ChainState` 结构处理 message 分叉。message prefix hash 用于选择 chain，相同 prompt 的重复采样形成 sibling，sub-agent 和上下文压缩形成新 chain，最后一次 assistant rewrite 则通过 rollback 原地替换。

第三，它没有 finalize 阶段的轨迹重建。最终训练序列直接来自 chain buffer：

```text
prompt_ids + response_ids → input_ids
response_mask → loss_mask
```

因此当前实现的关键正确性前提是：

```text
完整新 history 的 chat-template token
==
已有 token context
+ 增量 messages 的独立模板 token
```

这个等式对于任意 chat template、tool parser 和 tokenizer 都不自动成立，尤其在 Claude Code、Anthropic content blocks、tool use、assistant rewrite 和动态工具模板场景下需要真实模型 tokenizer 的逐轮验证。

更稳妥的方向是：

```text
完整 message list 模板化结果 = 语义真值
token prefix 检查 = 是否允许增量复用的判据
chain buffer = 通过验证后的在线缓存
```

换言之，增量编码应该是一种有验证的优化，而不应该成为未经验证的 token 语义来源。

---

## 相关代码

主要实现位于：

```text
uni_agent/gateway/session/codec.py
  MessageCodec.encode_full()
  MessageCodec.encode_incremental()

uni_agent/gateway/session/session.py
  TrajectoryBuffer
  ChainState
  GatewaySession._prepare_generation_inputs()
  GatewaySession._select_chain()
  GatewaySession._commit_generation_to_chain()
  GatewaySession._build_materialized_trajectory()

uni_agent/gateway/adapters/openai.py
  openai_to_internal()

uni_agent/gateway/adapters/anthropic.py
  anthropic_to_internal()

uni_agent/framework/framework.py
  _select_session_trajectories()
  _trajectory_to_tq_field_and_tag()
```

相关测试位于：

```text
tests/uni_agent/gateway/test_gateway_actor_on_cpu.py
tests/uni_agent/gateway/test_session_multiple_chains_on_cpu.py
tests/uni_agent/framework/test_generate_sequences_on_cpu.py
```
