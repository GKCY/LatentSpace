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

以当前 SWE-Bench 预处理为例，数据集中的 `prompt` 是只包含问题描述的 OpenAI chat 格式：

```python
prompt = [
    {"role": "user", "content": example["problem_statement"]},
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

当前内置 `run_task` 路径会把 `raw_prompt` 写回样本 Task Config 的 `prompt`；它是数据集的权威
消息来源，`tools_kwargs["task"]` 主要提供任务配置和元数据。任务最终执行：

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
  { "role": "system", "content": "..." },
  { "role": "user", "content": "..." },
  {
    "role": "assistant",
    "content": "",
    "tool_calls": [
      {
        "id": "call_123",
        "type": "function",
        "function": {
          "name": "read_file",
          "arguments": { "path": "foo.py" }
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
    generation_versions,
)
```

各字段的语义是：

| 字段                  | 含义                                                                                                   |
| --------------------- | ------------------------------------------------------------------------------------------------------ |
| `prompt_ids`          | 第一次模型调用时由 Continuous Token builder 构造的 prompt token                                        |
| `response_ids`        | 历次模型生成 token，加上轮次之间插入的上下文 token                                                     |
| `response_mask`       | `1` 表示模型实际生成，`0` 表示工具结果、用户消息或模板连接 token                                       |
| `response_logprobs`   | 请求 logprobs 时与 `response_ids` 对齐；模型生成位置是真实 logprob，上下文位置填 `0.0`，未请求时可为空 |
| `routed_experts`      | 可选的 MoE routing 信息；backend 每轮对完整上下文重算后替换，最终覆盖整条序列                          |
| `generation_versions` | 每次生成对应的 rollout 权重版本范围，rollback 时随被删除的生成一起移除                                 |

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

当 Session 没有找到可复用的 chain 时，它会对完整 message list 构造初始 token 流：

```python
prompt_ids = codec.build_initial_tokens(
    messages,
    tools=tools,
    image_data=image_data,
    video_data=video_data,
)

buffer = TrajectoryBuffer(prompt_ids=prompt_ids)
```

文本模型的底层仍会调用 chat template；当前入口是 Continuous Token builder 的
`build_initial_tokens()`，而不是 Session 自己调用一个 `encode_full()` 方法。其效果相当于：

```python
prompt_ids = apply_chat_template(
    tokenizer,
    messages,
    tools=tools,
    add_generation_prompt=True,
)
```

其中 `add_generation_prompt=True` 会在序列尾部添加 assistant 开始生成所需的模板标记。

多模态模型由 processor-backed Continuous Token builder 处理。Gateway 会把图片、视频和
processor 参数交给同一个初始 token 构造入口；概念上仍然是：

```python
prompt_ids = codec.build_initial_tokens(
    messages,
    tools=tools,
    image_data=image_data,
    video_data=video_data,
)
```

具体模型族可以在 builder 内调用 processor 展开图片占位 token；最终的图像/视频张量由
Framework 在写入训练数据时根据完整媒体列表重建。

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

Session 会先通过 `MessageCodec.merge_assistant_tokens()` 将这些 token 合并到当前 runtime
token 流，再由 Continuous Token builder 对 `response_mask` 和 logprob 做对齐：

```python
response_ids = list(output.token_ids)
runtime_token_ids = buffer.prompt_ids + buffer.response_ids
merged_token_ids, response_mask, response_logprobs = codec.merge_assistant_tokens(
    runtime_token_ids,
    response_ids,
    buffer.response_mask,
    buffer.response_logprobs,  # 请求 logprobs 时传入，否则为 None
    assistant_logprobs=output.log_probs,  # 请求 logprobs 时传入，否则为 None
)
buffer.response_ids = merged_token_ids[len(buffer.prompt_ids):]
buffer.response_mask = response_mask
buffer.response_logprobs = response_logprobs or []
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

## 五、后续模型调用：Continuous Token 合并增量 messages

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

当前实现仍会从已有 chain 中取出增量 messages，但不再调用旧的
`codec.encode_incremental()`，也不再通过固定 system prefix 的长度做截断。Session 会保留
`previous_messages`、`updated_messages` 和当前 runtime token 流，调用：

```python
runtime_token_ids = buffer.prompt_ids + buffer.response_ids
merged_token_ids, merged_response_mask, merged_response_logprobs = (
    codec.merge_context_tokens(
        previous_messages,
        updated_messages,
        runtime_token_ids,
        buffer.response_mask,
        buffer.response_logprobs,  # 请求 logprobs 时传入，否则为 None
        tools=tools,
    )
)
```

`merge_context_tokens()` 由 verl 的 `ContinuousTokenBuilder` 执行。它会：

- 检查 `updated_messages` 是否只是对 `previous_messages` 的 append-only 延续；
- 按 tool、user、system、assistant 分组，使用 suffix diff 构造新增 token；
- 将当前 `tools` 传给增量合并，而不是只在首轮使用；
- 根据模型族修正边界。例如 Qwen/MiniMax 在生成结束 token 后补换行，GLM 在歧义边界处删除重复 token；
- 返回 `MergeResult`，由 `align_response_metadata()` 同步调整 mask 和 logprob。

因此，增量路径更接近：

```text
previous messages + appended messages
  → Continuous Token builder 做 token-level suffix diff
  → 必要时插入/删除模型特定边界 token
  → 对齐 response_mask/response_logprobs
  → 写回 response_ids
```

新增的上下文 token 和 builder 插入的边界 token 都标记为不可训练：

```python
buffer.response_ids = list(merged_token_ids[len(buffer.prompt_ids):])
buffer.response_mask = list(merged_response_mask)
buffer.response_logprobs = list(merged_response_logprobs or [])
```

Session 还会检查合并结果没有修改不可变的 `prompt_ids`。assistant rewrite 仍有单独的
rollback prepare 流程：先裁剪旧 assistant 及其 stale 的 turn separator / generation prompt，
再用同一个 `merge_context_tokens()` 合并 replacement suffix。它与普通追加的入口相同，区别在于
rollback 先恢复到 assistant 之前的 token 边界；不能把这一步误解为保留旧的
`encode_incremental()` API。

下一轮真正发送给 rollout backend 的上下文是：

```python
# 概念式表示；实际以 merge_context_tokens() 返回的 merged_token_ids 为准
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
- 不对 tool arguments 做额外的 JSON 字符串/dict 语义归一化；比较的是 canonicalized message 中保留的 arguments。

因此工具调用 ID 的差异不会轻易制造假分叉，但 arguments 的表示差异仍可能影响匹配。

另外，默认开启的 `coalesce_reserved_exact_requests` 会按 provider-normalized 的
`messages`、`tools` 和 `sampling_params` 指纹合并同一 Session 内的相同在途请求；这类请求不一定
各自产生 sibling。若要保留独立样本，需要关闭该选项。

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
2. 通过 Continuous Token 合并 `T` 的上下文 token；
3. 调用模型生成 `A2`；
4. 用更新后的状态替换原 Chain 1。

结果仍然只有：

```text
Chain 1: U → A → T → A2
```

### 3. 相同 prompt 多次采样产生 sibling

在请求不被同一在途指纹合并的情况下（例如顺序请求，或关闭
`coalesce_reserved_exact_requests`），连续三次发送相同 prompt：

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

这些条件还要经过请求的 `tools`/`active_tool_schemas`、assistant 消息数量和
`reserved_chain_ids` 过滤；只比较 message prefix hash 不足以判断某条 chain 可安全复用。

### 4. 主 agent 与 sub-agent

主 agent 轨迹：

```text
[system "main", user task, assistant A]
```

sub-agent 请求：

```text
[system "researcher", user subtask]
```

因为不存在匹配的消息前缀，Session 会对 sub-agent 的完整 messages 执行
`build_initial_tokens()` 并创建新 chain：

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

### 5. 最后一次（包括 first-assistant）rewrite

Uni-Agent 对最后一个可回滚 assistant 的改写有特殊 rollback 逻辑。当前实现也覆盖
first-assistant rewrite：在能唯一定位原 chain 时复用并 rollback；只有无法唯一定位、被占用或
关闭相应复用开关时才创建新 chain。

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

同时会校验并删除被 rollback assistant 前的 response-side turn separator / generation prompt，
再从记录的 `last_assistant_start` 通过普通增量路径合并新 suffix，避免重复 assistant 开始标记。

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

如果筛选后仍有多个同优先级 chain 都可能是 rollback 目标，且不存在更深或等深的 exact-prefix 候选，
无法唯一判断应该改写哪一条，Session 会创建新 chain，避免覆盖错误 sibling。若 replacement context
已耗尽可用 capacity，当前实现也会丢弃 abandoned chain；
因此 rollback 不保证始终留下一个 prompt-only trajectory。

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
    response_logprobs=(list(chain.buffer.response_logprobs)
                       if chain.buffer.response_logprobs else None),
    num_turns=...,
    chain_id=chain.chain_id,
    routed_experts=chain.buffer.routed_experts,
    multi_modal_data=...,
    extra_fields={
        "min_global_steps": ...,
        "max_global_steps": ...,
        "mm_processor_kwargs": ...,  # 多模态样本需要时写入
    },
)
```

`finished`、`reward_score` 和 `reward_metrics` 由 Agent Framework 在 Runner 返回后再附加，
不是 Gateway finalize 时的 `reward_info` 字段。

随后 Framework 做：

```text
list → torch.Tensor
prompt_ids + response_ids → input_ids
response_mask → loss_mask
计算 position_ids
放置 rm_scores
写入 TransferQueue
```

如果启用 `mask_unfinished_episode`，Framework 还会把未完成 trajectory 的 response/loss mask
置为 0；因此 `response_mask=1` 表示 rollout 阶段的 sampled-token provenance，不保证该位置最终
一定进入训练损失。

因此更准确地说：

> Uni-Agent 的 trajectory 是在 rollout 期间在线构造的；finalize 只是 materialize，Framework 主要负责
> 张量化、奖励/状态附加和可选后处理，不存在最终的 message-to-token 轨迹重建过程。

这意味着 chain buffer 一旦在中途构造错误，finalize 阶段不会根据完整 messages 自动纠正。

### 分叉最终如何进入训练

每条 chain 会 materialize 成一条独立 `Trajectory`；Runner 返回后，Framework 再把同一份
session-level `finished`/`reward_score`/`reward_metrics` 挂到该 session 保留的 trajectories 上。当前处理顺序是
Gateway finalize → trajectory selection → 可选 `trajectory_postprocessor` → reward scoring →
TransferQueue/logging。Framework 支持两种保留策略：

- `trajectory_selection="all"`：所有分支先保留；经过可选 postprocessor 过滤、裁剪或重排后写入
  后续 TransferQueue。启用 RewardLoopWorker 时，当前默认只对 session 保留列表中的代表 trajectory
  评分，再将结果广播到该 session 的其他分支，并不等于每个分支独立重算 reward；
- `trajectory_selection="longest"`：只保留 trainable token 数最多的分支。

`longest` 大致按以下优先级选择：

```text
response_mask 中 1 的数量
  → response_ids 总长度
  → turn 数量
  → 输出索引（当前 max key 下较晚项优先）
```

---

## 九、关键风险：Continuous Token 已修正边界，但不等于全历史 parity

Claude Code 在请求层面仍会发送完整 message list。当前 Gateway 命中已有 chain 后，不是把
`ΔM` 单独执行一次 chat template 再删除 system prefix，而是把已有 runtime token 流交给
`ContinuousTokenBuilder.merge_context_tokens(previous_messages, updated_messages, tools=tools)`。

该 builder 会对追加消息做 append-only 检查和 token-level suffix diff，并由模型族实现处理边界：

- 包括 Qwen/MiniMax 在生成结束 token 后补缺失的换行；
- 包括 GLM 在 observation/user 歧义边界处删除重复 token；其他模型族还有各自的 builder 边界规则；
- `MergeResult` 记录插入或删除的 token，`align_response_metadata()` 同步修正 mask 和 logprob。

因此旧文中的下面这条公式已经不再描述当前实现：

```text
P1' = P0 + G0 + strip_system(F(ΔM, tools=None))
```

当前路径更接近：

```text
runtime_token_ids
  + CT(previous_messages, updated_messages, tools)
  + model-specific boundary edits
  → next context_ids
```

### 1. 仍需区分 CT suffix 校验和完整历史 parity

Continuous Token 的 `render_delta_token_id()` 会基于 synthetic anchor 对用于构造 suffix 的
prefix/full 模板做 token 前缀检查，但 Gateway 并不会在每一轮都把真实完整 history 的全量模板结果
与最终 runtime token 流逐 token 比较。
因此仍需验证：

- 模型族 builder 对当前 chat template 的边界规则是否完整；
- append-only 消息分组是否覆盖 Anthropic content blocks、reasoning 和 tool-use 变体；
- 工具 schema、`apply_chat_template_kwargs` 或 tokenizer 更新后，suffix diff 是否仍保持一致。

### 2. 多模态增量有明确限制

当前 `merge_context_tokens()` 对增量 image/video data 会显式报错；多模态首轮可以通过 processor
构造 token，但增量媒体上下文尚未由 Gateway CT 路径支持。不能把文本增量路径的结论直接推广到多模态。

### 3. Anthropic adapter 仍是独立的一致性边界

Anthropic 请求会先转换成内部消息，再参与 prefix 匹配和 CT 合并。当前 adapter 会把中间位置的
system 内容折叠为 user 侧的 `<system-reminder>`，丢弃 thinking block（并在必要时警告 prefix drift），
同时拒绝 `redacted_thinking`。因此 `tool_use`、`tool_result`、system 字段及 content blocks 的
规范化必须与对应模型族 builder 一致；adapter 能正确转换请求，也不自动证明 CT suffix 与标准
完整模板结果逐 token 相同。

---

## 十、这是否造成训练与真实使用不一致

需要区分两种推理路径。

### 场景 A：推理也经过 Uni-Agent stateful Gateway

如果 Claude Code 在推理时仍连接同一个 Uni-Agent Gateway，那么：

```text
训练 rollout：Continuous Token 增量合并
推理 rollout：Continuous Token 增量合并
```

两者内部逻辑一致。这里的问题不是训练与该 Gateway 推理不一致，而是当前模型族的
Continuous Token builder 是否覆盖了标准完整模板的全部边界语义。

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
  + Continuous Token 合并的增量 messages
```

那么两者不一定一致。

标准 prefix/KV cache 与 Uni-Agent 当前逻辑的区别是：

```text
标准服务：
  先确定完整请求的 token truth
  再检查 token prefix 能否复用 KV cache

当前 Uni-Agent：
  通过 Continuous Token builder 构造并修正下一轮 token 流
```

前者只优化计算，不改变语义；后者依赖 builder 对具体模型模板的覆盖和测试。

---

## 十一、当前测试验证了什么，缺少什么

现有 continuation 测试通常构造：

```python
# 概念性表示；实际由 codec.merge_context_tokens() 返回
expected_prompt_ids = (
    initial_prompt_ids
    + generated_tool_call_ids
    + continuous_token_context_delta(previous_messages, updated_messages, tools)
)
```

然后断言 Gateway 第二轮收到的 `prompt_ids` 与之相同。

这证明：

> Session、Codec 和测试预期都遵循同一套增量算法。

但它没有证明：

```python
# 概念性完整模板结果
full_template(complete_second_turn_messages, tools)
==
initial_prompt_ids
+ generated_tool_call_ids
+ continuous_token_context_delta(previous_messages, updated_messages, tools)
```

当前 main 已增加 Continuous Token 的模型特定边界测试；但如果没有把真实完整 history 的
`full_template(...)` 与最终 runtime token 流逐 token 对比，仍然只能证明实现内部一致，不能证明
完整模板 parity。

旧实现中的：

```python
# TODO: check if delta tokenization is better than remove_system_prompt
```

以及 `remove_system_prompt` 路径已经被 Continuous Token 集成替代；这段 TODO 不能再作为当前
main 的实现依据。

---

## 十二、进一步增强：用完整模板检查非 assistant 结构

当前 Continuous Token 已经把增量合并从未经处理的 prefix strip 提升为带模型特化边界处理的优化，
但仍可以把完整 message list 的 canonical 模板结果作为独立比较基准。它不能无条件当作整个 runtime
token 流的绝对真值：某些模板会重写或省略历史 assistant reasoning/thinking，而旧 assistant token
及其 logprob 必须保留 token-in-token-out provenance。更可靠的原则是：

> 用完整模板检查稳定的非 assistant 结构和边界；只有在模型模板保证历史 assistant token 不会被重写时，
> 才把严格 token 前缀检查作为复用已有 chain buffer 的充分条件。

每一轮执行：

```python
full_ids = full_template(
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

这里的 `full_ids` 是 canonical comparator；它对历史 assistant token 的改写不能直接覆盖
`old_context_ids` 中已经采样的 token。

然后验证：

```python
full_ids[:len(old_context_ids)] == old_context_ids
```

### 情况 1：严格 token 前缀成立且模板保留历史 assistant token

如果严格 token 前缀成立，可以安全计算：

```python
incremental_ids = full_ids[len(old_context_ids):]
```

并追加：

```python
response_ids += incremental_ids
response_mask += [0] * len(incremental_ids)
response_logprobs += [0.0] * len(incremental_ids)
```

此时增量复用只是性能优化，完整模板结果可作为 token truth。若模板不提供历史 assistant 保留保证，
则只能把这个结果用于非 assistant 边界的校验，不能据此重写旧 sampled token。

### 情况 2：严格 token 前缀不成立

这可能表示新一轮完整模板修改了稳定的旧 token 上下文，也可能只是 canonical 模板重写了历史
assistant reasoning/thinking。需要先区分两类情况：

- 非 assistant 结构、tool schema 或边界发生变化，例如：

  - assistant 边界不同；
  - tool schema 渲染发生变化；
  - chat template 根据完整历史改变格式；
  - tokenizer 边界发生变化。

- 仅历史 assistant reasoning/thinking 被 canonical 模板重写；这不是旧 sampled token 可以被静默替换的理由。

第一类情况不能继续把 delta 静默拼到旧 buffer；第二类情况应保留旧 sampled token，改用
Continuous Token 的模型族边界校验或切分新 trajectory。

最安全的处理是：

1. 将旧 chain materialize；
2. 对稳定的非 assistant 结构从当前 `full_ids` 创建一条新 trajectory；
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

这种校验设计可以在保留在线 buffer 和低 finalize 成本的同时，发现 Continuous Token builder
与真实完整模板之间的差异。

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

| 维度               | Uni-Agent 在线 chain buffer                          | Message Tree + turn snapshot        |
| ------------------ | ---------------------------------------------------- | ----------------------------------- |
| 运行时结构         | 简单，直接维护训练序列                               | 需要维护 tree 和 turn records       |
| sampled token 保存 | 直接保留 backend token                               | 每轮 snapshot 保留                  |
| 多轮拼接           | rollout 时完成                                       | 导出时验证并完成                    |
| finalize 成本      | 很低                                                 | 较高                                |
| 模板一致性         | CT suffix/boundary 校验；仍可增加 full prompt parity | 可用 full prompt snapshot 验证      |
| rewrite/compaction | rollback 或新 chain                                  | tree 分叉后按 token provenance 处理 |
| 错误恢复           | buffer 构造错误会直接进入训练                        | 导出阶段仍可检测不一致              |

在线构造本身并不是错误。当前实现已在 CT 合并阶段验证 append-only 和局部 suffix；若再增加完整
token prefix parity 检查，便能同时保留低 finalize 成本和更强的 token 正确性。真正危险的是：

> 把 message 前缀相等误当成 chat-template token 前缀相等。

---

## 十四、结论

Uni-Agent 的轨迹管理有三个核心特征。

第一，它把 GatewaySession 当作 token trajectory owner。模型生成 token 直接来自 rollout backend，工具观察和用户续接通过 Continuous Token 增量合并插入，并用 `response_mask` 区分是否参与训练。

第二，它用扁平的多 `ChainState` 结构处理 message 分叉。message prefix hash 用于选择 chain，相同 prompt 的重复采样形成 sibling，sub-agent 和上下文压缩形成新 chain；可唯一定位的 latest/first assistant rewrite 通过 rollback 原地替换，否则创建新 chain。

第三，它没有 finalize 阶段的轨迹重建。最终训练序列直接来自 chain buffer：

```text
prompt_ids + response_ids → input_ids
response_mask → loss_mask
```

因此当前实现的关键正确性边界是：

```text
append-only messages
  → Continuous Token suffix diff
  → 模型特定边界修正
  → mask/logprob 对齐
```

普通追加路径已经处理了旧实现中的 system-prefix strip 和常见边界问题，但对任意 chat template、tool parser
和 tokenizer 仍不自动提供全历史 parity，尤其在 Claude Code、Anthropic content blocks、tool use、
assistant rewrite 和动态工具模板场景下需要真实模型 tokenizer 的逐轮验证。

更稳妥的方向是：

```text
完整 message list 模板化结果 = 语义真值
token prefix 检查 = 是否允许增量复用的判据
chain buffer = 通过验证后的在线缓存
```

换言之，当前 Continuous Token 是有局部验证的增量优化；完整模板 parity 仍应作为额外的回归检查。

---

## 相关代码

主要实现位于：

```text
uni_agent/gateway/session/codec.py
  MessageCodec.build_initial_tokens()
  MessageCodec.merge_context_tokens()
  MessageCodec.merge_assistant_tokens()
  ContinuousTokenBuilder.align_response_metadata()

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

## 核验基准

本文按 2026-09-20 可见的 Uni-Agent [`main`](https://github.com/verl-project/uni-agent/tree/main) 核对。
Continuous Token 路径由 [PR #101](https://github.com/verl-project/uni-agent/pull/101) 引入，并在
[PR #164](https://github.com/verl-project/uni-agent/pull/164) 继续整合；first-assistant rollback、奖励 API
和 trajectory postprocessor 又分别在
[PR #123](https://github.com/verl-project/uni-agent/pull/123)、
[PR #109](https://github.com/verl-project/uni-agent/pull/109) 和
[PR #143](https://github.com/verl-project/uni-agent/pull/143) 后继续变化；正文按当前 `main`
实现，而不是只按早期某个提交核对。实现细节以当前的
[`codec.py`](https://raw.githubusercontent.com/verl-project/uni-agent/main/uni_agent/gateway/session/codec.py)、
[`session.py`](https://raw.githubusercontent.com/verl-project/uni-agent/main/uni_agent/gateway/session/session.py)、
[`types.py`](https://raw.githubusercontent.com/verl-project/uni-agent/main/uni_agent/gateway/session/types.py)
和 [`framework.py`](https://raw.githubusercontent.com/verl-project/uni-agent/main/uni_agent/framework/framework.py)
为准；Gateway 的轨迹语义见
[官方文档](https://uni-agent.readthedocs.io/en/latest/concepts/gateway-and-trajectories.html)。
