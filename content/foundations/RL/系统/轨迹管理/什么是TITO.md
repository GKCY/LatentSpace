# TITO 问题及解决方案

## 写作目标

本文准备分析 TITO 问题中的 token drift：agentic RL 多轮 rollout 过程中，真实 token 历史如何在框架重建、消息序列化、模板渲染和训练样本拼接时发生漂移，以及如何在系统设计上避免这种漂移。

## 初步结构

### 1. 什么是 TITO

TITO 是 Token-In-Token-Out 的缩写。它描述的是 agentic RL 多轮 rollout 中一个 token 级别的不变量：模型在下一轮推理中看到的 prompt，必须把上一轮推理实际消费和生成过的 token 序列原样保留下来。

更形式化地说，设第 `n` 轮调用 inference engine 时的输入 token 序列为 `P_n`，模型在这一轮生成的 token 序列为 `R_n`。那么 TITO 要求：

```text
P_n + R_n == prefix(P_{n+1})
```

也就是说，第 `n` 轮的完整 token 序列，也就是 `prompt + response`，必须是第 `n+1` 轮 prompt token 序列的 bit-perfect prefix。这里强调的是 token ID 级别的一致，而不是解码后的文本语义一致。只要中间发生了 detokenize-retokenize、chat template 重新渲染、消息结构 parse 再 serialize、历史 reasoning 被模板裁剪等操作，即便文本看起来语义相同，也可能已经破坏 TITO。

所以这里讨论的 TITO 问题，范围只限于 token 序列本身在多轮交互中被改写。只要 trainer 看到的 token 历史不是 rollout 当时真实使用的 token 历史，问题就已经发生了。

TITO 问题就是 TITO invariant 被破坏后的训练-推理错配。此时某个 token `x_t` 在 rollout 中是基于上下文 `x_1...x_{t-1}` 采样出来的，但 trainer 计算 loss/logprob 时，可能用的是另一个上下文 `x'_1...x'_{t-1}`。即使模型权重完全一样，`pi(x_t | x_1...x_{t-1})` 和 `pi(x_t | x'_1...x'_{t-1})` 也可能差很多，于是名义上的 on-policy 更新会变成带偏差的 off-policy 更新。

这个定义也解释了为什么 TITO 在 agentic RL 里特别重要：一个任务往往不是单次生成，而是模型调用、工具结果、harness message、继续生成之间反复交替。如果每一轮都把历史消息重新渲染成字符串再 tokenize，系统很容易在不知不觉中改变 token 历史。TITO 的核心主张是：rollout 应该维护一个随 turn append-only 增长的 token buffer，并把这个真实 token buffer 直接交给训练侧，而不是依赖语义等价的文本或消息列表重建训练序列。

### 2. 问题来源

TITO 的 token drift 通常不是模型生成错了，而是 agent runtime 在多轮交互中把“真实 token 历史”降级成了文本、message list 或结构化字段，然后又试图从这些中间表示重建 prompt。这个重建过程只要不是 bit-perfect，就会让 trainer 看到一段模型当时没有真正使用过的上下文。

#### 2.1 Detokenize-retokenize 导致 token 序列不可逆

最直接的一类来源，是把模型生成的 token 先 decode 成字符串保存，下一轮再把字符串 encode 回 token。

这个过程看起来无害，因为文本内容没有变；但 tokenizer 的 `decode` 和 `encode` 不是互逆关系。`encode(text)` 通常会为同一段文本选择一个标准切分，例如最长匹配或 tokenizer 认为最 canonical 的切分；而 `decode(tokens)` 是多对一的：不同 token 序列可能 decode 成完全相同的字符串。

所以模型在 rollout 中实际采样出的 token 序列，不一定是 tokenizer 重新 encode 这段文本时会选出的 token 序列。例如模型可能用两个 token 拼出一个词，而重新 encode 时 tokenizer 可能直接用一个更长的 token 表示同一个词。文本表面相同，但 token ID 序列已经变了。

这会破坏 TITO 的原因是：第 `n` 轮的 response token 已经不是第 `n+1` 轮 prompt 的原样 prefix。trainer 之后计算 logprob 时，计算的是“canonical retokenize 后的 token 序列”，不是模型真实采样过的那串 token。对 RL 来说，这等于把一段实际 trajectory 换成了另一段 token trajectory。

这类问题最容易出现在：

- rollout 只保存 decoded text，不保存生成 token IDs
- 数据集落盘时为了可读性只存 message text
- 训练前重新用 tokenizer/tokenizer.apply_chat_template 构造 `input_ids`
- 中间服务用字符串 API 传递模型输出，而不是 token API

#### 2.2 每轮从 message list 重渲染 chat template

第二类来源，是 agent runtime 每一轮都维护一个 JSON-like message list，然后在调用 inference engine 前重新套 chat template，把整个历史消息重新渲染成 prompt 字符串，再 tokenize。

这在普通 chat serving 中很自然，但在 TITO 视角下很危险。因为 chat template 不是一个纯粹的“把消息拼起来”的函数，它经常会处理 role header、特殊 token、换行、空格、escape、tool call 格式、reasoning 字段、generation prompt 等细节。只要模板的渲染结果依赖“当前 message list 的整体形态”，那么后来追加一条消息，就可能改变前面历史消息的渲染方式。

一旦历史部分被重新渲染，`P_n + R_n` 就不再保证是 `P_{n+1}` 的 prefix。也就是说，第 `n+1` 轮 prompt 不是在上一轮真实 token buffer 后面 append 新 observation，而是从 message list 重新生成了一份“看起来等价”的 prompt。

这类漂移的典型机制包括：

- 追加新的 tool message 后，模板重新决定前一个 tool response 所在 block 的结束位置
- 追加新 message 后，模板根据新的 role 组合回头增删旧片段附近的换行、role header 或特殊 token

以 Qwen 系列的 chat template 为例，这些机制并不是抽象风险，而是模板里真实存在的分支逻辑。下面只看一种狭义场景：系统没有修改旧 message，只是在 message list 末尾 append 新内容，但重渲染后旧 token 被重新解释。

第一个例子是连续 tool response。Qwen2.5-Instruct 和 Qwen3 的模板都会根据 `tool` message 是否连续来决定 user block 的开闭。`tool` role 并不会被渲染成独立的 `<|im_start|>tool` block，而是被包进一个 user block 里的 `<tool_response>`。如果当前 tool message 后面没有另一个 tool message，模板才会补上 `<|im_end|>\n`。

只有一个 tool result 时，渲染大致是：

```text
<|im_start|>user
<tool_response>
A
</tool_response><|im_end|>
```

后面又追加一个连续 tool result，重渲染后会变成：

```text
<|im_start|>user
<tool_response>
A
</tool_response>
<tool_response>
B
</tool_response><|im_end|>
```

注意这里没有改旧 message，只是 append 了 `B` 这个新 tool message。但重渲染后，旧片段 `A` 后面的 `<|im_end|>` 被推迟到了 `B` 后面。上一轮真实 prompt 如果已经包含 `A</tool_response><|im_end|>\n`，下一轮 prompt 就不再以它为 prefix，TITO 被破坏。

所以 TITO 要求的不是“每轮 message list 语义一致”，而是“每轮新增内容只能 append 到真实 token buffer 后面，不能让旧 token 被重新解释”。

#### 2.3 Reasoning 被 chat template 的 cut-thinking 规则剪掉

第三类来源发生在 reasoning model 的 chat template 上。一些 reasoning 模型模板会把历史 assistant reasoning 当作可清理内容：在某个边界之前的 reasoning 不再渲染到下一轮 prompt 里。LMSYS 文章把这类边界称为 cut-thinking boundary。

问题在 agentic RL 里会被放大，因为 harness 不只是追加 tool message，还可能在任务中间注入 user message。例如终端输出可能被包装成 user role；解析失败、格式错误、重试提示也可能以 user role 插入。很多模板会根据最后一个 user message 移动 reasoning 保留边界，于是每注入一次 user message，边界就往后推一次，边界之前的 assistant reasoning 可能被静默删掉。

以 Qwen3 的 chat template 为例，模板会从后往前找最近一个普通 user message，计算 `last_query_index`，然后对 assistant reasoning 做不同渲染。只有在某些位置之后的 assistant message，模板才会把 `reasoning_content` 重新包进：

```text
<think>
...
</think>
```

例如某一轮只有 `user_0 -> assistant_1`，`assistant_1` 位于最近 user message 之后，模板会保留它的 thinking。之后 runtime 只是 append 一个新的普通 `user_2` message。此时最近 user message 变成 `user_2`，`assistant_1` 变成了边界之前的历史 assistant，重渲染时它的 reasoning 可能不再被保留。旧 assistant message 没有被修改，但它在 prompt 里的 token 表示已经变了。

这正是 cut-thinking boundary 的风险：append 一个 user observation、错误提示或环境反馈，就可能让模板回头改变历史 assistant thinking 的渲染方式。对 TITO 来说，这不是“少展示了一段思考”，而是旧 prefix 被改写。

从文本或 UI 看，这像是模板在帮你清理历史思考；但从 TITO 看，这是直接改写了 token 历史。上一轮模型确实是在包含那些 reasoning token 的上下文中继续生成的，下一轮 prompt 却把它们移除了。于是后续 token 在 trainer 侧对应的 prefix，已经不是 rollout 侧采样时的 prefix。

这类问题的危险之处在于它不是乱码或 crash，而是“模板按设计工作”。如果没有 token prefix 检查，训练样本看起来仍然是合法多轮对话，只是缺了一段模型真实看过的历史 reasoning。

#### 2.4 Tool call 和 tool result 的 parse-then-serialize 漂移

第四类来源，是结构化字段的 parse/serialize 往返。agentic rollout 里，模型经常生成 tool call，runtime 会解析这段文本，得到结构化对象，再在下一轮 message list 里以 `tool_calls`、`tool_result` 或类似字段保存。

问题是：结构化对象保留的是语义，不一定保留原始字节。模型可能生成了紧凑 JSON、特定 key 顺序、特定空格和换行、某种 escape 写法；runtime 一旦 parse 成对象，再用模板里的 `tojson` 或 serializer 渲染回字符串，就可能引入默认空格、改变 key 顺序、规范化 escape，甚至调整换行。

例如 rollout 侧真实采样的是紧凑 JSON：

```json
{"name":"bash","arguments":{"cmd":"ls"}}
```

下一轮模板重渲染时可能变成：

```json
{"name": "bash", "arguments": {"cmd": "ls"}}
```

两者语义完全一样，但字节不同，token IDs 也可能不同。TITO 关心的不是 JSON 语义是否等价，而是下一轮 prompt 是否原样包含上一轮模型生成过的 token。parse-then-serialize 的默认行为通常做不到这一点。

这类漂移也可能出现在：

- tool result 被漂亮打印、压缩、排序或转义
- shell 输出末尾换行被吞掉或补上
- harness 把 stderr/stdout 包装成新的文本格式
- engine retry message 把原始错误重新组织成另一段 prompt
- 日志/存储层对字符串做 Unicode normalization 或 escape

#### 2.5 Role 边界和特殊 token 处理不一致

第五类来源，是 role 边界附近的特殊 token、换行和 stop token 处理。TITO 不只要求普通文本 token 连续，也要求 `<|im_start|>`、`<|im_end|>`、BOS/EOS、role header、工具边界标记等特殊 token 连续。

现实系统里，模型停止生成的位置和 chat template 认为一轮 assistant message 应该结束的位置不总是一致。例如模型可能在某个 end-of-message token 处停止，但模板规范里这个 token 后面还应该有一个换行；如果下一轮直接在模型停止 token 后拼接 tool response，就会少一个换行 token。反过来，如果训练侧重渲染模板时补上了这个换行，trainer 看到的 prefix 又会比 rollout 侧多一个 token。

还有一种情况是 stop token 同时承担“上一段结束”和“下一段开始”的作用。模型可能生成了一个边界 token，但 harness 下一步注入的 role 与这个边界 token 暗示的 role 不一致。此时如果系统简单 append 新消息，就会得到错误边界；如果系统重渲染整个 prompt，又会改写历史 token。

所以 TITO 系统通常需要在 splice point 做模型相关的边界修补：哪些 token 是模型真实生成并参与 loss 的，哪些 token 是为了让下一段消息合法接上而补的，必须在 token buffer 和 loss mask 里明确区分。

#### 2.6 Message list 被修改，而不是 append-only 增长

第六类来源，是 agent runtime 没有把 trajectory 当作 append-only 日志，而是会回头修改已有 message。

这类修改可能来自很多正常工程需求：把解析失败的 assistant 输出替换成标准 tool call；把长 tool output 截短；把敏感字段脱敏；把错误消息合并；把中间 system reminder 移动到更靠后的位置；为了 UI 展示把 assistant reasoning 折叠到独立字段。每一个操作从产品或日志角度都可能合理，但只要它改变了已经被 inference engine 消费过的历史 token，就会破坏 TITO。

TITO 下更稳妥的数据模型是：历史 token buffer 一旦被模型消费或生成，就不能再被重写。后处理可以新增 segment、增加 metadata、增加 mask，甚至标记某些区域不训练；但不能把旧 token 当作可重新格式化的文本。

#### 2.7 长上下文截断按文本/message 做，而不是按 token buffer 做

第七类来源，是长上下文处理。agentic 任务经常会有几十轮交互，prompt 很快接近上下文窗口。系统如果按 message 数量、文本长度、UI 可读性或语义摘要来裁剪历史，就很容易裁掉模型真实依赖过的 token，或者在裁剪边界处重新生成分隔符。

截断本身不一定违反 TITO；模型上下文窗口有限，旧 token 可以被明确移出窗口。但它必须是 token-aware 的：第 `n+1` 轮 prompt 应该对应一个清楚定义的 token suffix，而不是对 message list 做一次文本级压缩后重新 tokenize。否则 trainer 很难知道某个 response token 到底是在什么 prefix 下被采样的。

在训练样本里，截断还会影响 loss mask 和 reward attribution。如果裁剪点不是 token segment 边界，或者裁剪后重新套模板插入了新的 role header，就可能导致 response 区间、tool 区间、prompt 区间全部错位。

#### 2.8 Tokenizer、chat template 和 special token 配置不一致

最后一类来源，是 rollout 侧和训练侧使用的 tokenizer/template 配置不一致。即使系统保存了 message list，只要两边 tokenizer 版本、special token 表、chat template 参数、thinking 开关、tool 格式配置不同，重建出的 token 序列就可能不同。

这类问题在多模型、多 harness、多训练 worker 的系统里尤其常见：rollout 服务升级了 tokenizer，训练 job 还在用旧版本；某个模型的 thinking template 默认清理历史 reasoning；某个 harness 开启了 user-role tool output；另一个训练脚本没有传同样的 template kwarg。

TITO 的要求是把这些配置当作训练数据契约的一部分。样本不应该只说“这是 Qwen 的一段对话”，而应该能明确回答：用的是哪个 tokenizer、哪个 chat template、哪些 special token、哪些 role 可以被 harness append、thinking/reasoning 是否保留，以及 token buffer 是如何从这些配置生成和验证的。

### 3. 影响

TITO 被破坏后，最麻烦的地方不是文本看起来变了，而是训练系统里“状态”和“动作”的定义变了。RL 里一次 token 级 action 本来应该是：模型在 rollout 时基于 prefix `x_1...x_{t-1}` 采样 token `x_t`。如果训练侧重建出来的是 `x'_1...x'_{t-1}` 或 `x'_t`，那 trainer 优化的就不再是当时真实发生过的那次 action。

#### 3.1 Trainer 在错误上下文下计算 logprob

最直接的影响，是 logprob 计算对不上。rollout 中 token `x_t` 的真实概率应该是：

```text
pi(x_t | x_1...x_{t-1})
```

但 token drift 后，trainer 可能计算成：

```text
pi(x_t | x'_1...x'_{t-1})
```

甚至在 detokenize-retokenize 的场景下，`x_t` 本身也可能变成另一个 token 切分。这样一来，trainer 看到的不是“模型当时为什么生成这个 token”，而是“模型在另一个 prefix 下对这个 token 的概率”。这会污染 PPO/GRPO 里最核心的 logprob、KL、ratio 和 entropy 统计。

对 PPO 来说，policy ratio 通常依赖新旧 logprob 之差：

```text
r_t = exp(logprob_new - logprob_old)
```

如果 `logprob_old` 对应 rollout 真实 prefix，而 `logprob_new` 对应漂移后的 prefix，二者就不是同一个状态-动作对上的概率比。ratio 看起来仍然是一个正常数字，但语义已经错了。clip、KL penalty、advantage weighting 都会建立在这个错误 ratio 上。

#### 3.2 On-policy 假设被静默破坏

TITO 问题本质上会把 on-policy RL 变成一种隐式 off-policy 训练。区别在于，这里不是因为 policy 权重版本过旧，而是因为训练侧 state 被改写了。

在 agentic RL 中，state 就是当前 token prefix。模型下一步能看到哪些 tool output、哪些 reasoning、哪些 role header、哪些边界 token，都会影响下一步 action 分布。如果 token drift 改变了 prefix，那么 trainer 优化的是另一个 state 下的 action。更严重的是，这种 off-policy 偏差通常不会被系统显式标出来：样本文本仍然可读，reward 仍然存在，loss 也能正常下降。

这会导致一个很隐蔽的后果：训练可能在“奖励正确、样本合法、loss 正常”的表象下，把梯度打到错误的条件分布上。模型被强化或惩罚的，不是它在真实 rollout 中做出的选择，而是某个重建 prompt 下看起来对应的选择。

#### 3.3 Action mask 和 loss mask 容易错位

大模型 RL 训练通常不会对整段 token 都算 loss。系统会区分 prompt token、assistant response token、tool call token、tool result token、environment message、padding token 等不同区域，然后用 action mask 或 loss mask 决定哪些 token 参与训练。

token drift 会破坏这些边界。比如一个 tool call 被 parse 后重新 serialize，token 数变了；一个 `<|im_end|>` 被模板移动了；一个 `<think>...</think>` block 被剪掉了。此时原来记录的 response 起止位置、tool call 起止位置、assistant/action token 范围都可能偏移。

错位后的后果包括：

- prompt token 被当成 action token 训练
- tool observation 被误算 loss
- assistant response 的一部分被 mask 掉
- reward 或 advantage 被广播到错误 token 区间
- EOS、role header、tool boundary token 被错误强化或错误忽略

这些错误通常不会导致 tensor shape 报错，因为序列仍然能 padding 成 batch。真正的问题是训练信号被施加到了错误位置。

#### 3.4 Reward attribution 变得不可信

agentic RL 的 reward 往往是 trajectory 级别的：任务是否成功、代码是否通过测试、工具调用是否有效、最终答案是否正确。训练时再把这个 reward 或 advantage 分配到一段 assistant token 上。

如果 TITO 被破坏，reward 对应的真实 trajectory 和 trainer 看到的 token trajectory 就不是同一个东西。例如 rollout 中模型先生成某段 reasoning，再发出 tool call，工具返回结果后继续生成；但训练样本里历史 reasoning 被剪掉，tool call JSON 被重排，tool response 边界也变了。最终 reward 仍然来自真实环境执行结果，但梯度却施加在重建后的 token 序列上。

这会让 credit assignment 更差。模型可能因为一个成功 trajectory 得到正向强化，但被强化的 token 上下文不是它当时成功所依赖的上下文；也可能因为失败 trajectory 被惩罚，但训练侧已经看不到导致失败的关键 reasoning 或 tool boundary。长链路任务里，这种错配会不断放大。

#### 3.5 数据集看起来合法，但不能真实 replay

TITO 漂移后的样本往往仍然是合法 JSON、合法 chat message、合法 `input_ids`。这也是它难发现的原因。人工看文本时，会觉得对话语义没有明显问题；训练脚本也能正常跑；甚至 loss 曲线也可能平滑。

但这类样本已经失去了一个关键能力：不能作为真实 rollout 的 replay log。你无法保证把样本喂回 inference engine 时，每一步模型看到的 prefix 和原始 rollout 一样。这样一来，后续所有诊断都会变弱：

- 复现某条失败 trajectory 时，模型看到的上下文可能已经不同
- 对比 rollout logprob 和 trainer logprob 时，不知道差异来自模型还是 token drift
- 分析某个 tool call 为什么出现时，训练样本里的前文可能已经不是当时前文
- 做数据清洗时，文本级 diff 看不出 token 级 prefix 是否被破坏

这会把问题从“训练 bug”变成“数据真实性 bug”。后者更难排查，因为它藏在系统边界和序列化细节里。

#### 3.6 多轮任务中漂移会累积

单轮 chat 里，一个边界 token 或换行漂移可能只是局部问题；但 agentic rollout 是多轮的。每一轮 prompt 都建立在上一轮 token 历史之上，只要某一轮出现 drift，后续所有 token 的条件分布都可能被污染。

这种累积效应有两个特点。

第一，它不是线性可控的。早期一个 role boundary 被改写，可能改变后面模型是否进入 tool-call 模式、是否保留 reasoning、是否继续生成某类格式。后面看到的差异可能已经很大，但根因是很早之前一个很小的 token splice 错误。

第二，它很难从最终文本判断。最终答案可能仍然正确，工具调用也可能成功，reward 也可能为正。但训练时的 prefix 已经和 rollout 不同。模型得到的是“看似成功 trajectory”的梯度，却不是“真实成功 trajectory”的梯度。

#### 3.7 系统指标会被误导

TITO 问题还会污染训练监控指标。比如 `old_logprob`、`ref_logprob`、KL、entropy、clip fraction、response length、tool-call success rate 等指标，表面上都能正常计算，但它们对应的 token 序列可能已经不是 rollout 真实序列。

这会带来错误判断：

- KL 变大时，可能不是 policy 真的偏离 reference，而是训练侧 prefix 漂移
- clip fraction 异常时，可能不是学习率或 advantage 有问题，而是 ratio 的状态-动作对错了
- response length 变化时，可能混入了 retokenization 或 template 重渲染带来的 token 数变化
- tool-call 成功率和 loss 对不上时，可能是 tool call 原始 token 被结构化重写了

所以 TITO 检查应该在训练指标之前发生。否则系统会拿一批已经漂移的 token 继续计算高级指标，然后把底层数据契约问题误诊为算法或超参数问题。

### 4. 解决思路

- 在 rollout 生命周期内维护 append-only token buffer
- 训练样本直接消费 rollout 返回的 token IDs，而不是从文本或 message list 重建
- 把 prompt、response、tool result、harness message 都表示成 token segment，并保留 segment 边界和 mask
- 文本只用于展示、工具调用和环境交互，不作为训练序列的唯一来源
- 对每一轮 rollout 加入 TITO invariant 检查：`P_n + R_n == prefix(P_{n+1})`
- 固定 tokenizer、chat template 和 special token 配置，并把配置版本写入样本元数据
- 对截断、过滤、脱敏等后处理建立 token-aware 规则

### 5. 工程取舍

- token buffer 会让 rollout 数据结构更复杂，但能避免训练侧重复猜测上下文
- 保留完整 token 历史会增加存储和传输成本，需要和长上下文截断策略一起设计
- tool/harness 仍然需要文本接口，因此系统必须明确区分“交互文本”和“训练 token”
- TITO 检查越严格，越容易暴露模板和序列化问题，但也要求各组件共享清晰的数据契约

### 6. 总结

TBD

## 参考材料

- [No Token Left Behind: Demystifying Token-In-Token-Out in Miles](https://www.lmsys.org/blog/2026-05-13-no-token-left-behind/)
- [Qwen3-8B tokenizer_config.json](https://huggingface.co/Qwen/Qwen3-8B/resolve/main/tokenizer_config.json)
- [Qwen2.5-7B-Instruct tokenizer_config.json](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/resolve/main/tokenizer_config.json)
- [[verl-ray-ppo-trainer]]
