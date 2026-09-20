---
title: Pi 的上下文管理：它怎样决定模型下一次能看到什么
description: 从一次长程 coding agent 任务出发，结合 Pi 的会话树、上下文投影、自动压缩、分支摘要和扩展接口，解释上下文是怎样被保存、裁剪和重建的。
date: 2026-09-20
tags:
  - Agent
  - Context-Engineering
  - Coding-Agent
  - pi
---

# Pi 的上下文管理：它怎样决定模型下一次能看到什么

让 pi 改一个小函数时，几乎感觉不到上下文管理的存在。任务一长，情况就不同了：它读了十几个文件，跑了几轮测试，某个方案失败后换了方向；这时下一次请求既不能把所有输出原样塞回模型，也不能让模型忘掉已经验证过的约束。

Pi 把已经发生的事情存进 session，再根据当前分支、压缩记录和扩展规则，重新构造下一次请求。模型看到的是这次构造出来的上下文；session 文件的全文不会直接进入请求。

关于 Pi 的整体结构，可以先看[前一篇源码介绍](./pi-coding-agent-mechanisms)。

## 先跟着一条消息走

从 session 文件到模型请求，中间经过几次明确的转换。Pi 大致经过这条路径：

```mermaid
flowchart LR
    J[Session JSONL] --> P[当前 leaf 的父链]
    P --> C[压缩或分支摘要]
    C --> A[AgentMessage]
    A --> H[transformContext]
    H --> L[convertToLlm]
    L --> M[Provider transcript]
    M --> R[模型请求]
```

Session 是 JSONL 文件，每条记录都有自己的类型和父节点。当前 leaf 决定从哪条路径回溯；回溯得到的是 Pi 内部的 `AgentMessage[]`，其中可以有用户消息、assistant 的文本和 tool call、tool result，以及扩展注入的消息。真正发给 provider 前，Agent loop 先运行 `transformContext`，再用 `convertToLlm` 转成 provider 能理解的消息。相关顺序在 [`agent-loop.ts`](https://github.com/earendil-works/pi/blob/71dca871bc80b6bc97be37f0ca3189399d651fff/packages/agent/src/agent-loop.ts)，消息转换的约束见 [`packages/agent/README.md`](https://github.com/earendil-works/pi/blob/71dca871bc80b6bc97be37f0ca3189399d651fff/packages/agent/README.md)。

这层转换给扩展留下了一个很有用的插入口：它可以改变本次请求的消息，而不必改写原来的 session。再往后，system prompt 和工具声明会被编码进 transcript 的 system message；“Provider transcript”是请求的概念视图，不是 session 里另存的一份对象。

**保存下来的历史，和模型这次看见的历史，本来就是两件事。**

## 规则文件和运行结果分开进入上下文

Pi 启动时会加载项目规则和能力说明。全局 `~/.pi/agent/` 下的文件先被读取，然后从当前工作目录逐级向上查找父目录中的 `AGENTS.md` 或 `CLAUDE.md`；某一级目录里的 `AGENTS.override.md` 会替代该目录的普通文件。[官方使用文档](https://pi.dev/docs/latest/usage#context-files)还列出了 `.pi/SYSTEM.md` 和 `APPEND_SYSTEM.md`，前者替换默认 system prompt，后者追加内容。

Skills 采用渐进披露。系统提示里只放名称、描述和路径，模型判断某个 skill 与任务相关后，再用 `read` 或 `bash` 打开 `SKILL.md`。完整说明不会一开始就占据上下文，但是否正确选中并读取 skill，仍然取决于模型。[Skills 文档](https://pi.dev/docs/latest/skills)

运行中的信息则来自消息本身。一个 assistant 消息可以同时包含文字、thinking 和多个 tool call；工具完成后，用 `toolCallId` 关联的 `toolResult` 会追加回来。工具调用和结果必须成对保留，后面的压缩切点也要遵守这个关系。[Session File Format](https://pi.dev/docs/latest/session-format#agentmessage-union)

扩展状态还有另一种保存方式。`CustomEntry` 只记录扩展自己的状态，不进入模型上下文；`CustomMessageEntry` 才会作为消息投影给模型。索引、UI 状态和遥测数据可以放在前者里，不必为了持久化它们而增加 prompt。[`session-manager.ts`](https://github.com/earendil-works/pi/blob/71dca871bc80b6bc97be37f0ca3189399d651fff/packages/coding-agent/src/core/session-manager.ts)

## session 是一棵树

Pi 把 session 保存在 `~/.pi/agent/sessions/` 下的 JSONL 文件中。除 header 外，记录通过 `id` 和 `parentId` 连成树，同一个文件里可以留下多次探索。`/tree` 改变当前 leaf，`/fork` 从某个节点另开一个 session 文件，`/clone` 复制当前活动分支。[Sessions 文档](https://pi.dev/docs/latest/sessions)

```text
任务：重构缓存层
├── 方案 A → 修改 → 测试失败
└── 回到共同祖先 → 方案 B → 实现
```

回到共同祖先只改变对话历史，磁盘里的代码仍保持现状，文件状态要交给 Git、容器快照或其他工具。Session 树还会保存一些不进 LLM context 的条目，`buildSessionContext` 再从活动路径投影出模型可见的消息。

换方案时，旧路径可以回看，新方案从共同祖先继续；如果需要，Pi 还可以把离开分支的工作总结后带到新分支。

## 压缩做了什么

同一条路径不断追加消息，迟早会接近模型的上下文上限。Pi 当前的默认设置是：

```json
{
  "compaction": {
    "enabled": true,
    "reserveTokens": 16384,
    "keepRecentTokens": 20000
  }
}
```

触发条件是：

```text
contextTokens > contextWindow - reserveTokens
```

`reserveTokens` 给即将生成的回复预留空间；压缩不会每次都从历史里删除固定数量的 token。工具结果追加后、开始下一次 assistant 回复前，Pi 会检查这个条件；新 prompt 前和底层 agent run 结束后也会检查。如果工具批次已经结束且没有排队消息，中间的检查可以跳过。`/compact [instructions]` 则允许用户手动压缩，并告诉摘要模型应该关注什么。[Compaction 文档](https://pi.dev/docs/latest/compaction#when-it-triggers)

压缩时，Pi 从最新消息向前估算 token，找一个切点，保留大约 `keepRecentTokens` 的近期内容。切点前的消息交给一次单独的 LLM 请求生成摘要，摘要和 `firstKeptEntryId` 一起写成 `CompactionEntry`。下一次请求重建为：

```text
system prompt
+ 早期历史的摘要
+ firstKeptEntryId 之后的近期消息
```

假设 Agent 刚读完二十个测试文件，最后一次测试还返回了一大段日志。切点前的旧读取结果会进入摘要，最近的测试调用和结果仍按原消息保留；下一次请求因此还能接着最近的失败继续查，而不用携带二十份完整文件内容。

旧 JSONL 记录没有被删掉。改变的是后续请求使用的投影；这也是为什么 session 还能被审计、重新导航，甚至再次压缩。[`compaction.ts`](https://github.com/earendil-works/pi/blob/71dca871bc80b6bc97be37f0ca3189399d651fff/packages/coding-agent/src/core/compaction/compaction.ts)

此外，`CompactionEntry` 还会保存压缩边界处的 system prompt 和工具声明 checkpoint。重建上下文时，旧范围里的 system entries 会被 checkpoint 替代，避免不同时间的 prompt patch 和工具声明重复出现。[`session-manager.ts`](https://github.com/earendil-works/pi/blob/71dca871bc80b6bc97be37f0ca3189399d651fff/packages/coding-agent/src/core/session-manager.ts)

Pi 不会把切点放在 `toolResult` 上，因为工具调用和结果要保持配对。通常它也尽量在 turn 边界切；如果一个 turn 自己就比 `keepRecentTokens` 还长，切点可能落在 assistant 消息中间，这时会标记为 `isSplitTurn`，再单独处理被切开的 turn 前缀。[切点实现](https://raw.githubusercontent.com/earendil-works/pi/71dca871bc80b6bc97be37f0ca3189399d651fff/packages/coding-agent/src/core/compaction/compaction.ts)

摘要提示要求保留任务目标、约束、已完成和未完成的工作、关键决定、下一步以及继续任务所需的上下文，并尽量保留精确的文件路径、函数名和错误信息。工具结果在序列化给摘要模型时默认截到约 2,000 个字符，过长部分用截断标记替代。[摘要格式与序列化](https://pi.dev/docs/latest/compaction#summary-format)

token 统计也不是每次重新 tokenize 全部历史。若已有有效的 assistant usage，Pi 以那份统计为基线，再估算之后新增的消息；没有 usage 时，则用字符数近似。文本、thinking、tool call 参数和 bash 输出都走估算，图片使用固定字符预算。这个办法足以决定何时压缩，却不是精确的 tokenizer 或计费结果。

摘要本身也要花 token。当前源码把摘要请求的最大输出限制为 `min(floor(0.8 * reserveTokens), model.maxTokens)`，并关闭一次性摘要请求的 prompt-cache 写入；`CompactionEntry.usage` 会记录摘要调用的用量。[`generateSummaryWithUsage`](https://raw.githubusercontent.com/earendil-works/pi/71dca871bc80b6bc97be37f0ca3189399d651fff/packages/coding-agent/src/core/compaction/compaction.ts)

如果 provider 返回 context overflow，或者出现可恢复的 `length` 截断，Pi 不会把相同输入直接重发。对于失败或不完整的 assistant 响应，它会从 Agent 当前状态移除该响应，先压缩，再最多重试一次；如果已经拿到可用但超出窗口的响应，则只压缩并继续。这个恢复路径和普通 provider retry 不是一回事。[`agent-session.ts`](https://github.com/earendil-works/pi/blob/71dca871bc80b6bc97be37f0ca3189399d651fff/packages/coding-agent/src/core/agent-session.ts)

## 换分支时，摘要会跟过去

自动压缩处理的是“同一路径太长”。`/tree` 切到另一条路径时，Pi 会找到旧 leaf 和目标 leaf 的共同祖先，收集旧分支的 entries，在 `contextWindow - branchSummary.reserveTokens` 的预算内优先保留较新的内容，然后生成 `BranchSummaryEntry` 挂到目标分支。默认 `branchSummary.reserveTokens` 是 16,384，摘要输出最多 4,096 token，`skipPrompt` 默认关闭，所以交互式切换会询问是否生成摘要。[Branch Summarization](https://pi.dev/docs/latest/compaction#branch-summarization)

```text
             ┌─ B ─ C ─ D（离开的分支）
A ───────────┤
             └─ E ─ F（目标分支）

切换后： A ─ E ─ F ─ [B、C、D 的摘要]
```

新分支得到的是旧方案的工作结论，原始工具输出不会全部带过去。摘要还会带上已读取、已修改的文件记录，便于模型知道哪些文件已经看过、改过。至于代码目录本身有没有回到旧状态，仍由 Git 等工具决定。

## 用户输入什么时候生效

Pi 的上下文还有一个时间问题：用户现在输入的内容，究竟何时送给模型。

按 Enter 是 steering。它会排队，等当前 assistant 这批 tool calls 全部完成后，在下一次 LLM 调用前送达。Alt+Enter 是 follow-up，要等当前 run 没有工具调用和 steering、准备停下时才送达；Escape 会中止并把队列内容放回编辑器。[Message Queue](https://pi.dev/docs/latest/usage#message-queue)

所以，发现 Agent 正在错误方向上搜索时，steering 适合马上补充约束；想等它完成后再要求补文档或跑测试，应该用 follow-up。当前代码执行完一批 tool calls 后才注入 steering，旧文档里“跳过剩余工具”的说法不再适用。[`agent-loop.ts`](https://raw.githubusercontent.com/earendil-works/pi/71dca871bc80b6bc97be37f0ca3189399d651fff/packages/agent/src/agent-loop.ts)

`steeringMode` 和 `followUpMode` 还可以配置为一次送一条，或一次送完所有排队消息。这个机制解决的是消息进入上下文的时机，而不是上下文窗口的大小。

## 扩展能改哪一层

Pi 通过扩展接口开放这些位置。`before_agent_start` 可以按本轮 prompt 调整 system prompt 或注入消息；`context` 在每次 LLM 调用前收到一份可安全修改的消息深拷贝；`session_before_compact` 可以取消压缩或提供自定义摘要；`session_before_tree` 可以接管分支摘要；`tool_result` 可以在结果进入后续上下文前做处理。[Extensions 文档](https://pi.dev/docs/latest/extensions)

还有一个更靠后的 `before_provider_request`。它发生在 provider payload 已经生成、请求尚未发出时，可以查看或替换最终 payload，但不会改变 session，也不会反映到 `ctx.getSystemPrompt()`。调试时要分清这三个东西：session 里保存的历史、`context` hook 处理过的消息，以及最后序列化出去的请求。[before_provider_request](https://pi.dev/docs/latest/extensions#before-provider-request)

这种自由度适合做检索、工具结果裁剪和自定义压缩，也会把策略责任交给使用者。扩展运行在 Pi 进程权限内，多个扩展的执行顺序会影响最终 prompt；简单地按消息条数删掉旧内容，还可能拆散 tool call 和 tool result。Pi 的默认压缩保守地处理了这些关系，自己的扩展也应该如此。

## 长任务里要注意什么

长期可靠的事实不要只放在摘要里。接口约束、迁移决定、测试命令和已经确认的错误原因，应该写进代码、README、设计文档或测试。摘要用来接上进度，不能当数据库。

`reserveTokens` 和 `keepRecentTokens` 可以在全局 `~/.pi/agent/settings.json` 设置，再用项目 `.pi/settings.json` 覆盖；`compaction.modelOverrides` 可以针对精确的 `provider/modelId` 调整。大上下文模型可以预留更多回复空间，小上下文模型则需要更早压缩。设置后要观察任务表现、摘要成本和压缩后的约束保留情况。

```json
{
  "compaction": {
    "enabled": true,
    "reserveTokens": 16384,
    "keepRecentTokens": 20000,
    "modelOverrides": {
      "provider/model-id": {
        "reserveTokens": 32768,
        "keepRecentTokens": 30000
      }
    }
  }
}
```

工具输出最好先控制体积。读取大文件时分页，测试命令保留关键尾部，把完整日志落盘后按需读取。Pi 自己也会截断工具输出；摘要序列化时，单个 tool result 默认只保留约 2,000 个字符。完整日志留在文件里，并不代表模型已经记住了完整日志，关键结论仍应在压缩前写下来，或者在后续重新读取。

可以把它拆成三个分工：session 留下发生过的事实，当前 leaf 和压缩边界决定沿哪条路径读取，`transformContext` 和扩展决定最终发给模型的消息。同一份 session 可以被不同方式重新组织，但风险仍在：摘要会漏信息，token 统计只是近似，扩展可能改写请求，会话树也不会回滚工作目录。

接口约束和测试结论应该落到代码、文档或测试里；当前任务的临时信息才适合留在 prompt；需要跨阶段延续的进度，则交给结构化摘要。上下文窗口变大只能推迟问题，下一次调用能不能重建出正确的历史，才决定长任务能不能继续。

## 资料与版本

本文参考了[知乎文章《万字长文谈 pi agent context 管理》](https://zhuanlan.zhihu.com/p/2077441274156790849)，正文沿默认 CLI 的 `AgentSession → Agent loop` 路径展开，不讨论仓库里另一套 harness API。源码固定在 [`71dca871bc80`](https://github.com/earendil-works/pi/commit/71dca871bc80b6bc97be37f0ca3189399d651fff)，按该提交于 **2026-09-20** 核对。

## 参考资料

- [Pi 官方文档：Compaction & Branch Summarization](https://pi.dev/docs/latest/compaction)
- [Pi 官方文档：Session File Format](https://pi.dev/docs/latest/session-format)
- [Pi 官方文档：Using Pi](https://pi.dev/docs/latest/usage)
- [Pi 官方文档：Extensions](https://pi.dev/docs/latest/extensions)
- [自动压缩源码：`compaction.ts`](https://github.com/earendil-works/pi/blob/71dca871bc80b6bc97be37f0ca3189399d651fff/packages/coding-agent/src/core/compaction/compaction.ts)
- [分支摘要源码：`branch-summarization.ts`](https://github.com/earendil-works/pi/blob/71dca871bc80b6bc97be37f0ca3189399d651fff/packages/coding-agent/src/core/compaction/branch-summarization.ts)
- [Agent loop 源码：`agent-loop.ts`](https://github.com/earendil-works/pi/blob/71dca871bc80b6bc97be37f0ca3189399d651fff/packages/agent/src/agent-loop.ts)
