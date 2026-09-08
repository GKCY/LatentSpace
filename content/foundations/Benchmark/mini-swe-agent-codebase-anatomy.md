---
title: 把 Agent 缩成一个循环：mini-swe-agent 源码解剖
description: 从固定提交出发，拆解 mini-swe-agent 的消息循环、模型与环境协议、trajectory、SWE-bench runner，以及可复现实验边界。
---

# 把 Agent 缩成一个循环：mini-swe-agent 源码解剖

> 本文基于截至分析日的最新稳定版 mini-swe-agent `v2.4.6`，固定提交
> `a83fcae82d2a08f0ee0c688f9d137b3566c097f8`（发布于 2026-07-23，分析日期：2026-09-01）。
> mini-swe-agent v1 与 v2 在工具调用、trajectory 格式和组件职责上差异很大；除非特别说明，本文所有结论都只针对这个固定版本。
> SWE-bench evaluator 的交接示例固定到同一分析日的提交 `334882dd1f2664cc55c1abfe9de4884af023c0c0`。

mini-swe-agent 最吸引人的标签是“约 100 行的 coding agent”。这个说法没有错，但很容易让人得到一个错误印象：仿佛软件工程 agent 的全部问题，都被 100 行 Python 解决了。

真正读完代码后，我认为它更准确的定义是：

> mini-swe-agent 把 agent 内核压缩成一个 append-only 的消息循环。Model 承接模型与动作协议，Environment 承接命令执行，runner 与外部 harness 再负责批量调度、隔离、产物和评分。

它的价值不在于“功能少到什么都没有”，而在于选择了一条很窄、很稳定的中间接口：模型只需要提出 shell 命令，环境只需要返回合并后的命令输出、return code 与异常信息，agent 只负责把二者交替追加到历史中。

这篇文章沿一条完整调用链向下拆：prompt 如何生成，模型响应如何变成 bash action，命令如何落地，什么时候退出，trajectory 保存了什么，以及内置 SWE-bench runner 如何把这个小循环扩展成一条批量推理流水线。最后再讨论它作为 benchmark harness 的优点、隐含变量和安全边界。配置细节与复现实验 manifest 放在附录，避免打断核心路径。

## 阅读地图：先划清五层边界

如果不先分层，很容易把外部 benchmark 或 runner 的能力算到 mini-swe-agent 内核头上。

| 组件                  | 负责什么                                                       | 不负责什么                        |
| --------------------- | -------------------------------------------------------------- | --------------------------------- |
| mini-swe-agent core   | 模型—shell 循环、history、预算、trajectory                     | 题目真实性、隐藏测试、最终 reward |
| 内置 SWE-bench runner | 数据加载、镜像启动、并发、prediction 收集                      | SWE-bench 测试与评分              |
| SWE-bench harness     | 应用 model patch、运行 task tests、汇总 resolved               | 生成 patch 的 agent 策略          |
| DeepSWE task package  | prompt、固定 repo、隐藏 verifier、评分合同                     | 通用 agent 控制循环               |
| Pier                  | 发现 DeepSWE task、驱动 agent、sandbox 生命周期、artifact 搬运 | mini-swe-agent 内部的模型决策     |

本站的 [[deep-swe-task-and-benchmark-pipeline|DeepSWE 任务与评测流程]] 分析的是另一条 artifact 路径：官方 leaderboard 由 Pier 驱动 mini-swe-agent，再由 task 的独立 verifier 评分。90 分钟限制、no-network、separate verifier，以及 committed patch 收集，都来自 DeepSWE task/Pier 合同，不是 `DefaultAgent` 的通用默认行为。

本文第 1–3 节聚焦 core，第 4 节再进入仓库内置 SWE-bench runner；外部 evaluator 只在 prediction 生成后接手。

## 1. “100 行”究竟省略了什么

### 1.1 小的是 Agent 内核，不是整个系统

在本文固定版本中，`src/minisweagent/agents/default.py` 有 190 个物理行，`DefaultAgent` 类从第 38 行延伸到文件末尾。若忽略 import、配置声明、docstring、序列化与空行，真正负责推理—执行循环的主体确实只有约百行；核心的 `step` 甚至只有一句：

```python
return self.execute_actions(self.query())
```

但可安装、可运行、可批量评测的 mini-swe-agent 远不止这个类。代码大致分成五层：

| 层               | 主要职责                                            | 代表文件                                          |
| ---------------- | --------------------------------------------------- | ------------------------------------------------- |
| Agent            | 保存消息、检查预算、串起 query 与 execute、处理退出 | `agents/default.py`                               |
| Model            | 调 API、解析 action、格式化 observation、计算 cost  | `models/litellm_model.py`、`models/utils/`        |
| Environment      | 在本机或容器执行命令，返回结构化结果                | `environments/local.py`、`environments/docker.py` |
| Config / CLI     | 合并 YAML 与命令行覆盖，实例化三类组件              | `config/`、`run/mini.py`                          |
| Benchmark runner | 加载数据集、并发跑实例、收集 patch 与 trajectory    | `run/benchmarks/swebench.py`、`programbench.py`   |

所以，“100 行”描述的是控制循环的最小闭环，不是总体代码量，更不包含：

- LiteLLM、OpenRouter、Portkey 等 provider 适配；
- Docker、Singularity、Bubblewrap、SWE-ReX、Contree 等环境后端；
- prompt、输出裁剪、重试、成本计算与 cache control；
- 数据集、容器镜像、隐藏测试和最终 grader；
- DeepSWE 场景中的 Pier 调度与 artifact 搬运。

这并不削弱“minimal”的意义。项目的关键取舍不是把所有复杂度都塞进 agent class，而是让核心类只拥有它必须拥有的状态。

**它删掉了哪些传统 agent 机制。**

默认内核没有这些常见模块：

- planner / executor 两阶段状态机；
- 文件搜索、补丁编辑、测试等多套专用 tool schema；
- 常驻 PTY 或可变 shell session；
- 自动摘要、消息裁剪或复杂 history processor；
- task-specific memory、向量检索或知识库；
- 内置 verifier 与 reward 计算。

模型面对的唯一动作语义是 `bash(command: string)`。不过“只有 bash”不等于能力弱：shell 本身是一种元工具，能组合文件读取、搜索、编辑器、编译器、测试框架、Git、包管理器和脚本语言。mini-swe-agent 省掉的是每一种能力各自的 agent-side wrapper，把组合责任交给模型与运行环境。

### 1.2 三个协议构成最小架构

v2 在 [`minisweagent/__init__.py`](https://github.com/SWE-agent/mini-swe-agent/blob/a83fcae82d2a08f0ee0c688f9d137b3566c097f8/src/minisweagent/__init__.py#L39-L80) 中用 Python `Protocol` 描述了三个角色。它们不要求继承共同基类，只要求对象提供约定的方法：静态检查按结构类型判断，运行时则依赖鸭子类型约定。

| 协议          | 协议定义的核心方法                                                                         | 语义                                                            |
| ------------- | ------------------------------------------------------------------------------------------ | --------------------------------------------------------------- |
| `Model`       | `query`、`format_message`、`format_observation_messages`、`get_template_vars`、`serialize` | 把 provider 消息变成 action，再把环境结果变回 provider 可读消息 |
| `Environment` | `execute`、`get_template_vars`、`serialize`                                                | 执行一个 action，并暴露配置/平台变量                            |
| `Agent`       | `run`、`save`                                                                              | 驱动任务并保存 trajectory                                       |

这三个接口形成一条窄腰：

```text
Task ──> Agent.messages ──> Model.query
              ▲                  │
              │                  ▼ extra.actions
              │             [{command: ...}]
              │                  │
              │                  ▼
              │         Environment.execute
              │                  │ output / returncode / exception_info
              │                  ▼
              └── Model.format_observation_messages
```

关键变化发生在 v2：action parsing 和 observation formatting 从 Agent 移到了 Model。原因很实际：原生 tool call、Markdown 代码块和 Responses API 的消息结构不同，但这些差异都属于“怎样和模型说话”，不应该污染控制循环。

**扩展依赖动态工厂，而不是庞大注册框架。**

`get_model`、`get_environment` 和 `get_agent` 各自维护一张短名称到完整 import path 的映射。例如 `docker` 会解析到 `DockerEnvironment`。若配置里直接给完整类路径，工厂会动态 import，因此外部项目可以实现自己的类，而无需先向 mini-swe-agent 上游注册。

这让扩展方式很轻：

1. 自定义 Model 负责产生 `extra.actions`；
2. 自定义 Environment 接受 action dict；
3. 自定义 Agent 可以覆写 `query`、`step` 或 `execute_actions`；
4. YAML 中把 `*_class` 指向新类。

代价是接口合同主要依赖类型提示和运行时约定。比如自定义 Model 若返回没有 `extra.actions` 的消息，默认 Agent 不会主动报错，只会把空 outputs 交回该 Model 的 observation formatter；内置 formatter 此时不产生 observation，循环会直接进入下一轮。内置 Model 会用 `FormatError` 封住这个缺口，但协议本身没有强制它。内置 runner 还会直接读取 `model.config.model_name` 等字段，因此三个 `Protocol` 也不是全部运行时合同。

## 2. 一次推理—执行循环如何流动

### 2.1 `DefaultAgent.run`：整个状态机就是消息列表

一次 `run(task)` 开始时，Agent 把 `task` 放入模板变量，清空 `messages`，渲染并追加两条初始消息：

```text
messages[0] = system_template
messages[1] = instance_template(task=...)
```

此后反复执行 `step()`。简化后的控制流如下：

```text
render system + instance
          │
          ▼
┌── check step / cost / wall-time limit
│         │
│         ▼
│     Model.query(messages)
│         │ assistant message + actions
│         ▼
│     append assistant message
│         │
│         ▼
│     Environment.execute(action) × N
│         │ outputs
│         ▼
│     Model formats observations
│         │
│         ▼
│     append observation messages
│         │
│     save trajectory checkpoint
│         │
└─────────┴── if last role != exit, next step
```

#### 2.1.1 按固定源码合同展开的一条最小调用链

> 下面是依据固定提交构造的合同级示例，用来对齐消息角色、异常与保存时机；它不是某次真实 API 调用的录制。为突出控制流，省略了模板正文和 provider 专属字段。

```text
run(task)
  1. 渲染 system_template 与 instance_template；
     messages += system message, user message。

第 1 轮：执行普通命令
  2. Model.query(messages)
     -> assistant(tool_call=bash("rg -n 'bug' src"),
                  extra.actions=[{command: ..., tool_call_id: "call_1"}])
     -> Agent 先把 assistant 追加到 messages。
  3. Environment.execute(action)
     -> {output: "src/a.py:42: ...\n", returncode: 0, exception_info: ""}
  4. Model.format_observation_messages(...)
     -> tool(tool_call_id="call_1",
             content="<returncode>0</returncode> ...",
             extra.raw_output="src/a.py:42: ...\n")
     -> Agent 把 tool observation 追加到 messages。
  5. 本轮 finally 调用 save(output_path)；若配置了路径则覆盖写入
     中间 checkpoint。最后一条消息不是 exit，因此进入下一轮。

第 2 轮：提交结果
  6. Model.query(messages 去掉本地 extra 后的视图)
     -> assistant(tool_call=bash(
          "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"))
     -> Agent 先把 assistant 追加到 messages。
  7. Environment 执行命令；若 stdout/stderr 合并输出经 lstrip 后的首行等于 sentinel，
     且 returncode == 0，则以其余合并输出为 submission 并抛出 Submitted。
  8. run() 捕获 Submitted，把其携带的
     {role: "exit", content: "<patch>",
      extra: {exit_status: "Submitted", submission: "<patch>"}}
     追加到 messages；这条提交命令不会再产生 tool observation。
  9. finally 再次调用 save(output_path)：serialize() 汇总状态，
     配置了路径时覆盖写入最终 trajectory JSON。
 10. 最后一条消息角色为 exit，循环结束；
     run() 返回该 exit 消息的 extra。
```

状态只有几项：

| 字段                          | 用途                                                          |
| ----------------------------- | ------------------------------------------------------------- |
| `messages`                    | 唯一的对话/执行历史                                           |
| `cost`                        | 合法响应的成本，以及从 `FormatError` 异常路径补记的已计费成本 |
| `n_calls`                     | 发起过的逻辑模型轮次                                          |
| `n_consecutive_format_errors` | 连续格式错误计数；任一正常 step 后归零                        |
| `_start_time`                 | Agent 对象创建时刻，用于 wall-time                            |
| `extra_template_vars`         | task 与 runner 注入的额外上下文                               |

`step` 不做规划，也不理解命令内容。它只组合两件事：

```text
message = query()
observations = execute_actions(message)
```

#### 2.1.2 一轮消息到底怎样增长

以默认原生 tool-call 模式、每轮一个命令为例：

| 时刻            | 新增角色         | 主要内容                             |
| --------------- | ---------------- | ------------------------------------ |
| 初始化          | `system`         | 身份、工具使用规则                   |
| 初始化          | `user`           | issue/task、工作流、提交规则         |
| 第 1 轮 query   | `assistant`      | 推理文本、`bash` tool call           |
| 第 1 轮 execute | `tool`           | return code 与裁剪后的 stdout/stderr |
| 第 2 轮 query   | `assistant`      | 基于全部历史继续行动                 |
| 第 2 轮 execute | `tool` 或 `exit` | 新 observation，或提交结果           |

没有单独的 scratchpad store，也没有从环境里反向重建状态。只要一个事实没有写进消息、文件系统或模型自己的输出，它就不会神奇地留在 Agent 内。

#### 2.1.3 `run` 可以复用，但计数器不会重置

`run()` 会清空 `messages`，但不会重置 `cost`、`n_calls`、`n_consecutive_format_errors` 或 `_start_time`，也不会清空上一次留下的额外模板变量；本次传入的同名变量只会覆盖旧值。

因此在同一个 Agent 对象上连续调用两次 `run`，第二个任务会继承第一个任务的预算消耗，并继续从对象构造时计算 wall time。若上一次以 `RepeatedFormatError` 结束，连续错误计数也会被带入下一次调用。

内置 runner 每个实例都创建新 Agent，所以正常 benchmark 不受影响；若使用 Python bindings 做多任务服务，则应每题新建实例，或显式设计自己的 reset 语义。

### 2.2 Model 层：v2 默认使用 tool call，但动作仍只有 bash

默认 `LitellmModel` 向 provider 注册一个极小的 function schema。完整的 `tools` 元素是：

```json
{
  "type": "function",
  "function": {
    "name": "bash",
    "description": "Execute a bash command",
    "parameters": {
      "type": "object",
      "properties": {
        "command": {
          "type": "string",
          "description": "The bash command to execute"
        }
      },
      "required": ["command"]
    }
  }
}
```

一次 `Model.query(messages)` 做六件事：

1. 从每条消息中移除仅供本地记录的 `extra`；
2. 必要时重排 Anthropic thinking block，并设置 cache-control 标记；
3. 经 LiteLLM 发起 completion 请求；
4. 按模型名计算本次响应成本，并更新进程级全局统计；
5. 解析 tool calls；若失败，把成本和原始响应挂到 `FormatError`；
6. 把 provider 原生 assistant message 转成 dict，并把 actions 放入 `message.extra`。

Agent 不关心 provider 返回的是哪一种原生格式，只看：

```python
message["extra"]["actions"]
```

每个 action 至少包含 `command`；原生 tool call 还带 `tool_call_id`，这样命令输出能正确配对到对应调用。

#### 2.2.1 “不需要 tool calling”与“v2 默认 tool calling”并不矛盾

项目的设计主张是 agent 只需要 bash，不要求模型原生支持一整套专用工具。v1 会用正则从普通的 `bash` Markdown 代码块提取动作；v2 为避免与 prompt 中的命令示例冲突，把默认文本标记改成：

````text
```mswea_bash_command
...
```
````

v2 的 `*_textbased` Model 仍可从这种代码块提取命令；v2 默认配置则优先用 provider 的原生 function/tool calling。两种 v2 路径共享同一个 Agent 与 Environment，只是 Model 层的编码不同。

这也是 v2 最重要的架构调整之一：工具协议的载体可以换，动作空间仍是 shell。

#### 2.2.2 一轮可以有多个 tool call，但执行不是并行的

SWE-bench 默认模型参数允许 `parallel_tool_calls: true`，解析器也接受多个合法 bash call。然而 `DefaultAgent.execute_actions` 使用普通 list comprehension：

```text
for action in actions:
    env.execute(action)
```

若所有 action 都正常返回，多个动作会按 actions 列表顺序串行执行，全部结束后才一次性把 observations 加入消息。这里的 “parallel” 只是允许模型在一次响应里提出多个独立 tool call，不能理解为本地命令并发。

这个细节还会影响语义：第二个命令能够看到第一个命令留下的文件修改，但模型要等这一批全部执行完，才能看到任何一个结果。

终止 action 是例外。若中间某个命令触发 `Submitted` 或自定义 Environment 抛出其他 `InterruptAgentFlow`，list comprehension 会立刻短路，后续 action 不再执行；此前已完成 action 的 outputs 也还没进入 formatter。`DefaultAgent` 此时只追加异常携带的 exit 消息，因此同一批中较早命令的 observation 不会进入 trajectory。`InteractiveAgent` 为此覆写了执行路径，用 `finally` 尽量保留 partial outputs。

#### 2.2.3 格式错误也进入同一消息历史

原生 tool-call parser 遇到以下情况会抛出 `FormatError`：

- 没有 tool call；
- tool 名不是 `bash`；
- arguments 不是合法 JSON；
- 缺少 `command`。

这里还有一道刻意保持很薄的边界：parser 只检查 arguments 是 dict 且存在 `command`，不在 Python 侧再次验证值的类型。`{"command": null}` 或数字可能绕过 schema 约束进入 Environment，随后才变成命令执行异常 observation。换句话说，`command: string` 是 provider schema 合同，不是 parser 自己完整执行的运行时校验。

异常携带一条 `role=user` 的格式纠正消息。`run` 捕获它、追加到 history，然后进入下一轮，让模型自我修复；任一正常完成的 step 会把连续错误计数归零。默认连续 3 次格式错误时，Agent 再追加 `role=exit`、`exit_status=RepeatedFormatError` 并终止；把 `max_consecutive_format_errors` 设为 0 才会禁用这道门槛。

Model 还把 `finish_reason` 交给纠错模板。SWE-bench 配置会把“没生成 tool call”和“输出达到 token limit 后被截断”区分开，后者提示模型缩短推理并尽快给出命令。文本模式则要求正则恰好匹配一个 action，多一个或少一个都视为格式错误。

### 2.3 Environment 层：没有常驻 shell，不等于环境无状态

Environment 接收 action dict，返回统一结构：

```json
{
  "output": "stdout and stderr",
  "returncode": 0,
  "exception_info": ""
}
```

命令失败、timeout 或命令启动异常通常不会终止 Agent。环境会把它们编码成 observation，下一轮由模型决定如何恢复。

#### 2.3.1 LocalEnvironment

LocalEnvironment 对每个 action 新建一次 `subprocess.Popen`，再用 `communicate` 等待结束：

- `shell=True`；
- stdout 与 stderr 合并；
- UTF-8 解码，非法字节替换；
- 默认命令超时 30 秒；
- 继承宿主全部环境变量，再由配置值覆盖；
- 每个 action 新建一个 shell 进程；
- POSIX 上为它创建新 session，timeout 时向整个 process group 发送 `SIGKILL`，避免遗留子进程。

严格来说，tool 名虽叫 `bash`，LocalEnvironment 并没有显式传入 Bash；POSIX 上 `shell=True` 默认固定经 `/bin/sh` 执行，而不是读取用户的 `$SHELL`。`/bin/sh` 的具体实现可能是 dash、Bash 或其他兼容 shell。DockerEnvironment 才通过可配置的 `interpreter` 明确使用 `bash -lc`，SWE-bench 配置进一步改成 `bash -c`。若 prompt 依赖 Bash-only 语法，本机后端的可移植性需要单独验证。

#### 2.3.2 DockerEnvironment

DockerEnvironment 先启动一个长寿命容器；`container_timeout` 默认是 2 小时：

```text
docker run -d ... <image> sleep <container_timeout>
```

每个 action 再执行一次：

```text
docker exec -w <cwd> ... <container> bash -lc <command>
```

因此它复用的是同一个容器文件系统，而不是同一个 shell session。

| 状态                            | 跨 action 是否保留 | 原因                             |
| ------------------------------- | ------------------ | -------------------------------- |
| 文件修改、Git working tree      | 是                 | 同一个宿主目录或容器文件系统     |
| 已安装到文件系统的依赖          | 是                 | 磁盘状态保留                     |
| `cd` 后的工作目录               | 否                 | 下次重新按配置 `cwd` 启动        |
| `export`、shell variable、alias | 否                 | 新 shell 进程                    |
| 后台进程与外部副作用            | 可能               | 环境不是纯函数，代码没有整体回滚 |
| 对话记忆                        | 由 `messages` 保留 | 不依赖 shell                     |

“无常驻 shell”减少了 prompt 状态与终端状态不一致、PTY 卡死、残留交互模式等问题，但不能把它误写成“每步完全隔离”。文件、进程、网络和系统服务仍可能形成跨步状态。

#### 2.3.3 timeout 是 observation；本机与容器的杀进程强度不同

Local 与 Docker 后端都把 `subprocess` 超时捕获成：

```text
returncode = -1
exception_info = "... timed out ..."
```

Agent 可以在下一轮改用更小的测试或检查部分输出。LocalEnvironment 在 POSIX 上显式杀掉整个 process group；DockerEnvironment 超时时，终止的是宿主侧 `docker exec` 客户端进程，源码没有额外展示对容器内整个进程树的强制清理。因此容器路径不应仅凭这个字段推断所有后代进程都已停止。

### 2.4 退出不是模型说“完成了”，而是一条可执行协议

默认环境只认一个 magic sentinel：

```bash
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
```

命令结束后，Environment 会：

1. 对合并输出做左侧空白清理并拆行；
2. 对第一行再做 `strip()`，检查结果是否等于 sentinel；
3. 检查命令 return code 是否为 0；
4. 把其余输出作为 `submission`；
5. 抛出携带 `role=exit` 消息的 `Submitted`。

这解释了为什么 prompt 反复强调提交命令的位置和组合方式。只在自然语言里写“任务完成”不会退出；sentinel 前若已有普通输出也不会退出；return code 非零同样不会退出。

在日常 `mini` CLI 中，submission 可以为空，只表示停止。在内置 SWE-bench prompt 中，最终命令是：

```bash
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt
```

于是 sentinel 后的合并输出正好就是交给评测器的 patch。Agent 核心完全不知道什么是 Git diff，它只搬运字符串。

**用异常表达控制流。**

所有预期的流程中断都继承 `InterruptAgentFlow`：

| 异常               | 是否终止                      | 写入 trajectory 的含义                                |
| ------------------ | ----------------------------- | ----------------------------------------------------- |
| `Submitted`        | 是                            | 正常提交，带 submission                               |
| `LimitsExceeded`   | 是                            | step 或 cost 预算耗尽                                 |
| `TimeExceeded`     | 是                            | agent wall-time 耗尽                                  |
| `FormatError`      | 前两次否，默认连续第 3 次终止 | 追加纠错消息；超限时再追加 `RepeatedFormatError` exit |
| `UserInterruption` | 通常否                        | 用户拒绝、补充任务或切换模式                          |

`run` 不需要同时传播 `done`、`status`、`submission` 与 partial observations；异常携带要追加的消息，最后只检查 `messages[-1].role == "exit"`。

循环内、也就是 `step()` 阶段的未知异常会走另一条路径：Agent 先追加包含异常类型、字符串与 traceback 的 exit 消息，保存 trajectory，然后重新抛出。外层 runner 因而既能拿到诊断记录，也不会把基础设施错误伪装成正常结束。初始模板渲染和初始 `format_message` 位于这个 `try` 之外；它们若失败，日常 `run()` 不会自行追加 exit 或执行该轮 checkpoint。

### 2.5 预算、超时与进程级限制的语义

为避免把不同层级的计数混在一起，本文后文统一使用下面四个术语：

| 术语              | 含义                                                      |
| ----------------- | --------------------------------------------------------- |
| step / Agent 轮次 | 默认 Agent 的一次 `query()` 加该响应中全部 action 的执行  |
| 逻辑模型调用      | `n_calls`；trajectory 序列化时写成 `api_calls`            |
| 底层传输尝试      | Model adapter 内一次真实 API/HTTP attempt；重试会产生多个 |
| action            | 一条交给 Environment 的命令；一个逻辑模型调用可以产生多条 |

`AgentConfig` 提供 step、cost 和 wall-time 三种实例级限制，Environment 另有每条 command timeout，Docker 还有容器寿命。进程还可以通过 `MSWEA_GLOBAL_COST_LIMIT` 与 `MSWEA_GLOBAL_CALL_LIMIT` 设置所有 Model 共享的全局门槛。

| 限制                       | 检查位置                     | v2.4.6 的实际语义                                      |
| -------------------------- | ---------------------------- | ------------------------------------------------------ |
| `step_limit`               | 发起下一轮模型请求前         | 最多允许 N 次逻辑 `Model.query`                        |
| `cost_limit`               | 发起下一轮模型请求前         | 已记录成本达到阈值后不再发下一次；最后一 call 可以越界 |
| `wall_time_limit_seconds`  | 发起下一轮模型请求前         | 只在轮次边界检查，当前模型/命令不会被它中途打断        |
| 进程级 cost / call limit   | 合法响应返回、解析 action 前 | 跨实例共享；触发时抛普通 `RuntimeError`                |
| Environment `timeout`      | 每条命令                     | 超时被转成 observation，通常继续                       |
| Docker `container_timeout` | 容器主进程                   | 到期后整个容器自然退出                                 |

这张表描述的是 `DefaultAgent`。日常 CLI 使用的 `InteractiveAgent` 在有 TTY 时会捕获 step/cost 的 `LimitsExceeded`，询问用户新的上限后继续；无人值守或 stdin 非交互时才按原状态退出。wall-time 无法靠提高 step/cost 解开，因此 `TimeExceeded` 始终直接结束。batch runner 使用 `DefaultAgent` 子类，不会弹出续费提示。

两个细节尤其值得做实验时记录。

第一，`n_calls` 在调用 Model 之前就加一。如果 provider 在内部重试十次后仍失败，Agent 记的是一个逻辑轮次，不是十个 HTTP 请求。因此 trajectory 中的 `api_calls` 更准确的名字其实是 agent turns。

第二，cost 是响应回来后才累计，所以上限不是预授权预算。假设当前花费 2.90 美元、限制 3 美元，下一次调用仍会发生，即使它把总额推到 3.40 美元。

#### 2.5.1 模型请求重试

LiteLLM Model 用 Tenacity 做指数退避，默认最多 10 次，等待从 4 秒增长并封顶 60 秒。认证失败、权限失败、模型不存在、参数不支持、context window exceeded 等异常直接中止；临时网络或服务错误通常重试。

失败的底层传输尝试不会作为独立消息进入 trajectory。于是“同一份 trajectory + 同一个 `api_calls`”背后可能有不同的实际请求数、延迟和 provider 侧计费，这也是严格复现实验时必须从网关日志补齐的变量。

#### 2.5.2 v2.4.6 专门补上了格式错误的计费与审计

Model 先计算响应成本并更新 `GLOBAL_MODEL_STATS`，随后才解析 actions。若解析失败，正常的 `Model.query → Agent.query` 返回链会被异常打断，因此最初版本很容易漏掉 per-agent cost。

v2.4.6 为这条异常路径建立了显式合同：

1. Model 捕获 action parser 抛出的 `FormatError`；
2. 把本次 `cost` 与完整 provider response 写进异常消息的 `extra`；
3. `DefaultAgent.run` 单独捕获 `FormatError`，从异常消息补记实例 cost；
4. 保存纠错消息和原始响应，供 trajectory 审计；
5. 连续 3 次解析失败后以 `RepeatedFormatError` 退出。

在正常进入 `FormatError` 路径时，这使“已计费但没形成合法 action”的响应同时受到实例 cost limit、连续错误上限和 trajectory 记录约束。v2.4.6 的 release note 也把这项修复列为该版本唯一的用户可见功能修复。

进程级全局限制是例外。`LitellmModel` 在解析 action 前先调用 [`GLOBAL_MODEL_STATS.add()`](https://github.com/SWE-agent/mini-swe-agent/blob/a83fcae82d2a08f0ee0c688f9d137b3566c097f8/src/minisweagent/models/litellm_model.py#L81-L98)；若 `MSWEA_GLOBAL_COST_LIMIT` 或 `MSWEA_GLOBAL_CALL_LIMIT` 在这里触发 `RuntimeError`，控制流还没进入 parser，实例 cost 与完整 provider response 也尚未按 `FormatError` 合同持久化。Agent 最终只会记录普通的 uncaught-exception exit。因此，实例预算和进程级预算必须作为两套不同机制报告。

## 3. 模型究竟看到了什么，轨迹又保存了什么

### 3.1 “线性历史”是语义事实，不是字节级事实

mini-swe-agent 没有 history processor。每一轮就是在 `messages` 尾部追加 assistant 和 tool/user observation，再把历史交给下一轮。这使 trajectory 很适合调试、监督微调和 RL rollout 分析：行动之前模型看到了什么，通常能沿列表直接重放。

不过“trajectory 就是下一轮 prompt”需要加三个限定：

1. 发 API 前会删除每条消息的 `extra`，所以原始命令输出、cost、timestamp 等本地元数据不会送回模型；
2. Anthropic 路径可能重排 thinking block，并在末条消息加 ephemeral cache-control；
3. Responses API 路径会把保存的 response object 展开成 output items。

因此它是 append-only 的语义历史，而不是把落盘 JSON 原封不动回传。

**没有上下文压缩。**

默认 Agent 不会自动摘要旧消息、丢弃早期 observation 或检索相关片段。优点是：

- 数据流透明；
- 不引入摘要器质量这一额外变量；
- trajectory 更容易用于训练；
- 不会因隐藏的 memory policy 改变 benchmark 行为。

缺点也直接：长任务最终可能触发 context-window error，而该错误在默认 Model 的 abort 列表中，不会重试。ProgramBench 这类 1000-step 上限任务尤其依赖模型上下文、prompt 纪律和命令输出控制。

### 3.2 Observation 模板决定模型看到的环境反馈

环境返回的是原始结果，真正放进模型上下文的内容由 Model 的 `observation_template` 决定。默认 CLI 与 SWE-bench 配置采用相同的长度策略：

```text
output < 10,000 chars  → 全量返回
output ≥ 10,000 chars  → 前 5,000 + 后 5,000 + 截断警告
```

同时附上 `returncode` 和可选 `exception_info`。这看似只是 token 优化，实际上会改变 agent 能力：

- traceback 的中段可能被裁掉；
- 大型测试列表只保留头尾；
- 模型必须学会用 `sed`、`head`、`tail` 或重定向主动缩小结果；
- 同一模型在不同 observation budget 下不是同一个系统。

模板把裁剪后的文本放进 `content`，又把完整原始输出放进 message `extra.raw_output`。所以 trajectory 通常同时服务两种消费者：模型只看到受控内容，研究者仍可分析完整命令输出。

ProgramBench runner 则覆写序列化，删除 `raw_output` 以避免 trajectory 过度膨胀；它不改变当轮模型已经看到的裁剪内容。这再次说明：trajectory policy 属于 runner/harness 配置，不是 `DefaultAgent` 不可变的性质。

### 3.3 Trajectory 保存了可审计执行证据

`DefaultAgent.serialize()` 生成的顶层结构可概括为：

```json
{
  "info": {
    "model_stats": { "instance_cost": 0.0, "api_calls": 0 },
    "config": {
      "agent": {},
      "model": {},
      "environment": {},
      "agent_type": "...",
      "model_type": "...",
      "environment_type": "..."
    },
    "mini_version": "2.4.6",
    "exit_status": "Submitted",
    "submission": "..."
  },
  "messages": [],
  "trajectory_format": "mini-swe-agent-1.1"
}
```

每次进入循环后的 `finally` 都调用 `save`。只要配置了 `output_path`，正常 step、格式错误、预算退出和 step 内未知异常都会尽量覆盖写入最新 checkpoint；初始化 system/user 消息时发生的异常不在这层保护中。

但这只是“持续落盘”，不是“断点恢复”：仓库没有从 trajectory 反序列化 Agent 状态并继续运行的默认路径。进程崩溃后可以审计已经发生的步骤，不能直接从下一步续跑。

还有一个实际安全问题：序列化会保存 model kwargs、environment config 和消息 `extra`。如果用户把 API key 直接写进 `model_kwargs` 或配置的 env map，它可能进入 trajectory。生产流水线应把凭证放在 provider 支持的 secret 注入层，并在公开轨迹前做字段审计。

## 4. 内置 SWE-bench runner 如何把小循环变成批量评测

mini-swe-agent 自带两个 SWE-bench 入口：batch runner 与便于调试的 single-instance runner。以下重点分析 batch 路径。

### 4.1 默认实验配置

默认实验条件不是只来自一个 YAML，而是 [`swebench.yaml`](https://github.com/SWE-agent/mini-swe-agent/blob/a83fcae82d2a08f0ee0c688f9d137b3566c097f8/src/minisweagent/config/benchmarks/swebench.yaml#L100-L183)、[runner CLI 默认值](https://github.com/SWE-agent/mini-swe-agent/blob/a83fcae82d2a08f0ee0c688f9d137b3566c097f8/src/minisweagent/run/benchmarks/swebench.py#L185-L222)与组件配置类默认值合并后的结果：

| 项                        | 默认值                                                       | 来源                      |
| ------------------------- | ------------------------------------------------------------ | ------------------------- |
| 数据集 subset / split     | `lite` / `dev`                                               | runner CLI                |
| runner workers            | 1                                                            | runner CLI                |
| shuffle / redo existing   | `false` / `false`                                            | runner CLI                |
| 模型                      | `anthropic/claude-sonnet-4-5-20250929`                       | `swebench.yaml`           |
| temperature               | 未显式设置，采用 provider/model 默认值                       | `swebench.yaml`           |
| parallel tool calls       | `true`                                                       | `swebench.yaml`           |
| Agent step limit          | 250                                                          | `swebench.yaml`           |
| Agent cost limit          | 3 美元                                                       | `swebench.yaml`           |
| Agent wall-time           | 未启用                                                       | `AgentConfig`             |
| Environment               | Docker                                                       | `swebench.yaml`           |
| 工作目录                  | `/testbed`                                                   | `swebench.yaml`           |
| 单条命令 timeout          | 60 秒                                                        | `swebench.yaml`           |
| Docker 启动 timeout       | 120 秒，覆盖整个 `docker run`，包括必要时拉取镜像            | `DockerEnvironmentConfig` |
| Docker container lifetime | 2 小时                                                       | `DockerEnvironmentConfig` |
| shell interpreter         | `bash -c`                                                    | `swebench.yaml`           |
| shell 初始化              | `BASH_ENV=/root/.bashrc`，让非 login shell 激活 testbed 环境 | `swebench.yaml`           |
| observation               | 10,000 字符，达到阈值后改为头尾各 5,000 字符                 | `swebench.yaml`           |

命令行 `-m`、`--model-class`、`--environment-class` 或额外 `-c` 可以覆盖 agent 环境，数据选择与并发则由 runner flags 控制。尤其注意 v2.4.6 的 SWE-bench 配置已经不再显式写 `temperature: 0.0`；“没配置”不是“确定为零”。报告 benchmark 结果时，不能只写“mini-swe-agent v2”：至少还要固定 YAML、模型类、模型版本、provider endpoint、provider 默认采样参数、预算与环境镜像。

内置 CLI 调 `datasets.load_dataset(dataset_path, split=split)` 时没有暴露 `revision` 参数。若上游 Hugging Face dataset 更新，同一条命令未必加载同一快照；严谨实验需要预先固定本地数据快照，或扩展 runner 显式传 revision。

### 4.2 一道题的端到端数据流

```text
Hugging Face dataset row
  ├─ instance_id
  ├─ problem_statement
  └─ image_name / docker_image / fallback image
             │
             ▼
       create Model
             │
             ▼
       start task container at /testbed
             │
             ▼
  ProgressTrackingAgent.run(problem_statement)
             │
       inspect → edit → test
             │
             ▼
 model creates patch.txt and submits sentinel + patch
             │
             ├─ <instance>.traj.json
             └─ preds.json[instance_id].model_patch
                                      │
                                      ▼
                         external SWE-bench evaluator
                         applies patch and runs tests
```

runner 本身做的工作是：

1. 将 `lite`、`verified`、`full` 等短名映射到数据集；
2. 加载 split；启用 shuffle 时先按 instance ID 排序、再以 seed 42 乱序，随后应用 regex filter，最后应用 slice；
3. 为每题确定容器镜像并启动 Environment；
4. 每题新建 Model 与 Agent；
5. 用 `ThreadPoolExecutor(max_workers=...)` 并行处理实例；
6. 保存逐题 trajectory；
7. 在锁保护下更新聚合的 `preds.json`。

它没有运行 gold tests，也没有计算 resolved rate。官方 SWE-bench evaluator 读取 `instance_id`、`model_name_or_path` 与 `model_patch`，在评测容器中应用 patch 后才执行测试和评分。换句话说，mini-swe-agent 的 built-in runner 是 prediction producer，不是 grader。

生成与评分还必须显式使用同一个 dataset 和 split。固定版本的 [runner](https://github.com/SWE-agent/mini-swe-agent/blob/a83fcae82d2a08f0ee0c688f9d137b3566c097f8/src/minisweagent/run/benchmarks/swebench.py#L185-L210) 默认生成 `SWE-bench_Lite/dev` prediction，而固定于分析日的 [SWE-bench evaluator](https://github.com/SWE-bench/SWE-bench/blob/334882dd1f2664cc55c1abfe9de4884af023c0c0/swebench/harness/run_evaluation.py#L797-L813) 默认 split 是 `test`；直接省略评分侧的 `--split` 会让 prediction IDs 与待评数据错位。下面是一组参数对齐的示例：

```bash
mini-extra swebench --subset lite --split dev -o runs/mini-v246-lite-dev

python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Lite \
  --split dev \
  --predictions_path runs/mini-v246-lite-dev/preds.json \
  --run_id mini-v246-lite-dev
```

这里的 evaluator 命令也只是交接示例；正式报告仍应固定 SWE-bench commit、镜像和 evaluator 参数。

### 4.3 Patch 是模型生成的协议产物

SWE-bench prompt 要求模型按顺序：

1. 只对实际修改的 source 文件执行 `git diff -- ... > patch.txt`；
2. 单独检查 patch header 与内容；
3. 用 sentinel 加 `cat patch.txt` 提交。

runner 不会再次从 Git working tree 自动收集 diff，也不校验 submission 是否真是 patch。一个格式错误的 submission 仍会写入 `preds.json`，最终由外部 evaluator 在 apply 阶段失败。

这是一项有意的最小化：交付纪律也交给模型。但它把 prompt compliance 混入了最终解题率。若两个 harness 使用同一模型，一个自动提取 Git diff，另一个要求模型手工生成并提交 patch，它们测到的并不完全是同一能力。

### 4.4 并发、跳过和失败语义

`--workers` 控制线程并发，每个任务实例拥有独立 Model、容器与 Agent；`preds.json` 的 read-modify-write 由专用全局锁保护。进程级成本统计和进度管理器也各自有同步机制，因此这里不能把锁缩写成只存在一处。

重跑时，只要某个 instance ID 已经出现在 `preds.json`，默认就会跳过它。失败路径又分三类：

- Agent 已创建：保存 trajectory，并在 `finally` 更新 prediction；
- 进入主 `try` 后、Agent 创建前发生环境错误：没有 trajectory，但会写空 patch；
- Model 构造、task 读取或 progress 初始化等 `try` 之前的步骤失败：交给 future 外层记录 uncaught error，不写该题 prediction，也没有 trajectory。

已经写入空 patch 的失败实例同样会被视作“已有结果”；要重试必须使用 `--redo-existing` 或删除对应条目。

这有利于单进程 batch 在 `preds.json` 完整时跳过已有结果，但它不是自动失败重试，更不是从 trajectory 续跑。文件锁只是进程内 `threading.Lock`，`write_text` 也不是原子提交；多个进程不能安全共享同一个 output directory，进程恰在写文件时崩溃也可能留下损坏 JSON。

### 4.5 容器不自动等于严格 sandbox

通用 `DockerEnvironment` 的默认 `run_args` 只有 `--rm`。SWE-bench 配置没有额外加入 `--network none`、CPU/memory 限制、只读挂载或 capability drop，因此源码层面不能声称内置 SWE-bench run 默认无网或具备强隔离。

对比之下，[ProgramBench 配置](https://github.com/SWE-agent/mini-swe-agent/blob/a83fcae82d2a08f0ee0c688f9d137b3566c097f8/src/minisweagent/config/benchmarks/programbench.yaml#L183-L204)显式设置：

- `--network none`；
- CPU 与内存上限；
- 非 root 用户；
- drop `SYS_PTRACE`；
- [6 小时 agent wall-time](https://github.com/SWE-agent/mini-swe-agent/blob/a83fcae82d2a08f0ee0c688f9d137b3566c097f8/src/minisweagent/config/benchmarks/programbench.yaml#L1-L4) 与 7 小时容器寿命。

两套 benchmark 使用的是同一个 Environment 实现，只是 contract 与配置不同。安全、污染控制和资源公平性属于完整实验协议，不能由“用了 Docker”四个字代替。

## 5. 极简设计的收益与代价

前四节的实现事实可以收束成一张权衡表。mini-swe-agent 极简的是接口和控制流，不是 action 空间、权限或完整实验系统。

| 设计选择                         | 直接收益                                       | 代价与必须固定的实验变量                                         |
| -------------------------------- | ---------------------------------------------- | ---------------------------------------------------------------- |
| 只暴露 bash action               | 复用现有开发工具链，Agent 与语言生态解耦       | 单个 action 权限巨大；预装工具、网络、用户和可见文件都会改能力   |
| 每次 action 新建 shell 进程      | 减少 PTY、交互模式和 shell session 的隐藏状态  | `cd`、`export` 不保留；文件、副作用与后台进程仍可能跨步存在      |
| append-only messages             | 控制流透明，便于调试、FT/RL rollout 与失败分析 | history 持续增长；没有自动摘要、检索或 sliding window            |
| 一次响应可提出多个 tool call     | 减少模型往返，保持 provider 原生协议           | 默认顺序执行；模型在整批结束前看不到任何中间结果                 |
| magic sentinel 作为结束协议      | 与模型和 Environment backend 解耦              | 依赖合并输出首行与零退出码，容易受日志前缀和命令组合影响         |
| 生成与外部 verifier 分离         | reward 可独立计算，trajectory 不必内置 grader  | patch、commit、命令输出等 artifact 协议本身会进入最终能力测量    |
| DockerEnvironment 只做执行适配   | backend 简单、可替换                           | 网络、capability、资源与 secret 策略必须由配置和部署显式给出     |
| 每轮覆盖写 trajectory checkpoint | 格式错误、预算退出和异常都能留下统一诊断结构   | 不是断点续跑；写入非原子，强中止与容器清理仍依赖外层 job manager |

这种线性结构让研究者能更明确地区分：成绩变化来自模型还是 scaffold，失败来自定位、observation 裁剪还是提交协议，以及 native tool call 与文本 action 编码究竟带来多少差异。

安全边界则不能因为接口短而被淡化。LocalEnvironment 会在宿主机以 `shell=True` 执行模型生成的任意字符串，并继承宿主环境变量；日常 CLI 的确认模式只是一道人机闸门，`-y/--yolo` 与 batch `DefaultAgent` 都不会代替 sandbox。不要把不可信 issue、仓库或网页内容交给 LocalEnvironment 后无人值守运行。

## 6. 如何把它用成一个严谨、可复用的实验装置

### 6.1 如果拿它做严谨 benchmark，应固定什么

trajectory 保存了 prompt、消息、组件配置与成本，但“可观察”不等于“可复现”。一次可审计实验至少要固定五组变量：

1. mini-swe-agent commit、Python、依赖版本/锁文件，以及完全解析后的配置；
2. 模型 snapshot、provider endpoint、网关/路由、实际请求参数、transport retry，以及进程级 cost/call limit；
3. dataset revision、split、filter/slice/shuffle、确切 instance 顺序和并发度；
4. 每题容器 image digest、网络/权限/资源策略、Docker/runtime 与宿主架构；
5. evaluator commit、dataset/test-spec revision、split、timeout、缓存策略与评分配置。

SWE-bench fallback image 使用 `:latest`，模型 API 也可能在相同名字下更新。仅有一份 `preds.json` 无法证明两次试验条件等价。附录 B 给出一份覆盖上述字段的 manifest 示例。

此外还应同时发布：

1. 每题 trajectory；
2. 原始 prediction artifact；
3. exit-status 分布；
4. evaluator 的逐题日志；
5. API/基础设施失败与模型失败的区分；
6. 多次采样与聚合方法。

这时“某模型在 mini-swe-agent 上的成绩”才有可审计含义。否则 mini-swe-agent 这个名字掩盖了 prompt、Model adapter、sandbox、预算和 patch 协议等大量会改变结果的自由度。

### 6.2 真正值得复用的设计原则

若自行实现一个最小 coding agent，可以优先复用以下原则：

1. **让 Agent 只编排，不解析 provider 细节**：action parsing 和 observation formatting 属于 Model adapter。
2. **把 Environment 压成结构化函数**：输入 action，输出合并后的命令输出、return code、exception；文件状态留在明确的 sandbox 中。
3. **用 append-only messages 作为单一事实来源**：先获得可解释性，再按需要引入 memory policy。
4. **把 prompt、output budget 与提交协议像代码一样纳入版本控制**：它们都是 Agent-Computer Interface 的组成部分。
5. **把生成与评分分开**：agent 产生可重放 artifact，独立 verifier 判断行为是否正确。
6. **区分逻辑轮次、真实 API attempt、模型成本与 wall time**：不要用一个 `api_calls` 覆盖四种量。
7. **sandbox policy 必须显式**：Docker 只是载体，网络、权限和资源限制才是合同。
8. **保存失败路径**：格式错误、timeout、预算耗尽和异常都应进入 trajectory，而不是只留下成功样本。
9. **所有“默认值”都进入实验 manifest**：默认配置也会随版本变化。

## 附录 A：配置如何组装并进入 prompt

### A.1 YAML、key-value spec 与覆盖顺序

mini-swe-agent 的 YAML 顶层通常分成四块：

```yaml
agent: # prompt、step/cost/time limit、trajectory 路径
model: # provider/model、动作格式、observation 模板、采样参数
environment: # cwd、命令 timeout、镜像、环境变量、隔离参数
run: # task 或 benchmark runner 自己使用的选项
```

配置加载有两种输入：YAML 文件，以及 `model.model_kwargs.temperature=0` 这样的 key-value spec。[配置解析器](https://github.com/SWE-agent/mini-swe-agent/blob/a83fcae82d2a08f0ee0c688f9d137b3566c097f8/src/minisweagent/config/__init__.py#L28-L56)先尝试按 JSON 解析后者的值，因此 `0`、`false`、数组和对象能保留类型；无法解析的值才当普通字符串。多个配置通过 `recursive_merge` 从左到右递归合并，后者覆盖前者。CLI 使用特殊的 `UNSET` 哨兵跳过“用户没有提供”的参数，避免一个默认 `None` 意外抹掉 YAML 中的值。

这里有一个容易踩的坑：显式使用 `-c` 后，Typer 不会再自动保留完整默认配置。只写：

```bash
mini -c model.model_kwargs.temperature=0
```

并不等于“在默认配置上改 temperature”。正确写法需要把基线也带上：

```bash
mini -c mini.yaml -c model.model_kwargs.temperature=0
```

### A.2 Prompt 变量的来源与 secret boundary

`DefaultAgent.get_template_vars()` 依次合并：

1. Agent config；
2. Environment 提供的变量；
3. Model config；
4. `n_model_calls`、`model_cost`、`elapsed_seconds`；
5. `run(task, **kwargs)` 注入的任务变量；
6. 当前调用额外传入的变量。

越靠后的同名字段优先级越高。Jinja 使用 `StrictUndefined`，引用不存在的变量会立刻失败，而不会悄悄渲染为空字符串。这对 benchmark 很重要：一个拼错的 `{{ task }}` 不会让整批实验在空 prompt 上无声运行。

LocalEnvironment 还会把 `os.environ` 平铺进模板变量。默认模板不会自动把全部环境变量发给模型，可是自定义模板能够引用其中任何值。因此，模板也属于 secret boundary；不要为了调试写出会展开 API key 或 token 的通配式内容。

## 附录 B：可审计实验 manifest 示例

下面是实验元数据 schema 的示例，不是 mini-swe-agent 自身可直接读取的配置。它刻意同时记录“显式值”和“沿用的默认值”。

```yaml
runtime:
  python: ...
  platform: ...
  dependency_lock_path: requirements.lock
  dependency_lock_sha256: ...
  critical_packages:
    litellm: ...
    tenacity: ...
    datasets: ...

agent:
  repository: SWE-agent/mini-swe-agent
  commit: a83fcae82d2a08f0ee0c688f9d137b3566c097f8
  agent_class: ProgressTrackingAgent
  base_agent_class: DefaultAgent
  resolved_config_path: resolved-config.yaml
  resolved_config_sha256: ...
  step_limit: 250
  cost_limit: 3.0
  wall_time_limit_seconds: 0

model:
  provider: ...
  api_base: ...
  gateway_or_route: ...
  requested_model: ...
  resolved_snapshot: ...
  model_class: LitellmModel
  effective_model_kwargs_path: effective-model-kwargs.json
  temperature: omitted
  tool_encoding: native
  transport_retry:
    MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT: 10
    wait: exponential(min=4,max=60)

process_limits:
  MSWEA_GLOBAL_COST_LIMIT: 0
  MSWEA_GLOBAL_CALL_LIMIT: 0

environment:
  backend: DockerEnvironment
  image_digests_path: instance-image-digests.json
  command_timeout_seconds: 60
  network: docker-default # 复现内置默认；none 是另一组加固条件
  run_args_path: docker-run-args.json
  cpu: ...
  memory: ...
  docker_version: ...
  host_arch: ...

runner:
  dataset_id: princeton-nlp/SWE-bench_Lite
  dataset_revision: ...
  split: dev
  filter_regex: ""
  slice: ""
  shuffle: false
  shuffle_seed: 42
  ordered_instance_ids_path: instance-ids.txt
  workers: 1
  task_retry_policy: none

evaluator:
  repository: SWE-bench/SWE-bench
  commit: 334882dd1f2664cc55c1abfe9de4884af023c0c0
  dataset_id: princeton-nlp/SWE-bench_Lite
  dataset_revision: ...
  split: dev
  predictions_path: preds.json
  run_id: mini-v246-lite-dev
  instance_timeout_seconds: ...
  max_workers: ...
  cache_level: ...
  run_config_sha256: ...
```

## 结语

mini-swe-agent 的核心不是“模型输出一条 bash，然后执行”这句显而易见的伪代码，而是围绕这句伪代码做出的一整套边界选择：

> Agent 只维护线性历史；Model 负责语言协议；Environment 负责执行协议；runner 负责规模化与 artifact；benchmark verifier 负责正确性。

这种拆法让 `DefaultAgent` 足够短，短到可以完整读懂、轻易覆写。它没有把 sandbox、安全、复现、上下文管理和评分包装成已经解决的问题，而是把这些边界明确留给周围组件与配置。

因此，mini-swe-agent 最适合被看成一把研究用手术刀。它不是功能最丰富的 coding agent，却能把“模型能力”和“agent scaffold”之间那层常被隐藏的组织切得很薄。拿它做 baseline 时，真正要比较的不是那 100 行是否神奇，而是当控制循环被压到最小后，prompt、shell、observation 和 verifier 分别贡献了多少。

## 参考资料

- [mini-swe-agent v2.4.6 release](https://github.com/SWE-agent/mini-swe-agent/releases/tag/v2.4.6)
- [mini-swe-agent v2.4.6 固定源码](https://github.com/SWE-agent/mini-swe-agent/tree/a83fcae82d2a08f0ee0c688f9d137b3566c097f8)
- [`DefaultAgent` 控制循环](https://github.com/SWE-agent/mini-swe-agent/blob/a83fcae82d2a08f0ee0c688f9d137b3566c097f8/src/minisweagent/agents/default.py#L38-L190)
- [Model / Environment / Agent Protocol](https://github.com/SWE-agent/mini-swe-agent/blob/a83fcae82d2a08f0ee0c688f9d137b3566c097f8/src/minisweagent/__init__.py#L39-L80)
- [配置解析](https://github.com/SWE-agent/mini-swe-agent/blob/a83fcae82d2a08f0ee0c688f9d137b3566c097f8/src/minisweagent/config/__init__.py#L10-L56)、[递归合并与 `UNSET`](https://github.com/SWE-agent/mini-swe-agent/blob/a83fcae82d2a08f0ee0c688f9d137b3566c097f8/src/minisweagent/utils/serialize.py)
- [Model 工厂与进程级统计](https://github.com/SWE-agent/mini-swe-agent/blob/a83fcae82d2a08f0ee0c688f9d137b3566c097f8/src/minisweagent/models/__init__.py#L12-L104)
- [LiteLLM Model 与 tool-call 路径](https://github.com/SWE-agent/mini-swe-agent/blob/a83fcae82d2a08f0ee0c688f9d137b3566c097f8/src/minisweagent/models/litellm_model.py#L23-L150)
- [原生 tool-call parser/formatter](https://github.com/SWE-agent/mini-swe-agent/blob/a83fcae82d2a08f0ee0c688f9d137b3566c097f8/src/minisweagent/models/utils/actions_toolcall.py#L11-L109)、[文本 action parser/formatter](https://github.com/SWE-agent/mini-swe-agent/blob/a83fcae82d2a08f0ee0c688f9d137b3566c097f8/src/minisweagent/models/utils/actions_text.py#L13-L67)
- [模型请求重试](https://github.com/SWE-agent/mini-swe-agent/blob/a83fcae82d2a08f0ee0c688f9d137b3566c097f8/src/minisweagent/models/utils/retry.py#L19-L25)
- [LocalEnvironment](https://github.com/SWE-agent/mini-swe-agent/blob/a83fcae82d2a08f0ee0c688f9d137b3566c097f8/src/minisweagent/environments/local.py#L12-L87)、[DockerEnvironment](https://github.com/SWE-agent/mini-swe-agent/blob/a83fcae82d2a08f0ee0c688f9d137b3566c097f8/src/minisweagent/environments/docker.py#L13-L152)
- [`InteractiveAgent`](https://github.com/SWE-agent/mini-swe-agent/blob/a83fcae82d2a08f0ee0c688f9d137b3566c097f8/src/minisweagent/agents/interactive.py#L30-L196)
- [SWE-bench 默认配置](https://github.com/SWE-agent/mini-swe-agent/blob/a83fcae82d2a08f0ee0c688f9d137b3566c097f8/src/minisweagent/config/benchmarks/swebench.yaml#L1-L183)、[batch runner](https://github.com/SWE-agent/mini-swe-agent/blob/a83fcae82d2a08f0ee0c688f9d137b3566c097f8/src/minisweagent/run/benchmarks/swebench.py#L48-L254)
- [ProgramBench 默认配置](https://github.com/SWE-agent/mini-swe-agent/blob/a83fcae82d2a08f0ee0c688f9d137b3566c097f8/src/minisweagent/config/benchmarks/programbench.yaml#L1-L255)、[runner](https://github.com/SWE-agent/mini-swe-agent/blob/a83fcae82d2a08f0ee0c688f9d137b3566c097f8/src/minisweagent/run/benchmarks/programbench.py#L29-L177)
- [固定版本的 v2 migration guide](https://github.com/SWE-agent/mini-swe-agent/blob/a83fcae82d2a08f0ee0c688f9d137b3566c097f8/docs/advanced/v2_migration.md)
- [固定版本的 control-flow 文档](https://github.com/SWE-agent/mini-swe-agent/blob/a83fcae82d2a08f0ee0c688f9d137b3566c097f8/docs/advanced/control_flow.md)
- [SWE-bench evaluator 固定源码](https://github.com/SWE-bench/SWE-bench/blob/334882dd1f2664cc55c1abfe9de4884af023c0c0/swebench/harness/run_evaluation.py)、[evaluation guide](https://github.com/SWE-bench/SWE-bench/blob/334882dd1f2664cc55c1abfe9de4884af023c0c0/docs/guides/evaluation.md)
- [DeepSWE 固定示例 task](https://github.com/datacurve-ai/deep-swe/tree/435ee89ec2f2e2289f33b0da4f992f0b7b7266b9/tasks/abs-stepped-slices)、[Pier runner v0.3.0](https://github.com/datacurve-ai/pier/tree/e69a20e4e0ac073ec71fde0274bab3d9f40bac87)
