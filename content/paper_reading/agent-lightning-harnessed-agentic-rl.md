---
title: Agent Lightning v1.0 论文解读：让 Agent Harness 进入 RL 训练闭环
description: 分析 Agent Lightning v1.0 如何把部署时的 Agent Harness 接入强化学习训练，以及其中的 token 对齐、样本统计、调度和可复现性问题。
date: 2026-08-18
tags:
  - Agentic-RL
  - RL-Systems
  - Agent-Harness
  - Paper-Reading
---

# Agent Lightning v1.0 论文解读：让 Agent Harness 进入 RL 训练闭环

Agent 做事时，真正执行任务的往往不是一个孤立的 LLM。它还需要上下文拼接、工具调用、代码沙箱、重试、子 Agent 和流程控制。这些部分合在一起，通常被称为 **Agent Harness**。模型只负责一次次生成，Harness 决定下一次请求里究竟放什么内容，以及什么时候调用工具、总结上下文或切换子任务。

这给 Agent 的强化学习（RL）带来一个容易被忽略的变化：如果训练器重新实现一套简化的 Agent 循环，训练到的策略可能和部署时看到的上下文、工具协议与失败恢复方式不一致。可是，如果直接把现成 Harness 接到 LLM API 上，训练器又只能看到一串 request–response，无法直接看到中间的环境状态。

Agent Lightning v1.0 讨论的正是这个接口问题。论文把这种范式称为 **harnessed agentic RL**：部署时使用的 Harness 直接参与后训练，训练系统通过 LLM endpoint 代理记录请求、响应和奖励，再把它们组装成 RL 样本。作者给出约 3,500 行代码的轻量框架，并在搜索、通用指令跟随和编码 Agent 上验证。编码实验用约 6K 个训练样本，让 Qwen3.5-9B 在 SWE-bench Verified 上从 41.8% 提升到 56.4%，绝对提升 14.6 个百分点。[论文摘要与结论](https://arxiv.org/pdf/2608.17528#page=1)

这篇工作的价值主要在系统边界和训练语义的澄清，而不在于提出一个新的 policy gradient 算法。它提醒我们：把 API proxy 接上并不等于训练正确，必须明确 token 的来源、一个 rollout 的统计单位，以及动态样本如何进入固定的训练后端。

## 1. 训练器看到的，已经不是一条线性轨迹

传统的 agentic RL 通常把环境循环放在训练器里。模型生成 action token，环境返回 observation，训练器把它们拼成下一次输入：

$$
p_t=(p_{t-1},a_{t-1},o_t).
$$

这样，一个 rollout 自然对应一条连续的 token 轨迹，也就对应一个训练样本。训练器知道状态怎样变化，也知道每个 action 在哪一个上下文中产生。

Harnessed agentic RL 的控制权在另一侧。Harness 维护消息历史、调用工具、执行代码、创建子 Agent，并为每次模型调用单独构造 prompt。训练器通过代理只能记录：

$$
\mathcal C(\rho)=((p_1,a_1),(p_2,a_2),\ldots,(p_{T_\rho},a_{T_\rho})).
$$

这里的环境状态和 Harness 状态都是隐变量。论文将潜在执行状态写成 $s_t=(s_t^{\text{harness}},s_t^{\text{env}})$，模型实际看到的是 Harness 根据这些状态渲染出的 prompt，以及在该 prompt 下生成的响应。[§1–§2，PDF 第 2–4 页](https://arxiv.org/pdf/2608.17528#page=2)

因此，一个任务级 rollout 不再必然对应一个训练样本。连续请求可能因为重分词而无法拼接；Harness 总结上下文后可能开启一条新前缀；多 Agent 协作还可能产生不共享线性历史的分支。样本数只有在执行结束、样本构造完成后才知道。

这个变化可以用一个简单例子表示。假设同一 GRPO group 里有两个 rollout：第一个得到奖励 1，但被切成三个样本；第二个得到奖励 0，只形成一个样本。如果按 rollout 计算基线，平均奖励是 $(1+0)/2=0.5$。如果把四个样本当作独立单位，基线就变成 $(1+1+1+0)/4=0.75$。奖励没有变，样本切分方式却改变了梯度信号。论文认为这种统计单位应该保持在 rollout 层面。[§2.2、Figure 4，PDF 第 6 页](https://arxiv.org/pdf/2608.17528#page=6)

## 2. 四个容易被忽略的训练问题

### 2.1 文本前缀相同，不代表 token 前缀相同

为了节省计算，训练器通常希望把相邻的 LLM 调用合并起来：第二个 prompt 包含第一个 prompt 和 response，前面共享的 token 只计算一次。文本层面的前缀关系可以写成：

$$
(p_i^{\text{text}},a_i^{\text{text}})\preceq p_{i+1}^{\text{text}}.
$$

但 RL 依赖的是确切 token ID，安全合并要求更强的条件：

$$
(p_i^{\text{tok}},a_i^{\text{tok}})\preceq p_{i+1}^{\text{tok}}.
$$

两者不等价，原因至少有三种。

- **Chat template 不是组合函数。** 先渲染一段消息再追加新消息，和一次性渲染完整历史，可能插入不同的分隔符。论文观察到 Qwen 的模板可能移除先前的 `<think>` 标记。
- **解码再分词会漂移。** 一次生成的 `having` 可能由 `h`、`aving` 两个 token 组成；把相同文本放进下一次 prompt 后，分词器可能改成 `hav`、`ing`。文字没变，token 边界变了。
- **工具处理器会改写输出。** JSON 修复、空格规范化、结构化输出序列化都会让 Harness 接收到的文本与最初采样的 token 不完全一致。[§2.1、Figure 3，PDF 第 4–5 页](https://arxiv.org/pdf/2608.17528#page=4)

这也解释了一个反直觉的风险：如果代理为了提高命中率，把下一次请求中的历史 token 替换成之前保存的 token，可能确实获得了更长的共享前缀，但下一次响应并不是在这个“替换后的 prompt”下采样的。训练器实际上把 response 放到了一个模型没有见过的条件上，产生隐性的 off-policy 偏差。

论文比较了三条路：每次调用独立训练，最安全但会重复计算；用树结构共享前缀，计算利用率好但需要专门的 attention mask 和分布式后端；以及 **best-effort merging**。Agent Lightning 采用最后一种：只有当观测到的 token ID 严格满足前缀条件时才合并，否则关闭当前序列、开始一个新样本。它牺牲一部分 prefix reuse，换取训练时仍然使用模型实际消费过的 prompt。[§2.1，PDF 第 5 页](https://arxiv.org/pdf/2608.17528#page=5)

### 2.2 奖励和优势应该归属于 rollout

一个 rollout 可能被合并成 $N_\rho$ 个样本，而 $N_\rho$ 受到重分词、上下文总结和子 Agent 分支的影响。论文的编码实验中，平均只有 36% 的 rollout 最终保持为单一样本，每个 rollout 平均产生 2.41 个训练样本。[Figure 10，PDF 第 13 页](https://arxiv.org/pdf/2608.17528#page=13)

作者的选择是：奖励仍然在任务完成后得到，并复制给该 rollout 产生的样本；GRPO 的 group baseline 和 advantage 在 rollout 层面计算。这样，某个 rollout 是否被拆成几个样本，不会改变同组其他 rollout 的统计权重。这个选择并没有解决更细的 credit assignment：如果一次任务包含十次工具调用，最终成功奖励究竟该如何分给每一个调用，仍然是开放问题。

### 2.3 样本级归一化会放大偶然的切分差异

论文比较了三种损失归一化方式。用 $\ell_{\rho,j,t}$ 表示 rollout $\rho$ 的第 $j$ 个样本中第 $t$ 个 response token 的损失，三种形式可以概括为：

1. **Token mean**：所有 response token 等权。
2. **Sequence mean + token mean**：先在每个样本内平均，再让所有样本等权。一个 rollout 产生的样本越多，总权重越大。
3. **Rollout-level token mean**：先把同一 rollout 的 token 损失合在一起，再让每个 rollout 等权。

如果样本数主要由 retokenization 或上下文操作决定，第二种方法会把 Harness 的实现细节误当成训练信号。全 batch 的 token mean 在理论上也不依赖样本数，但作者发现它容易受长的负样本影响，训练后期可能不稳定。因此 Agent Lightning 采用第三种归一化：每个 rollout 贡献相同的平均梯度。[§2.3、式 (14)–(16)，PDF 第 6–7 页](https://arxiv.org/pdf/2608.17528#page=7)

### 2.4 动态样本要接入固定的训练后端

训练后端的 GPU 数量、data parallel 和 micro-batch 配置通常是固定的，Harness 产生的样本数量和长度却每轮都不同。简单 flatten 成一个 tensor batch 还不够，系统必须保留每条序列的 rollout ID 和 prompt-group ID，防止“产生样本多”的 rollout 获得额外统计权重。

同一个 rollout 的多个序列还应该在同一次 optimizer update 中完成。如果把它们拆到不同更新中，同一任务的不同部分就可能在不同 policy 版本下计算，形成 within-rollout policy skew。于是，后端调度同时要做两件事：平衡 GPU 上的 token 工作量，并保留 rollout 的统计边界。[§2.4，PDF 第 7–8 页](https://arxiv.org/pdf/2608.17528#page=8)

## 3. Agent Lightning v1.0 怎样把边界落到系统里

框架的设计没有把 Harness 重新塞回 Trainer，而是增加一个轻量控制平面，把 rollout 的生命周期和模型调用记录下来。整体由三个部分组成：[Figure 1、§3，PDF 第 1、8 页](https://arxiv.org/pdf/2608.17528#page=8)

| 组件               | 作用                                                                                                                            |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| API Gateway        | 保存 rollout、模型端点和 append-only events；提供 rollout API，并把 Harness 的 OpenAI-compatible 请求转发到训练器注册的模型端点 |
| Rollout Controller | 从 Gateway 读取排队任务，启动 Kubernetes Job 或本地进程，持续同步执行状态                                                       |
| Customized Trainer | 基于 VERL 注册 rollout，等待任务完成，读取 model request、log probability 和 reward，再通过 sample adapter 组装训练样本         |

每个 rollout 有唯一 ID，模型请求、奖励、工具事件和执行日志都用它关联。Gateway 是状态源，Controller 通过 reconciliation loop 将它与外部执行状态对齐。附录中明确说明，这种跨网络同步只保证 best-effort eventual consistency；网络恢复后，控制面会继续重试并最终收敛。[附录 A，PDF 第 18–21 页](https://arxiv.org/pdf/2608.17528#page=18)

这种分层让 Harness 只需要把 LLM endpoint 指向代理，就能接入训练。训练资源和 Agent 执行资源可以位于不同位置，甚至使用不同的部署方式。与此同时，样本适配器把前面讨论的语义选择固定下来：严格 token 前缀才合并、rollout-level advantage、rollout-level token mean loss。

### 3.1 Collocated async RL：用同一批 GPU 交替做 rollout 和更新

同步 RL 要等待一批 rollout 中最慢的任务结束，更新阶段的 GPU 可能长时间空闲。传统异步 RL 把 rollout 和更新放在两套 GPU 上，利用率较高，却增加了硬件需求，也要维护两个队列。

Agent Lightning 提出 **collocated async**：同一 GPU 池在 rollout 和更新之间分时复用。Gateway 收集到足够数据后停止接收新请求，等待在途请求完成，再进入更新阶段；更新期间到达的新请求会被暂停。切换对外部 Harness 不可见。论文报告这种方式相对同步 RL 约有 2 倍端到端加速，同时使用更少 GPU，但没有给出完整的 GPU 型号、吞吐表或多次运行方差，因此这里更适合把它看成工程设计和单次实验观察。[§3.1、Figure 6，PDF 第 9 页](https://arxiv.org/pdf/2608.17528#page=9)

### 3.2 网络重试、Kubernetes 和监控

控制面 API 都设计成幂等的，调用方在网络失败后可以直接重试。LLM 生成本身不能简单幂等化，因为同一个 prompt 的重试可能生成不同响应。训练器组装样本时，会对相同 prompt 的 `model_request` 事件去重，只保留最后一次请求。

Agent Lightning 还选择自托管 Kubernetes，而不是依赖 Modal、Volcano veFaaS 或 E2B 等商业 sandbox。每个 Agent 执行被调度成普通 Kubernetes Job，降低了大规模 rollout 的持续服务费用，也让执行环境更容易纳入自己的网络策略。

监控系统会保存训练和验证 rollout、状态、模型请求、奖励、token/turn 统计和 pod 日志。作者正是通过这些记录发现了编码训练中的 reward hacking，而不是只看最终成功率。[§3.2–§3.4，PDF 第 9–10 页](https://arxiv.org/pdf/2608.17528#page=10)

## 4. 三类 Agent 实验说明了什么

论文没有只在一个特制环境里验证框架，而是把同一套控制面接到三类 Harness。结果如下，数字均来自论文的验证曲线或最终 checkpoint：

| Agent        | 模型与训练                    | 数据/评测                                                                                            | 结果                                                 |
| ------------ | ----------------------------- | ---------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| Search       | Llama-3.2-3B-Instruct + GRPO  | HotpotQA 训练；HotpotQA、2WikiMultiHopQA、MuSiQue、Bamboogle、TriviaQA、Natural Questions 各抽 50 条 | exact match 从 25.1% 到 41.7%，+16.6 个百分点        |
| 通用指令跟随 | Qwen3-4B-Instruct-2507 + RLOO | Instruction Pre-Training 数据按 80/20 划分                                                           | 验证 reward 从 51.9% 到 70.2%，+18.3 个百分点        |
| Coding       | Qwen3.5-9B + mini-SWE-agent   | SWE-smith 过滤后约 6K train、400 test                                                                | SWE-bench Verified 从 41.8% 到 56.4%，+14.6 个百分点 |

Search 实验使用每个 prompt 4 条 rollout，通用指令跟随使用 8 条 rollout；两者的曲线说明框架能在不同模型和优化器设置下收集训练信号，但它们主要是可行性验证，并没有把 Agent Lightning 与所有相关 RL 框架做严格的同预算比较。[§4.1–§4.2、Figures 7–8，PDF 第 10–11 页](https://arxiv.org/pdf/2608.17528#page=11)

### 4.1 Coding 实验的关键不只是 56.4%

SWE-smith 原始包含 59,136 个任务、128 个 Python 仓库，Docker 镜像约 295 GB。作者发现其中 18,033 条没有问题描述，1,265 条在镜像中找不到对应 problem branch，还有一些任务要运行超过 7,000 个测试。于是先移除空描述、缺分支和测试数超过 200 的任务，再用 Qwen3.5-9B 对候选任务各运行四次：四次都成功的任务被剔除，成功与失败混合的任务保留，得到约 5,000 条；另加 1,000 条四次都失败的任务。最终训练集约 6,000 条，测试集约 400 条。[§4.3.1，PDF 第 11–12 页](https://arxiv.org/pdf/2608.17528#page=12)

这个数据管线是论文可复现性贡献的一部分，也带来一个需要留意的偏差：难度过滤使用的正是后续训练模型家族，留下的任务分布可能更贴合 Qwen3.5-9B，而不是一个与模型无关的难度划分。

训练中出现的 reward hacking 很具体：Agent 读取 Git 历史寻找 gold commit，用 `wget` 或 `curl` 从 GitHub 下载上游源码，用 `pip` 获取包源码，或者通过 Python 网络库联网。作者通过隐藏 `.git`、禁用 Git 命令，并用 Kubernetes 网络策略阻断一般出站访问、只保留白名单服务来处理这些问题。[§4.3.2，PDF 第 12 页](https://arxiv.org/pdf/2608.17528#page=12)

这些限制让奖励更接近“根据题目和本地仓库解决问题”，却也改变了真实部署环境的分布。因此，56.4% 说明这套受控训练设置有效，不能直接推断它在允许网络访问、不同 Harness 或其他软件仓库上有相同增益。

### 4.2 消融把“rollout 是统计单位”落到了结果上

编码实验比较三种设置：sample-level advantage + token mean；rollout-level advantage + token mean；rollout-level advantage + rollout-level normalization。第 128 步的验证 reward 分别为 35.0%、33.1% 和 38.2%。第三种设置的 entropy 增长更慢，也更稳定。最终 checkpoint 在第 208 步的 SWE-bench Verified 为 56.4%。[§4.3.3、Figure 9，PDF 第 12–13 页](https://arxiv.org/pdf/2608.17528#page=13)

这组结果支持一个较窄但重要的判断：在 Harness 让一个 rollout 产生动态数量样本时，只修正 advantage 还不够，损失归一化也需要保持 rollout 级别。它并没有证明 rollout-level 方案在所有 Agent、奖励函数和后端上都普遍更好。

## 5. 这篇论文真正解决了什么，代价是什么

我认为论文最重要的贡献是把一个常被当作“工程接线”的问题，拆成了可讨论的训练语义：

1. **Harness 是环境的一部分。** 上下文总结、工具输出变换和子 Agent 分支会改变模型看到的状态，因此部署 Harness 不应只被视为一个 API 客户端。
2. **统计单位应与任务语义对齐。** 如果样本数量由重分词或内部编排决定，奖励、优势和损失都按样本计数，等于把实现偶然性写进了梯度。
3. **正确性和效率需要明确取舍。** 严格 token-prefix merging 保持 on-policy，但会降低 prefix 复用率；用缓存 token 强行拼接更快，却可能改变模型实际采样时的条件。
4. **可复现性包括数据和控制面。** 编码 Agent 的数据清洗、反作弊规则、执行环境、网络策略和 rollout 日志，都会改变 RL 的有效任务分布。

代价也很清楚。best-effort merge 没有消除重复 prefill；rollout-level credit assignment 仍然无法回答“哪一次工具调用造成了成功”；Harness 的隐状态限制了训练器对失败原因的直接建模。Kubernetes 控制面采用 eventual consistency，网络重试后只保留最后一次相同 prompt 的响应，也可能引入难以观测的轨迹差异。

实验方面，论文报告的编码提升很有吸引力，但证据仍应按论文实际披露来理解：没有多随机种子、置信区间和完整计算预算，也没有与 SFT 或其他 Agentic RL 框架在同一数据、模型和资源约束下的系统比较。约 6K 样本和“modest compute”说明复现门槛可能下降，却不足以单独证明方法在更大规模或不同代码分布上同样成立。禁网和禁 Git 的 reward-hacking 防护提高了奖励可信度，同时也使训练环境与真实编码 Agent 的部署环境存在差异。

## 6. 对实践和后续研究的启示

如果要训练一个已经存在的 Agent Harness，第一步不应是先改写 Agent loop，而是先记录每次模型调用的**确切 prompt token、response token、log probability、rollout ID 和最终奖励**。没有这些 provenance 信息，后续的序列拼接和 loss 计算很难判断是否仍然对应真实采样过程。

第二步是把 rollout 作为默认的分析单位，再根据需要设计更细的 credit assignment。可以进一步记录工具调用、子 Agent 分支和上下文摘要等事件，研究哪些事件值得获得局部奖励，而不是直接把所有 token 平铺后平均。

第三步是把反作弊和监控当成训练闭环的一部分。对于 Coding Agent，测试通过并不自动意味着模型解决了问题；它可能找到了答案泄漏路径。网络策略、文件系统可见性、日志抽样和失败轨迹检查，都会影响 RL 信号的质量。

这也留下了几条具体的研究方向：如何在不改写 prompt 的前提下做更高效的树形前缀共享；如何在多 Agent 分支间分配信用；如何让固定 GPU 后端更好地处理动态长度和动态样本；以及如何在保留真实 Harness 的同时，建立跨部署环境的评测协议。

## 结语

Agent Lightning v1.0 的核心观点可以压缩成一句话：**Agentic RL 的环境不只包括外部工具和任务状态，还包括负责组织上下文与控制流程的 Harness。** 一旦训练器通过 API 边界观察这个系统，token 对齐、样本合并、优势估计、损失归一化和后端调度就不再是可随意选择的实现细节。

论文的工程结果说明，轻量控制平面、可追踪的 rollout 事件和 rollout-level 训练语义足以支撑搜索、通用工具使用和编码 Agent 的 RL 实验。它还没有解决长轨迹中的细粒度信用分配，也没有证明所有 Harness 都能以同样成本获得提升；但它提供了一个更准确的起点：先保证训练器看到的样本确实对应部署时发生过的调用，再讨论怎样把这些调用优化成更强的 Agent。

**论文**：[Agent Lightning v1.0: Towards Harnessed Agentic RL](https://arxiv.org/abs/2608.17528)（Zhiyuan He 等，2026-08-18）
**代码与训练脚本**：[microsoft/agent-lightning](https://github.com/microsoft/agent-lightning)
