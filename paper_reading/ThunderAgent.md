# ThunderAgent: A Program-Aware Agentic Inference System

> **论文**: [arXiv:2602.13692v1](https://arxiv.org/html/2602.13692v1)
> **标题**: ThunderAgent: A Simple, Fast and Program-Aware Agentic Inference System
> **作者**: Hao Kang (Georgia Tech), Ziyang Li, Xinyu Yang (CMU), Weili Xu (UIUC), Junxiong Wang, Chenfeng Xu, Simran Arora (Together AI), Beidi Chen (CMU), Tushar Krishna (Georgia Tech)
> **发表**: 2026-02-14

---

## 1. 核心问题

当前 Agent 系统（如 SWE-Agent、OpenHands）由**松散耦合**的组件构成：
- LLM 推理引擎（vLLM / SGLang）
- 工具编排器（Kubernetes）

每个 LLM 调用和工具调用被当作**独立的无状态请求**单独调度，导致三个核心问题：

### 1.1 KV Cache Thrashing
- Tool 执行期间，KV Cache 被清出以容纳新请求
- Tool 返回后需重新 Prefill 完整历史，延迟增加 **up to 7.14×**
- 吞吐量随并发数增加而严重下降

### 1.2 跨节点内存不均衡
- 现有策略将同一 Workflow 的请求固定到同一 GPU 节点
- 不同 Workflow 的 KV 足迹差异巨大，导致节点间内存不均衡
- 实测峰值不平衡达 **51%**，部分节点过载而其他节点空闲

### 1.3 工具生命周期管理缺失
- 已完成 Workflow 的 Docker 镜像等资源未被回收
- 磁盘占用持续线性增长直至系统崩溃
- 工具环境准备时间随并发数增加急剧增长

---

## 2. 核心贡献：LLM Program 抽象

论文将每个 Agent Workflow 抽象为 **Agentic Program**，作为调度的基本单元：

$$P = \langle ID, c, \mathcal{T}, \mathcal{L}, \tau, s \rangle$$

| 字段                                     | 含义                         |
| -------------------------------------- | -------------------------- |
| $ID$                                   | 全局唯一标识符                    |
| $c$                                    | 上下文 token 数（KV Cache 内存占用） |
| $\mathcal{T}$                          | 工具环境集合（Docker 沙箱、网络端口等）    |
| $\mathcal{L}$                          | 后端 GPU 节点位置                |
| $\tau \in \{R, A\}$                    | 执行阶段：Reasoning / Acting    |
| $s \in \{Active, Paused, Terminated\}$ | 调度状态                       |

**关键洞察**：Program 抽象让调度器第一次拥有端到端视角，看清每个 Workflow 的完整资源需求和所处阶段。

---

## 3. 成本模型

论文采用 **Space-Time Product (STP)** 作为资源消耗的核心指标：

$$\text{Cost}_x = \int_0^{t_x} M_x(t) \, dt$$

总成本分解为 5 项：

$$\text{Cost}_{total} \approx \underbrace{\text{Cost}_{decode} + \text{Cost}_{prefill}}_{\text{有效工作}} + \underbrace{\text{Cost}_{recompute} + \text{Cost}_{unused} + \text{Cost}_{caching}}_{\text{浪费开销}}$$

| 成本项 | 来源 |
|--------|------|
| $\text{Cost}_{recompute}$ | KV Cache Thrashing 导致的重新计算 |
| $\text{Cost}_{unused}$ | 跨节点内存不均衡导致的空闲 |
| $\text{Cost}_{caching}$ | Tool 执行期间 KV Cache 的空等 |

---

## 4. 程序感知调度器

### 4.1 调度操作原语

**Restore**：将 Paused 程序调度到有可用容量的后端节点：

$$P \leftarrow \langle ID, c, \mathcal{T}, \mathcal{L}', \tau, Active \rangle$$

**Pause**：将 Active 程序从后端移除，释放其 KV Cache：

$$P \leftarrow \langle ID, c, \mathcal{T}, \emptyset, \tau, Paused \rangle$$

### 4.2 状态感知暂停（State-aware Pausing）

核心观察：**Acting 阶段的 Program 可以安全暂停（释放 KV Cache）**，而不影响正确性。调度策略优先暂停 Acting 程序，保持 Reasoning 程序的 KV Cache 不被驱逐。

### 4.3 最短上下文优先驱逐（Shortest-First Eviction）

**引理 4.1**：重新填充 KV Cache 的成本随上下文长度**二次方增长**：

$$\text{Cost}_{recompute} \propto c_i^2$$

驱逐最短上下文的程序能最小化重计算代价。形式化为优化问题：

$$\min_{S} \sum_{i \in S} c_i^2 \quad \text{s.t.} \sum_{i \in S} c_i \geq \Delta C$$

恢复和暂停的评分函数：

$$S_{restore}(P) = \frac{1}{c_P} + \mathbb{I}(\tau = R)$$

$$S_{pause}(P) = \frac{1}{c_P} + \mathbb{I}(\tau = A)$$

### 4.4 全局程序感知等待队列

将所有 DP 节点的等待队列**统一为全局队列**。暂停的程序可以迁移到任意有空闲内存的节点恢复执行，解决了跨节点内存不均衡问题。

### 4.5 时间衰减机制

针对工具执行时间不可预测的问题，引入衰减函数平衡缓存成本和重计算成本：

- **连续时间**：$f(t) = e^{-\lambda t}$
- **离散时间**：$f(t) = x^{-t}$

在周期性检测中，有效内存判断变为：

$$C_{total} < \sum_{p \in \mathcal{L}, \tau=R} c_p + \sum_{q \in \mathcal{L}, \tau=A} c_q \times f(t_q)$$

---

## 5. 工具资源管理

### 5.1 Hook-based 垃圾回收
- Program 终止时自动触发 teardown
- 回收 Docker 沙箱、网络端口、计算槽位
- 有效维持磁盘内存在常数水平

### 5.2 异步环境准备
- 在高优先级程序即将恢复前，异步初始化工具环境
- 将环境准备时间隐藏在调度中，显著降低端到端延迟

---

## 6. 实验结果

### 6.1 Serving 吞吐量和 KV Cache 命中率

| 配置                                | 对比系统      | 吞吐量提升      |
| --------------------------------- | --------- | ---------- |
| GLM-4.6 / Qwen-3 + SWE-Bench-Lite | vLLM      | 1.48–3.58× |
| GLM-4.6 / Qwen-3 + SWE-Bench-Lite | Continuum | 1.17–3.31× |

ThunderAgent 在高并发下维持稳定吞吐量，KV Cache 命中率达到 ~100%（可预测工具场景）。

### 6.2 RL Rollout

| Workflow | 对比系统 | 吞吐量 |
|----------|---------|-------|
| mini-SWEAgent | vLLM + Gateway | 671.8 (1.79×) |
| OpenHands | vLLM + Gateway | 270.8 (3.92×) |

### 6.3 资源节省

- **磁盘内存节省**：up to **4.2×**
- **Prefill/Decode 延迟**：减少约 10%（工具资源管理贡献）

---

## 7. 与现有系统的比较

| 系统 | 问题 |
|------|------|
| vLLM (Request-aware) | 无 Program 感知，高并发下 Thrashing 严重 |
| Continuum | TTL 机制无法应对不可预测的工具执行时间，仍会触发 Thrashing |
| Autellix | 无 Workflow 局部性，高并发下互相驱逐 KV Cache |
| SGLang Model Gateway | 前缀感知路由导致负载集中到单一节点 |

ThunderAgent 的优势在于**程序级抽象 + 端到端协调**。

---

## 8. 系统集成

接入 ThunderAgent 只需三处改动：

1. LLM 请求中添加 `program_id` 字段
2. 工具执行时传递 `program_id`
3. Program 结束时发送 `POST /programs/release`

其他 API 和请求格式保持不变，集成成本极低。

---

## 9. 论文亮点

1. **Program 抽象**将 Agent 调度的核心问题从"请求级"提升到"工作流级"，这是系统设计上的关键洞察
2. **成本模型**形式化了 Agent 推理中的各类开销，为调度决策提供了理论依据
3. **最短上下文优先**策略由二次方重计算成本直接推出，简单有效
4. **时间衰减函数**的理论推导（指数/几何衰减）严谨且有说服力
5. 开源代码：https://github.com/HaoKang-Timmy/ThunderAgent

---

## 10. 思考与局限

- 论文假设推理引擎和工具编排器是分离的，若将 Tool 执行也纳入端到端调度，是否能进一步优化？
- 时间衰减函数的最优参数（$\lambda$, $x$）仍需经验调优，理论最优值的自动确定是未来方向
- 在更大规模集群（>100 GPU）上的表现有待验证

---

## 11. 一句话总结

> **ThunderAgent 的核心洞察**：把 Agent Workflow 抽象为"LLM Program"，让调度器看清每个 Workflow 的完整状态（KV Cache 大小、所处阶段、工具资源），从而实现"优先驱逐 Acting + 最短上下文优先 + 全局负载均衡"的三合一策略，解决了 KV Cache Thrashing 和跨节点内存不均衡两大核心问题。
