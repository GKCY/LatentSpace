# 从任务到评分：DeepSWE 的任务设计与评测流程

> 本文基于 DeepSWE 仓库提交 `435ee89ec2f2e2289f33b0da4f992f0b7b7266b9`（本地分析日期：2026-08-18）。文中的任务数量和统计值均来自该版本的 113 个 task。

如果只看排行榜，DeepSWE 很容易被理解成“又一批给 coding agent 修 GitHub 仓库的题”。读完代码后，我认为它更值得关注的地方其实是另一件事：它把一个开放式的软件工程需求，封装成了一个可复现、可隔离、跨语言、能够程序化判分的 task capsule。

它最核心的设计可以压缩成一句话：

> Agent 只在固定版本的代码仓库中工作；结束时只把已提交代码导出成 patch；一个独立启动的 verifier 容器重新应用这个 patch 和隐藏测试，再按精确的测试 ID 做 fail-closed 判分。

这篇文章不讨论具体模型成绩，重点拆两件事：DeepSWE 的 task 到底长什么样，以及一次 benchmark run 从 prompt 到 reward 是如何流动的。

## 1. 先划清边界：DeepSWE 仓库不是 runner

这个仓库几乎没有传统意义上的应用代码。根目录主要是说明文档和 `tasks/`，真正的任务发现、agent 适配、sandbox 生命周期、并行调度与 trajectory 管理由外部的 [Pier](https://github.com/datacurve-ai/pier) 完成。

各层职责可以这样理解：

| 层 | 负责什么 | 在哪里 |
| --- | --- | --- |
| 上游开源仓库 | Agent 要理解和修改的真实代码 | 91 个 GitHub 仓库 |
| DeepSWE | Prompt、固定代码版本、环境、隐藏测试、评分合同、参考解 | 本文分析的仓库 |
| Harbor | Task 的目录与配置协议 | [Harbor task format](https://www.harborframework.com/docs/tasks) |
| Pier | 读取 task、启动 sandbox、驱动 agent、搬运 artifact、保存 trajectory | 独立 runner |
| mini-swe-agent | 官方排行榜统一使用的 agent harness | Pier 驱动 |

所以，仓库代码能直接证明的是“一个 task 如何声明和评分”；诸如默认并发数、失败重试、模型 API 路由、外层 job 聚合等属于 Pier。分析 benchmark 时若不区分这两层，很容易把框架能力误写成数据集自身的实现。

## 2. 这 113 道题实际在测什么

当前语料覆盖 91 个活跃开源仓库和 5 种声明语言（按 `manifest.json` 统计）：

| 语言 | Task 数 |
| --- | ---: |
| TypeScript | 35 |
| Python | 34 |
| Go | 34 |
| Rust | 5 |
| JavaScript | 5 |

题型分布更能说明 benchmark 的能力画像：106 道是 `feature_request`，只有 4 道 `bugfix` 和 3 道 `enhancement`。也就是说，DeepSWE 主要测的不是在一个小范围内定位单点 bug，而是阅读陌生代码库、发现扩展点、设计 API，并把一个包含多个并行要求的功能完整接入现有系统。

对当前 task 文件做静态统计，还能看到它为什么被称为 long-horizon：

- `instruction.md` 按空白分词，平均约 302 词，中位数 291 词；prompt 并不算特别长。
- `solution.patch` 平均修改 7.4 个文件、增加约 668 行内容；中位数也有 6 个文件、612 行新增。
- 隐藏 `test.patch` 平均增加约 969 行内容，中位数约 795 行。
- 每题的需求测试 F2P 中位数为 44，回归测试 P2P 中位数为 165。

因此它的难度来源不是把答案写进一篇超长需求文档，而是“短一些的行为规格 + 大很多的实现面”。官方方法说明也明确称这些任务和参考解是从零编写的，而不是从已经合并的 commit 或 PR 反推出来；部分选题可能受未解决 issue 启发，但公开历史中不存在可直接拷贝的 gold patch。

另一个容易误读的元数据是 `tasks/manifest.json`：它保留了 `swe-bench-ultra` 来源数据集的 `ext_id`，并披露展示标题、描述和 kebab-case task ID 由 `wide-research` 调用模型生成。这里的 AI 生成对象是 label，不是任务规格、验证器和参考实现。

从官方披露的方法看，一道题进入 corpus 前还经历了一条 authoring/QA 流程：先从公开、活跃、至少 500 stars、宽松许可证的仓库中选题；再成套编写 prompt、行为 verifier 和 reference solution；authoring 时让 verifier 连续运行三次排查 flaky；最后结合诊断性 agent rollouts，做 LLM-assisted analysis 和独立人工 review。Review 重点是 prompt 与 verifier 是否双向对应、是否接受不同但合理的实现、任务是否真实、环境是否干净。仓库本身保存的是这条流水线的最终产物，而不是完整的创作脚本和逐题 review 记录。

## 3. 一个 task 是一个“可执行的评分胶囊”

113 个 task 的文件布局完全一致。以 `abs-stepped-slices` 为例：

```text
abs-stepped-slices/
├── instruction.md
├── task.toml
├── environment/
│   └── Dockerfile
├── tests/
│   ├── Dockerfile
│   ├── config.json
│   ├── grader.py
│   ├── test.patch
│   └── test.sh
└── solution/
    ├── solution.patch
    └── solve.sh
```

这 10 个文件不是简单地把“题目、答案、测试”放在一起，而是把一次评测需要的四类信息分开：

1. `instruction.md`：agent 能看到的需求合同。
2. `task.toml` 与 `environment/`：运行合同和可复现代码环境。
3. `tests/`：agent 运行时看不到的 verifier 与评分合同。
4. `solution/`：供 task 作者和 reviewer 做 QA 的 oracle，不参与正式评分。

### 3.1 `instruction.md`：描述行为，不指定内部实现

`abs-stepped-slices` 要求给 ABS 语言的数组和字符串加入三段式 slice：

```text
value[start:end:step]
```

但任务并不告诉 agent 应该改哪个 parser 类型或调用哪个内部 helper。它描述的是外部可观察行为：省略 start/end 时如何解析，正负 step 如何迭代，step 为 0 的错误字符串，数组和字符串赋值如何广播，以及 Unicode 字符必须按 rune 而不是 byte 处理。

这种 prompt 有两个鲜明特征：

- 它是一份多分支的行为规格。只实现 happy path，通常过不了所有隐藏测试。
- 它给出兼容性边界和精确错误语义，但尽量不绑定私有符号和内部结构。

所有 113 份 prompt 最后还有同一句交付要求：从 `main` 新建分支，并提交全部修改。其中 commit 不是礼仪性文字，而是评分协议的一部分——后面收集的是 `base..HEAD`，未 commit 的工作树修改不会被评分；是否真的另建了分支则没有被 verifier 单独检查。

### 3.2 `task.toml`：把任务约束写成机器可读合同

每份配置都是 Harbor schema 1.3。删去无关字段后，核心结构如下：

```toml
schema_version = "1.3"
artifacts = ["/logs/artifacts/model.patch"]

[metadata]
language = "go"
repository_url = "https://github.com/abs-lang/abs"
base_commit_hash = "cb1b3b..."

[verifier]
network_mode = "no-network"
environment_mode = "separate"
timeout_sec = 1800.0

[[verifier.collect]]
command = "cd /app && ... && git diff --binary cb1b3b... HEAD > /logs/artifacts/model.patch"
timeout_sec = 300.0

[agent]
network_mode = "no-network"
timeout_sec = 5400.0

[environment]
docker_image = "public.ecr.aws/...:<task-image>-v1.1"
os = "linux"
cpus = 2
memory_mb = 8192
storage_mb = 20480
gpus = 0
```

我遍历了全部 113 份配置，以下策略是 corpus 级统一值：

| 约束 | 值 |
| --- | ---: |
| Agent 最大运行时间 | 5400 秒（90 分钟） |
| Verifier 最大运行时间 | 1800 秒（30 分钟） |
| Collect hook | 300 秒 |
| CPU | 2 |
| 内存 | 8192 MB |
| 存储 | 20480 MB |
| GPU | 0 |
| Agent / verifier task 网络 | no-network |
| Verifier 模式 | separate |

这些是分阶段上限，不应该直接相加成一次 trial 的承诺 wall time。镜像缓存、调度等待、API 延迟和提前失败都会改变实际耗时。

最关键的字段不是资源限制，而是这条 collect 命令：

```bash
git diff --binary <base_commit> HEAD > /logs/artifacts/model.patch
```

它把 agent 的交付物从“一个被运行时状态污染过的容器”收缩成“相对固定基线的一份已提交 patch”。`--binary` 使新增、删除和二进制文件修改也能被携带；`HEAD` 则意味着仅编辑但忘记 commit 的正确答案仍会得到空 patch。

### 3.3 `environment/Dockerfile`：把上游仓库冻结在过去

每个环境都从同一个 `mars-base` 出发，在 `/app` clone 对应上游仓库，然后执行几件很有针对性的事：

1. 把仓库默认分支强制指向固定 `BASE_SHA`，避免 detached HEAD，又保留自然的 Git 工作流。
2. 移除 `origin`、其他本地分支和 base 之后的 tag。
3. 清空 reflog 并执行 `git gc --prune=now`。
4. 在镜像构建期下载依赖和安装测试 reporter。
5. 多数任务关闭 commit hook，避免 Husky 等开发工具阻塞 agent 的必要 commit。

这同时解决两个问题：运行时无网时依赖仍然可用；agent 也不能通过 `git log --all` 在未来历史里寻找一个相似实现。

正常评测优先拉取 `task.toml` 指向的预构建 ECR 镜像，目录里的 Dockerfile 更像可重建配方。不过这里需要区分两种“可复现”：base commit、task 配置和已发布镜像让同版本评测高度可重复；但 Dockerfile 的第一层仍是 `mars-base:latest`，镜像引用也使用 tag 而非 `@sha256`，所以从源码重新构建并不是 bit-for-bit reproducible。

### 3.4 `solution/`：QA oracle，不是评分模板

`solution/solution.patch` 提供一份已知正确实现，`solve.sh` 应用它并像正常 agent 一样创建分支、add 和 commit。它有三项用途：

- 验证 task 确实可解；
- 让 reviewer 检查 prompt、测试和实现是否一致；
- 与空实现（nop）做 differential，离线生成 F2P/P2P 测试名单。

正式评分从不比较 agent patch 与 reference patch，也不会计算文件或 AST 相似度。只要外部行为正确，完全不同的内部结构也可以通过。

### 3.5 `tests/`：隐藏行为测试 + 跨语言适配层

`tests/test.patch` 把 task 作者写的新测试注入上游代码库；`test.sh` 运行原项目测试框架；`config.json` 列出真正参与判分的测试 node ID；`grader.py` 把不同框架报告统一成相同 reward 语义。

当前 113 份 `grader.py` 字节完全一致，语言差异全部留在 task-local `test.sh` 中：

- 26 个任务使用 `pytest-junitxml`，另有若干 pytest 变体；
- 25 个任务使用 `go-ctrf-json-reporter`，另有 10 个直接适配 gotest；
- TypeScript/JavaScript 侧混合 Jest、Vitest、Mocha、Node test 和 Deno；
- 5 个 Rust task 使用 `cargo-nextest`。

最终有 78 个 task 由公共 grader 读取 CTRF，35 个读取 JUnit。这个边界设计很干净：每种生态只需负责产出带稳定 node ID 的报告，公共评分器不用理解 pytest、Go test 或 Vitest 的命令行细节。

## 4. 一次 benchmark run 如何流动

整个流程可以画成下面这条链：

```text
pier run / 确定性采样
        │
        ▼
拉取预构建 task image，代码停在固定 base commit
        │
        ▼
Agent 容器：拿到 instruction，在 /app 仓库中工作
        │  最多 90 分钟；修改、测试、commit
        ▼
collect：git diff --binary base HEAD
        │
        └──────────── model.patch ────────────┐
                                               ▼
                                    独立启动 verifier 容器
                                               │
                                  apply model.patch
                                  apply hidden test.patch
                                               │
                                  跑 P2P + F2P + 可选 gate
                                               │
                                  JUnit / CTRF 报告归一化
                                               │
                                  binary reward + diagnostics
```

### 4.1 发现与采样

全量、本地确定性子集和单题分别是：

```bash
# 全量
pier run -p deep-swe/tasks --agent mini-swe-agent --model <provider/model>

# 固定随机子集
pier run -p deep-swe/tasks --agent mini-swe-agent \
  --n-tasks 10 --sample-seed 0

# 单题
pier run -p deep-swe/tasks/abs-stepped-slices --agent mini-swe-agent
```

`tasks/dataset.toml` 还为 113 个发布 task 记录了各自的 sha256 digest；`manifest.json` 则承担可读的 corpus 索引和展示元数据。task 之间如何并行并不写在任务包内，官方运行使用 Modal parallel sandboxes，这属于 Pier 和执行后端的职责。

### 4.2 Agent 阶段

Pier 启动隔离环境，将 `instruction.md` 交给 agent。每个 task 都把 agent 和 verifier 的执行阶段声明为 `no-network`；Pier 的额外价值之一，是只为 CLI agent 的安装和模型 API 请求设置所需 host allowlist，而不给代码仓库一般互联网访问。这解释了“task 无网”和“agent 仍能请求模型”为什么不矛盾。

官方 leaderboard 固定使用 mini-swe-agent，让不同模型拿到相同的 bash 工具和共享 prompt，以减少 native harness 差异。不过 Pier 也能驱动 Codex、Claude Code、Gemini CLI、OpenCode 等；用不同 harness 跑出来的结果回答的是另一个问题，不能与统一 harness 排行榜直接混在一起。

### 4.3 Committed patch 穿越信任边界

Agent 结束后，collect hook 在 agent 容器中产生 `model.patch`。在这个 task 集中，跨环境声明的唯一评分 artifact 就是这份 patch。

随后 verifier 使用 `tests/Dockerfile` 创建新容器。它从同一 task 基础镜像启动，却把 `test.sh`、`test.patch`、`grader.py` 和 `config.json` 烘焙到 `/tests`；agent 从未看到或控制这个容器。

这一步是 v1.1 最重要的变化。它消除了很多共享容器评分的歧义：agent 留下的进程、环境变量、临时文件、现场 monkey patch 或测试缓存不会自然延续到 verifier。评分只重放可审计的 committed diff。

### 4.4 Prepare：先重放答案，再注入隐藏测试

公共 `grader.py prepare` 做两次受控应用：

1. 解析 `model.patch` 涉及的路径，只把这些路径恢复到固定 base，再应用 agent patch。
2. 解析 `test.patch` 涉及的路径，把这些测试路径恢复到 verifier 镜像的 `HEAD`，再应用隐藏测试。

这样既不会粗暴 reset 掉镜像构建阶段的合法状态，也能避免 agent 用同名文件遮蔽隐藏测试。

失败语义也被明确区分：

| 情况 | 结果 |
| --- | --- |
| 没有提交 patch | 评分未应用模型修改的基线状态，按任务构造应为 0 |
| `model.patch` 无法 apply | `reward=0`，并记录 `apply_failed=1` |
| 隐藏 `test.patch` 无法 apply | 基础设施错误，由 shell trap 产生 `reward.txt=-1` |
| verifier 在产生 reward 前崩溃 | 同样以 `-1` sentinel 表示基础设施失败 |

这里还有一个边缘情况：如果 reporter 静默损坏或没有生成报告，但公共 grader 仍能正常运行，缺失的 node ID 会被当成普通测试失败，得到 0 而不是 -1。因此“模型失败”和“报告器失败”在极端情况下并非完全可分。

### 4.5 Test：F2P 证明需求，P2P 防止回归

DeepSWE 没有简单地看测试进程退出码，而是在 `config.json` 中物化两组精确白名单：

- F2P（fail-to-pass）：nop/base 状态失败、oracle 状态通过的测试，证明新需求真的被实现。
- P2P（pass-to-pass）：原先通过、应用 oracle 后仍通过的测试，证明改动没有破坏选定的既有行为。

`test.sh` 通常把它们叫作 `base` suite 和 `new` suite。这里的 base/new 是“回归测试组/新功能测试组”，不是让代码在旧版本和新版本之间切换；两组都运行在已经应用 agent patch 的同一个 verifier 工作树中。

105 个 task 读取两个报告，另有 8 个 task 增加 build、typecheck、codegen 等第三个 gate。无论原始框架是什么，最后都通过报告里的 node ID 与白名单对齐。

### 4.6 Grade：主分严格二元，partial 只用于诊断

令 `F`、`P` 分别为 F2P 和 P2P 白名单，主 reward 是：

```text
reward = 1  当且仅当 |F| > 0、所有 F2P 通过、所有 P2P 通过
reward = 0  其他情况
```

同时 `reward.json` 还会写出：

```text
f2p     = f2p_passed / f2p_total
p2p     = p2p_passed / p2p_total
partial = (f2p_passed + p2p_passed) / (f2p_total + p2p_total)
```

但 `partial` 不是一个适合跨题排名的分数。全语料有 5,877 个 F2P node 和 231,352 个 P2P node，各题测试数量极不均衡。极端例子 `expr-try-catch-errors` 有 79 个 F2P、66,265 个 P2P：即使需求一个都没完成，只要回归测试全部通过，`partial` 仍约为 0.9988，而 binary reward 正确地保持为 0。

所以这里的设计取舍很明确：binary reward 用于回答“整道工程任务是否完成”，各 pass fraction 用于回答“差在哪里”。仓库只定义 per-task reward；官网如何聚合、重复运行多少次以及怎样计算置信区间，属于外层 benchmark protocol，不能从这个 task 仓库的 grader 里反推。

公共 grader 还有三条重要的 fail-closed 规则：

- 白名单 ID 没出现在任何报告里，视为 failed，而不是忽略。
- skipped、pending 和 other 都不能算 pass。
- 同一个 ID 重复出现时采用 worst-status-wins：`failed > skipped > passed`。

这使“提前退出、删掉测试、只报告成功项”很难伪装成通过。相反，非白名单测试即使失败也不会直接影响 reward；P2P 的选择质量因此决定了回归保护的覆盖边界。

### 4.7 输出：排名信号与诊断证据分开

一次正常 verifier 会产出：

```text
verifier/
├── reward.json       # binary reward、计数、f2p/p2p/partial
├── ctrf.json         # 只含白名单项的统一机器可读报告
├── test-stdout.txt   # harness 捕获的标准输出与失败原因
├── run.log           # 原始 suite stdout/stderr
└── reports/          # pytest/jest/nextest 等框架原生报告
```

这套输出同时服务三类消费者：leaderboard 只需要 `reward`，task 作者用 pass fraction 和失败 ID 定位 verifier 问题，研究者则可以结合 raw log 与 Pier trajectory 做行为分析。

## 5. 我认为最好的设计：patch 是信任边界，不只是文件格式

DeepSWE v1.1 最漂亮的地方不是用了 Docker，也不是选了 CTRF，而是把 committed patch 设计成 agent 世界与 verifier 世界之间的唯一窄接口。

这个接口带来几项连锁收益：

1. **环境隔离**：评分不依赖 agent 容器结束时留下的瞬态状态。
2. **可审计**：每个有效改动都能用普通 Git 工具检查和重放。
3. **跨语言**：Go、Python、Rust、TypeScript 最终都退化为“应用 patch，执行测试，读取 node ID”。
4. **实现无关**：reference solution 不参与比对，行为测试可以接受多种内部设计。
5. **失败关闭**：缺失报告和跳过测试不会被误判成成功。
6. **自然工作流**：agent 在真实分支上工作和 commit，而不是被困在 detached HEAD。

这也说明为什么“只把隐藏测试藏起来”是不够的。若 verifier 仍在 agent 污染过的容器里执行，测试本身虽不可见，运行它的 Python、Node、环境变量和文件系统却可能已经不可信。separate verifier 把保密和执行完整性同时提升了一个层级。

## 6. 设计取舍与局限

### 6.1 它主要测大型功能集成，不代表全部 SWE 工作

106/113 的 feature-request 比例非常高。对长链路需求跟踪、跨模块集成、API 设计很有区分度，但 bug localization、小修复、纯重构、C++/Java、私有仓库和长尾项目都代表不足。官方也把这些列为当前限制。

### 6.2 All-or-nothing 很适合排行榜，不一定适合训练

一项需求的 43 个分支实现了 42 个，主分仍与完全没做一样是 0。这符合“功能能否交付”的产品视角，也避免大测试套件获得额外权重；但若用于 RL 或过程能力研究，F2P pass fraction、失败类别和 trajectory 信号可能比 binary reward 更有价值。

### 6.3 Commit 要求把交付纪律也纳入了能力测量

正确代码若未 commit 会得 0。可以把这看成真实工程流程的一部分，也可以把它看成 harness compliance 对纯编码能力的混入。无论立场如何，比较不同 agent 时都必须保证系统 prompt 对 commit 约定同样清楚。

### 6.4 “Held-out”是运行时隔离，不是永远保密

公开仓库中包含全部 `test.patch` 和 `solution.patch`。它们不会被挂载给 trial 内的 agent，但从 benchmark 发布之日起，就可能进入搜索索引、缓存甚至未来训练语料。原始任务不来自历史 PR，降低了发布时的污染；这不等于一个公开 benchmark 可以永久 contamination-free。仓库放置 canary 和删除上游未来历史是在缓解风险，而不是数学保证。

### 6.5 防作弊主要依赖隔离和 fail-closed，不是完整的 patch policy

每份 `test.sh` 都在注释中列出 dependency manifest、测试 runner 配置、越界路径等 “Cheating signal / Out-of-scope signal”，并写着 “recorded only”。但当前公开 task 的评分路径只调用 `grader.py prepare/grade`，没有调用已有的 `patch-paths` 子命令，也没有生成相应扫描记录或因修改这些路径自动扣分。

因此更准确的说法是：独立 verifier、隐藏测试、外置 reporter、精确 node ID 和 missing-is-failure 大幅提高了作弊成本；显式 patch scope 在仓库代码里只体现为作者留下的扫描意图，尚未形成自动检测、落盘或扣分的硬门禁。

### 6.6 评测可重复，不等于环境可永久重建

三处 base SHA（`task.toml`、`config.json`、Dockerfile）在 113 个 task 中全部一致，数据集也有 task digest，这一点做得很好。但镜像使用 tag，重建 Dockerfile 又依赖 `latest` 基础层和构建期网络。若要做多年期、可司法式审计的基准，还应该固定 OCI digest、runner/Pier 版本、agent 配置、模型 endpoint、系统 prompt、采样策略和重复次数。

### 6.7 基础设施失败与模型失败仍有灰区

没有产出 reward 的 verifier crash 会用 `-1` 标记；但报告文件损坏且 grader 继续运行时，所有白名单 ID 只是“缺失”，结果会变成普通 0。稳定 reporter、固定 node 命名和 authoring 阶段的多次去 flaky 检查能减少这种情况，却不能从语义上完全消除。

### 6.8 元数据还有几处工程 hygiene 问题

这些问题不影响 verifier 的主体设计，但分析数据分布和复现 task 时值得知道：

- 随仓库附带的 `manifest.schema.json` 要求 `dataset` 固定为 `deep-swe`，实际 manifest 写的是 `deep-swe-1-1`，当前两者不能直接通过 schema 校验。
- 至少三项语言标签与代码明显不符：HTTPX cookie store 被标成 TypeScript、Prometheus transactional reload 被标成 TypeScript、Koota snapshot 被标成 Python；其 reference patch 分别是 Python、Go 和 TypeScript。因此上面的语言表是 metadata 声明分布，不应理解为对实际改动文件的无误分类。
- 110 个 task 使用完整 40 位 base SHA；`eicrud-keyset-pagination-cursor` 和 `langchain-request-coalescing` 使用 7 位短 SHA，`koota-entity-snapshot-rollback` 则是 39 位。它们在 task 配置、测试配置和 Dockerfile 内部一致，但长期寻址强度弱于完整对象 ID。

## 7. 如果要复用这套设计，真正值得抄的是什么

如果自己设计 coding-agent benchmark，我会优先复用以下原则，而不是照抄某个测试脚本：

1. **把 prompt、reference solution、verifier 做成三角校验**：prompt 定义需求，oracle 证明可解，verifier 只检查公开行为。
2. **把 agent 输出缩成可重放 artifact**：代码任务优先使用 committed patch，而不是复用 agent 容器。
3. **在独立 verifier 环境评分**：隐藏测试和测试运行时都不应由 agent 控制。
4. **同时设计 F2P 与 P2P**：既证明新功能存在，也证明旧行为没坏。
5. **按稳定 test ID 判分并默认失败**：不要只相信 suite exit code，也不要忽略缺失结果。
6. **把排名信号与诊断信号分开**：binary reward 便于比较，fraction、CTRF、raw log 和 trajectory 用于解释。
7. **统一协议，不强行统一生态**：让各语言保留原生测试框架，只在报告层归一化。
8. **把整个外层实验配置版本化**：task digest 之外，还要固定 runner、harness、模型参数与重复策略。

## 结语

DeepSWE 的 task 不是一条 issue，也不是一对“输入—标准答案”，而是一个带代码时间点、资源边界、网络策略、交付协议、隐藏行为测试、回归保护、参考 oracle 和审计产物的可执行实验单元。

从流程设计上看，它的主线非常清晰：

> 固定过去的代码世界，让 agent 自由工程化求解；只把已提交 patch 带出这个世界；再在另一个不受 agent 控制的世界中，用行为测试回答“这项工作是否真的交付”。

这条跨越信任边界的 patch 流，才是 DeepSWE 相比“跑一下 gold tests 看退出码”的 benchmark 更值得研究和借鉴的部分。

## 参考资料

- [DeepSWE GitHub 仓库](https://github.com/datacurve-ai/deep-swe)
- [DeepSWE 方法与任务构造说明](https://deepswe.datacurve.ai/blog/deepswe)
- [DeepSWE v1.1：独立 verifier 与 committed diff](https://deepswe.datacurve.ai/blog/deepswe-v1-1)
- [Pier：DeepSWE 使用的 Harbor-compatible runner](https://github.com/datacurve-ai/pier)
- [Harbor Task Structure 与 separate verifier 规范](https://www.harborframework.com/docs/tasks)
- [示例 task：abs-stepped-slices](https://github.com/datacurve-ai/deep-swe/tree/435ee89ec2f2e2289f33b0da4f992f0b7b7266b9/tasks/abs-stepped-slices)
