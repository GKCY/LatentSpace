# verl: 读懂 `ray_trainer.py` 里的 PPO 训练主线

> 文件: `verl/trainer/ppo/ray_trainer.py`
>
> 核心类: `RayPPOTrainer`
>
> 核心问题: verl 如何用一个 Ray single controller 串起 rollout、reward、old logprob、reference policy、critic、advantage、actor update、checkpoint 和 validation。

---

## 1. 这个文件到底负责什么？

`ray_trainer.py` 不是模型结构实现，也不是 PPO loss 的底层数学实现。它更像 verl PPO 训练的调度中枢：

> driver 进程负责组织数据流，真正重的模型计算交给 Ray worker group。

可以把它理解成一个 single-controller trainer。它在本地 driver 上维护训练状态、dataloader、global step、metrics 和 checkpoint 节奏，然后通过 RPC 调用不同 worker group：

| 角色 | 负责内容 |
|---|---|
| `actor_rollout_wg` | actor 训练、生成 old logprob、给 rollout replica 提供最新权重 |
| `critic_wg` | value model 推理和 critic update |
| `ref_policy_wg` | reference policy logprob，或者由 actor 去掉 LoRA adapter 代替 |
| `reward_loop_manager` | reward model / reward function 相关计算 |
| `llm_server_manager` | 管理 rollout serving replicas |
| `async_rollout_manager` | 面向 trainer 的 rollout / agent loop 入口 |
| `checkpoint_manager` | 负责训练权重和 rollout replicas 之间的同步、sleep/wake |

所以阅读这个文件时，不要期待在里面看到所有模型细节。它真正有价值的地方是：

```text
一个 prompt batch 进入训练循环后，到底经过哪些系统组件，最后怎么变成一次 PPO 更新。
```

---

## 2. 从 `main_ppo.py` 进入 `RayPPOTrainer`

训练入口在 `verl/trainer/main_ppo.py`。`TaskRunner.run` 会先准备 worker mapping、resource pool、tokenizer、processor、dataset，然后创建 trainer：

```python
trainer = RayPPOTrainer(
    config=config,
    tokenizer=tokenizer,
    processor=processor,
    role_worker_mapping=self.role_worker_mapping,
    resource_pool_manager=resource_pool_manager,
    ray_worker_group_cls=ray_worker_group_cls,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    collate_fn=collate_fn,
    train_sampler=train_sampler,
)

trainer.init_workers()
trainer.fit()
```

这里有一个重要分工：

```text
main_ppo.py:
  决定有哪些 worker class、资源池怎么分、数据集怎么建

ray_trainer.py:
  把这些 worker 和数据组织成 PPO 训练数据流
```

`Role` 定义在 `verl/trainer/ppo/utils.py`，包括：

```text
ActorRollout
ActorRolloutRef
Critic
RefPolicy
RewardModel
TeacherModel
```

当前主路径使用的是 unified model engine worker。actor 和 rollout 通常被放在同一个 `ActorRolloutRefWorker` / `ActorRollout` worker group 里，而 rollout serving replica 由后面的 `LLMServerManager` 管起来。

---

## 3. `__init__`: 先判断这次训练需要哪些组件

`RayPPOTrainer.__init__` 主要做四件事。

第一，保存配置和 tokenizer / processor。

第二，根据 config 判断功能开关：

```python
self.use_reference_policy = need_reference_policy(self.config)
self.use_teacher_policy = need_teacher_policy(self.config)
self.use_rm = need_reward_model(self.config)
self.use_critic = need_critic(self.config)
```

这些开关决定后面训练循环中是否要算 `ref_log_prob`、是否要启动 reward model、是否要算 values、是否要走 distillation。

第三，处理 LoRA 下的 reference policy：

```python
self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
```

如果 `ref_in_actor=True`，reference policy 不一定需要一个独立 worker。算 ref logprob 时，可以让 actor 在 `no_lora_adapter=True` 的模式下 forward，相当于用 base model 作为 reference。

第四，创建 dataloader，并计算总训练步数：

```python
self._create_dataloader(...)
self.total_training_steps = len(self.train_dataloader) * total_epochs
```

`train_dataloader` 用的是 `StatefulDataLoader`。这很关键，因为 checkpoint 不只保存模型，也会保存 dataloader state，恢复训练时可以接着上一次 batch 继续。

---

## 4. `DataProto`: 贯穿全链路的 batch 容器

理解这个文件，最好先理解 `DataProto` 的用法。训练循环里几乎所有阶段都在操作同一个 batch 容器。

可以粗略分成三层：

| 字段 | 放什么 |
|---|---|
| `batch` | tensor，例如 `prompts`、`responses`、`attention_mask`、`old_log_probs`、`values` |
| `non_tensor_batch` | 非 tensor 信息，例如 `uid`、`data_source`、`reward_model`、多模态输入 |
| `meta_info` | 控制信息，例如 temperature、global step、是否 validate、timing |

训练循环里常见操作包括：

```python
batch = DataProto.from_single_dict(batch_dict)
gen_batch = self._get_gen_batch(batch)
batch = batch.repeat(repeat_times=rollout_n, interleave=True)
batch = batch.union(gen_batch_output)
batch.reorder(global_idx)
```

这套写法的含义是：

```text
原始 prompt batch
  ↓
抽出生成需要的字段
  ↓
按 rollout.n 复制 prompt
  ↓
合并生成结果
  ↓
补上 reward / logprob / values / advantage
```

verl 这里没有把每个阶段拆成很多中间 class，而是让 `DataProto` 带着数据一路往下走。

---

## 5. `init_workers`: 把 Ray 资源和 worker group 接起来

`init_workers` 是系统初始化的核心。

第一步，创建 resource pool：

```python
self.resource_pool_manager.create_resource_pool()
```

resource pool 描述的是 Ray 侧的 GPU 分配。例如主训练池、reward model 池、teacher model 池。

第二步，根据 `role_worker_mapping` 创建不同角色的 worker class：

```text
ActorRollout / ActorRolloutRef
Critic
RefPolicy
```

如果多个角色落在同一个 resource pool 上，会通过 `create_colocated_worker_cls` 合并成 colocated worker class。这样一个 Ray worker 里可以挂多个 role 前缀。

第三步，spawn worker group：

```python
wg_dict = self.ray_worker_group_cls(...)
spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
```

spawn 后会得到类似：

```text
all_wg["actor_rollout"]
all_wg["critic"]
all_wg["ref"]
```

第四步，初始化模型：

```python
self.critic_wg.reset()
self.ref_policy_wg.init_model()
self.actor_rollout_wg.init_model()
```

actor rollout 放在最后初始化。代码注释里说明这是为了让 vLLM 更好估计 KV cache memory。

第五步，创建 rollout / reward / checkpoint 相关 manager：

```python
self.reward_loop_manager = RewardLoopManager(...)
self.llm_server_manager = LLMServerManager.create(...)
self.async_rollout_manager = AgentLoopManager.create(...)
self.checkpoint_manager = CheckpointEngineManager(...)
```

这里是 verl PPO 系统结构的分界点：

```text
actor_rollout_wg:
  训练侧 worker group，维护最新 policy 权重

llm_server_manager + replicas:
  rollout serving 侧，用来生成 response

checkpoint_manager:
  负责把训练侧权重同步给 rollout serving 侧
```

初始化最后会调用：

```python
self.checkpoint_manager.sleep_replicas()
```

意思是先让 rollout replicas 进入 sleep 状态，后续加载 checkpoint / 同步权重时再唤醒。

---

## 6. `fit`: PPO 训练循环的完整数据流

`fit` 是这个文件最重要的方法。它可以概括成下面这条主线：

```text
load checkpoint
sync weights to rollout replicas
optional validation before train

for epoch:
  for prompt batch:
    rollout generate
    reward
    old_log_prob
    ref_log_prob
    values
    rewards / KL penalty
    advantage
    update critic
    update actor
    checkpoint / weight sync / validation / logging
```

更贴近代码的数据流如下：

```text
DataLoader batch
  ↓
DataProto.from_single_dict
  ↓
add uid
  ↓
_get_gen_batch
  ↓
repeat by rollout.n
  ↓
async_rollout_manager.generate_sequences
  ↓
checkpoint_manager.sleep_replicas
  ↓
batch.union(generated responses)
  ↓
compute response_mask
  ↓
optional balance_batch
  ↓
reward model / extract_reward
  ↓
old_log_probs
  ↓
optional ref_log_prob
  ↓
optional values
  ↓
token_level_scores -> token_level_rewards
  ↓
compute_advantage
  ↓
update critic
  ↓
update actor
  ↓
checkpoint_manager.update_weights
```

这段代码的关键不是某一个函数，而是顺序。PPO 需要的一些字段必须按依赖关系逐步补齐：

| 阶段 | 新增字段 |
|---|---|
| rollout | `responses`、`attention_mask`、可能还有 `rollout_log_probs` |
| reward | `rm_scores` 或 reward extra info |
| extract reward | `token_level_scores` |
| old logprob | `old_log_probs` |
| ref policy | `ref_log_prob` |
| critic | `values` |
| KL penalty | `token_level_rewards` |
| advantage | `advantages`、`returns` |

最后 actor 和 critic worker 才能用这个完整 batch 做 update。

---

## 7. Rollout: 为什么生成后立刻 `sleep_replicas`

生成阶段在训练循环里是：

```python
combined_gen_output = self.async_rollout_manager.generate_sequences(combined_gen_batch)
self.checkpoint_manager.sleep_replicas()
```

这里的 `generate_sequences` 不直接调用 actor training worker，而是通过 `AgentLoopManager` 调 rollout serving client。底层可能是 vLLM、SGLang 或其他 rollout engine。

生成结束后马上 sleep rollout replicas，主要是为了释放或隔离 rollout engine 占用的资源。后面要计算 reward、old logprob、critic、actor update，这些可能复用同一批 GPU 或需要切换权重状态。

训练结束 actor update 后，会再调用：

```python
self.checkpoint_manager.update_weights(self.global_steps)
```

这一步把最新 actor 权重同步给 rollout replicas。于是每个 step 的节奏是：

```text
rollout replicas wake / serve
  ↓
生成 response
  ↓
rollout replicas sleep
  ↓
trainer update actor
  ↓
同步新权重到 rollout replicas
  ↓
下一轮 rollout
```

这也是 RL 训练系统和普通推理服务最大的区别之一：rollout engine 不是长期固定权重 serving，它要不断接收 trainer 的新 policy。

---

## 8. Reward: reward model 和 rule-based reward 最后都走 `extract_reward`

训练循环里的 reward 阶段是：

```python
if self.use_rm and "rm_scores" not in batch.batch.keys():
    batch_reward = self._compute_reward_colocate(batch)
    batch = batch.union(batch_reward)

reward_tensor, reward_extra_infos_dict = extract_reward(batch)
```

如果配置启用了 reward model，并且 rollout / agent loop 没有提前把 `rm_scores` 放进 batch，就会通过 `reward_loop_manager.compute_rm_score` 计算 reward model 分数。

但不管 reward 来自哪里，最后都会通过 `extract_reward(batch)` 统一抽取成：

```text
reward_tensor
reward_extra_infos_dict
```

随后写入：

```python
batch.batch["token_level_scores"] = reward_tensor
```

这里的命名值得注意：

```text
token_level_scores:
  原始 reward score

token_level_rewards:
  PPO advantage 真正使用的 reward
```

如果没有 KL-in-reward，二者相同。如果开启 KL penalty，`token_level_rewards` 会被扣掉 KL 项。

---

## 9. `old_log_probs`: PPO 更新的锚点

PPO policy loss 里通常会用：

```text
ratio = exp(log_prob_new - log_prob_old)
```

所以 batch 进入 actor update 前必须有 `old_log_probs`。

默认路径会调用：

```python
old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
batch = batch.union(old_log_prob)
```

`_compute_old_log_prob` 做了几件事：

```text
DataProto -> TensorDict
  ↓
padding 格式转 no-padding
  ↓
actor_rollout_wg.compute_log_prob
  ↓
拿到 log_probs / entropy / routed_experts / sum_pi_squared
  ↓
no-padding 转回 padding
  ↓
封装成 DataProto
```

这里的 `old` 指的是：

> rollout 后、actor update 前，用当前 actor 参数对这批 response 重新算出的 logprob。

如果配置了 rollout correction 的 bypass mode，代码会跳过重算，直接用 rollout 阶段带回来的 `rollout_log_probs`：

```python
apply_bypass_mode(...)
```

所以这块有两种语义：

```text
decoupled mode:
  old_log_probs 由 actor 在训练阶段重算

bypass mode:
  old_log_probs 直接来自 rollout_log_probs
```

这也是很多 fully async / disaggregated rollout 优化会盯上 `old_log_prob` 的原因。它既重，又经常卡在训练关键路径上。

---

## 10. Reference policy 和 KL penalty

如果配置需要 reference policy，训练循环会调用：

```python
ref_log_prob = self._compute_ref_log_prob(batch)
batch = batch.union(ref_log_prob)
```

reference policy 的需求来自两类配置：

```python
config.algorithm.use_kl_in_reward
config.actor_rollout_ref.actor.use_kl_loss
```

如果 `ref_in_actor=True`，会走：

```python
metadata["no_lora_adapter"] = True
output = self.actor_rollout_wg.compute_log_prob(batch_td)
```

否则走独立 ref worker：

```python
output = self.ref_policy_wg.compute_ref_log_prob(batch_td)
```

开启 `algorithm.use_kl_in_reward` 时，会调用 `apply_kl_penalty`：

```python
kld = core_algos.kl_penalty(
    data.batch["old_log_probs"],
    data.batch["ref_log_prob"],
    kl_penalty=kl_penalty,
)

token_level_rewards = token_level_scores - beta * kld
```

这里的 `beta` 来自 KL controller，并且会根据当前 KL 动态更新：

```python
kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
```

所以 KL-in-reward 路径里，reward 不是静态的。它会随着当前 policy 和 reference policy 的偏离程度调整惩罚系数。

---

## 11. Critic 和 advantage

是否启用 critic 由 `need_critic(config)` 决定。默认逻辑是：

```text
如果 critic.enable 显式设置:
  用这个值
否则如果 adv_estimator == GAE:
  需要 critic
否则:
  关闭 critic
```

启用 critic 时，训练循环先算 values：

```python
values = self._compute_values(batch)
batch = batch.union(values)
```

然后在 advantage 阶段调用：

```python
batch = compute_advantage(
    batch,
    adv_estimator=self.config.algorithm.adv_estimator,
    gamma=self.config.algorithm.gamma,
    lam=self.config.algorithm.lam,
    num_repeat=self.config.actor_rollout_ref.rollout.n,
    config=self.config.algorithm,
)
```

`compute_advantage` 支持多种 estimator：

| estimator | 依赖 |
|---|---|
| GAE | `token_level_rewards`、`values`、`response_mask` |
| GRPO | `token_level_rewards`、`uid` 分组、`response_mask` |
| REMAX | 额外的 `reward_baselines` |
| GDPO | reward extra info / non-tensor batch |
| optimal token baseline | `sum_pi_squared`、`old_log_probs` |

这里有一个设计点：

> advantage 在 driver 进程上算，不走 Ray worker。

原因是 advantage 计算相对轻，而且它主要是 batch tensor 上的后处理。重的 forward / backward 才交给 worker group。

---

## 12. Actor update 和 critic update

critic update 入口：

```python
critic_output = self._update_critic(batch)
```

actor update 入口：

```python
actor_output = self._update_actor(batch)
```

这两个函数的形态很像：

```text
DataProto -> TensorDict
  ↓
padding 转 no-padding
  ↓
写入 mini batch size / epochs / shuffle / seed 等 meta
  ↓
调用 worker group update
  ↓
收集 metrics
```

actor update 会额外设置：

```python
calculate_entropy
distillation_use_topk
compute_loss=True
```

需要注意 actor 的 PPO mini batch size 会乘上 rollout repeat 数：

```python
ppo_mini_batch_size = actor.ppo_mini_batch_size * rollout.n
```

这是因为原始 prompt batch 会被 `rollout.n` 扩展，每个 prompt 可能生成多条 response。

critic warmup 也在这里处理：

```python
if self.config.trainer.critic_warmup > self.global_steps:
    self.checkpoint_manager.update_weights(self.global_steps)
else:
    actor_output = self._update_actor(batch)
```

warmup 期间只训练 critic，不更新 actor。即便如此，代码仍然会 `update_weights`，主要是为了让 rollout replicas 维持正确的 wake/sync 节奏。

---

## 13. Batch balance: 解决不同 DP rank token 数不均

如果开启：

```python
config.trainer.balance_batch
```

训练循环会调用：

```python
self._balance_batch(batch, metrics=metrics)
```

它会按每条样本的有效 token 数估算 workload，然后重新排列 batch，使分到不同 data parallel rank 的 token 总量更接近。

直觉上，batch size 相同不代表计算量相同：

```text
样本 A: prompt + response 512 tokens
样本 B: prompt + response 8192 tokens
```

如果长样本集中落到某一个 DP rank，就会造成 straggler，其他 rank 等它。`_balance_batch` 通过重排样本降低这种尾部等待。

如果开启 `use_prefix_grouper`，它还会尽量把相同 `uid` 的样本放在同一个 rank 上，避免破坏 prefix sharing 的局部性。

---

## 14. Checkpoint: 不只保存模型，也保存 dataloader

`_save_checkpoint` 保存三类东西：

```text
actor checkpoint
critic checkpoint
dataloader state
```

路径形态是：

```text
default_local_dir/
  global_step_x/
    actor/
    critic/
    data.pt
  latest_checkpointed_iteration.txt
```

恢复时 `_load_checkpoint` 会找最新的 `global_step_x`，加载 actor / critic，再尝试恢复 dataloader：

```python
dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
self.train_dataloader.load_state_dict(dataloader_state_dict)
```

但如果 checkpoint 正好在 epoch boundary，代码会跳过 dataloader state restore：

```text
global_steps % len(train_dataloader) == 0
```

原因是这时保存下来的 dataloader state 可能已经标记为 exhausted。跳过恢复后，下一个 epoch 会从头开始迭代。

---

## 15. Validation: 也是 rollout，只是不更新模型

`_validate` 的流程和训练 rollout 类似：

```text
val dataloader
  ↓
repeat by val_kwargs.n
  ↓
generate_sequences(validate=True)
  ↓
optional reward model
  ↓
extract_reward
  ↓
process_validation_metrics
```

关键区别是：

```python
test_gen_batch.meta_info = {
    "recompute_log_prob": False,
    "do_sample": val_kwargs.do_sample,
    "validate": True,
    "global_steps": self.global_steps,
}
```

validation 不需要 old logprob、advantage、actor update。它只需要生成 response，然后通过 reward function / reward model 评估。

验证结果会按 `data_source` 聚合，核心指标放到类似：

```text
val-core/{data_source}/{var_name}/{metric_name}
val-aux/{data_source}/{var_name}/{metric_name}
```

如果配置了 `validation_data_dir`，还会把 prompt、output、ground truth、score dump 成 JSONL。

---

## 16. 一次训练 step 的字段视角

用字段变化重新看一遍，可能更容易调试。

最开始 dataloader 给的是 prompt 相关字段：

```text
prompts
attention_mask
position_ids
multi_modal_inputs
reward_model / data_source / extra_info
```

rollout 后增加：

```text
responses
attention_mask 更新
可能有 rollout_log_probs
可能有 rm_scores
```

reward 后增加：

```text
token_level_scores
reward extra info 写入 non_tensor_batch
```

logprob/value 后增加：

```text
old_log_probs
ref_log_prob
values
```

advantage 后增加：

```text
token_level_rewards
advantages
returns
```

actor update 最依赖的是：

```text
responses
response_mask
old_log_probs
advantages
returns
ref_log_prob 可选
rollout correction 字段可选
```

critic update 最依赖的是：

```text
values
returns
response_mask
```

如果训练中遇到 key missing，基本可以沿着这个顺序定位：到底是 rollout、reward、old logprob、ref、critic，还是 advantage 阶段没有把字段补上。

---

## 17. 读这个文件时最容易误解的几个点

### 17.1 Actor worker 不等于 rollout serving replica

`actor_rollout_wg` 是训练侧 worker group。rollout 真正对外生成 response，是通过 `LLMServerManager` 管理的 replicas。

二者通过 checkpoint engine 做权重同步：

```text
actor update 完成
  ↓
checkpoint_manager.update_weights
  ↓
rollout replicas 获得新权重
```

### 17.2 `token_level_scores` 不是最终训练 reward

`token_level_scores` 是 reward function / reward model 的原始分数。开启 KL-in-reward 后，真正进入 advantage 的是：

```text
token_level_rewards = token_level_scores - beta * KL
```

### 17.3 GRPO 不需要 critic，但需要 `uid`

GRPO 按同一个 prompt 的多条 response 做 group normalization，所以 `uid` 很重要。训练循环会给每条原始 prompt 加一个 uuid，然后 repeat 时保持分组关系。

### 17.4 `old_log_probs` 可能来自不同路径

默认是训练阶段重算。开启 rollout correction bypass mode 时，可以直接使用 rollout 阶段带回来的 `rollout_log_probs`。

调试 PPO ratio、KL、off-policy correction 时，要先确认当前配置到底是哪条路径。

### 17.5 Validation 也会调用 rollout manager

validation 不是单独的推理脚本。它复用 `async_rollout_manager.generate_sequences`，只是通过 `validate=True` 和 `recompute_log_prob=False` 让下游跳过训练专用逻辑。

---

## 18. 总结

`ray_trainer.py` 的核心不是“PPO 公式怎么写”，而是“verl 如何把 PPO 公式落到分布式系统里”。

它把一次训练 step 拆成一条稳定的数据生产线：

```text
prompt batch
  -> rollout
  -> reward
  -> logprob / value
  -> advantage
  -> critic update
  -> actor update
  -> rollout weight sync
```

这条线里有三个最重要的系统设计点：

第一，`DataProto` 是跨阶段的数据协议。所有 worker 和 manager 都围绕它添加、删除、重排字段。

第二，训练侧 actor 和 rollout serving 侧 replica 是分离的。actor update 后必须通过 checkpoint engine 把新权重同步到 rollout replicas。

第三，PPO 所需的 `old_log_probs`、`ref_log_prob`、`values`、`advantages` 都是在 driver 编排下逐步补齐的。理解这些字段什么时候出现，比逐行记住每个函数更重要。

如果只抓一个 mental model，可以记成：

```text
RayPPOTrainer 是 verl PPO 的交通调度员。
模型 forward/backward 在 worker 上跑，rollout 在 serving replica 上跑；
driver 负责让每个 batch 按正确顺序经过所有站点，并在最后触发 actor/critic 更新。
```
