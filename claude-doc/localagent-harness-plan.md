# Plan: LocalAgent 推理 Harness(CoT / 投票 / 两阶段 / Self-Refine)

## Context

`aibattle`(AI Battle Arena)是评估优先的多智能体博弈竞技场。当前 agent 层的 `ModelAgent` 是"单次 generation"的 LLM 包装:`observation → GameTemplate → prompt → 模型 → text → parse → action`。

本次目标:在 `ModelAgent` 之上**自研一批轻量的推理 harness / inference-time scaffolding**——通过多次 LLM 调用、结构化中间步骤来提升单个 agent 的决策质量。**注意**:这不是接入 LangChain/AutoGen 等外部框架(那是被否决的第一版理解),而是在框架内部实现 prompt 工程与多步推理能力。

### 首批四个 harness

1. **结构化 CoT** — 单次 generation,强制先输出结构化推理(牌力 / 底池赔率 / 对手可能动作)再给动作。最轻量,其余三个的基础。
2. **Self-Consistency(多数投票)** — 同一 prompt 用 `temperature>0` 跑 N 次,对动作做多数投票。算力换稳定性。
3. **两阶段:估计 → 决策** — Gen-1 估计对手牌力/范围 → 拼进 prompt → Gen-2 做最终决策。
4. **Self-Refine(自批评)** — Gen-1 初步动作+理由 → Gen-2 批评(审视 EV)→ Gen-3 修正。

### 学术支撑(已联网核实)

- **CoT**:[Chain-of-Thought Prompting Elicits Reasoning in LLMs (Wei et al., 2022, arXiv:2201.11903)](https://arxiv.org/abs/2201.11903) — 中间推理步骤显著提升复杂推理。
- **Self-Consistency**:[Self-Consistency Improves CoT Reasoning (Wang et al., 2022, arXiv:2203.11171)](https://arxiv.org/abs/2203.11171) — 采样多条路径再投票,GSM8K 56.5%→74.4%。
- **两阶段/估计对手范围**:[How Far Are LLMs from Professional Poker Players? (arXiv:2602.00528, 2026)](https://arxiv.org/abs/2602.00528) 把扑克拆为"用隐藏信息行动、估计对手范围、预判未来";[PokerSkill (arXiv:2605.30094)](https://arxiv.org/html/2605.30094v1) 用三阶段纯提示 scaffolding 达专家级、无需训练或求解器。
- **Self-Refine**:[Self-Refine: Iterative Refinement with Self-Feedback (Madaan et al., 2023, arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — 同一 LLM 当生成者/批评者/修正者,平均提升约 20%(NeurIPS 2023)。
- **(下一步)对手建模需要记忆**:[Readable Minds: Emergent ToM in LLM Poker Agents (arXiv:2604.04157)](https://arxiv.org/abs/2604.04157) — LLM 只有配持久记忆才发展出对手模型,印证"对手建模需跨手状态",本次先做无状态是对的。

### 关键决策(已与用户确认)

1. **范围**:四个 harness 全做;**无状态**(每次 `act()` 独立),对手建模/反思等跨手记忆暂缓。
2. **抽象**:**通用 harness + 游戏无关**——中间 prompt 通用拼接,最终决策步复用 `GameTemplate.parse`;一次写好适用所有游戏(Kuhn/Holdem/Connect4/Gomoku)。
3. **代码组织**:**共享基类 + 可组合步骤**(便于以后组合,如 CoT+投票)。
4. **可配置**:N(投票次数)、迭代轮数、temperature、自定义中间提示文本等在 YAML 里**可覆盖**(各有合理默认)。
5. **基石**:先把 `ModelAgent` 的 render/generate/parse/repair 循环抽成共享 helper,harness 复用它。
6. **config 类型**:新增 `type: local`,用 `harness: <名字>` 选具体 harness。

---

## 实现部分

### 步骤 1(基石):抽取共享原语

**新建 [src/aibattle/agents/template_loop.py](src/aibattle/agents/template_loop.py)**,提供 harness 共用的原语:

- `GenerateResult`(dataclass)— 归一化一次 generation 的输出:`content`(供 parse)、`full_text`(供日志,可含 thinking)、`meta`(provider 字段,合并进 metadata)。
- `async def run_template_loop(template, generate, request, *, max_retries=2) -> AgentResponse` — 逐字复刻 [model_agent.py](src/aibattle/agents/model_agent.py) 现有的 render→generate→parse→repair→INVALID 逻辑 + metadata 组装。`generate: Callable[[str], Awaitable[GenerateResult]]`。
- 一个**投票/解析辅助** `parse_or_none(template, text, request) -> Optional[Move]`(薄封装 `template.parse`),供 harness 在中间步用。

**改造 [src/aibattle/agents/model_agent.py](src/aibattle/agents/model_agent.py)**:`act()` 把 `ModelOutput → GenerateResult`(保留 `has_reasoning`/`finish_reason`/`truncated`/`completion_tokens`/`prompt_tokens` 这些 metadata key 不变),`return await run_template_loop(...)`。

> **风险(最高)**:日志/replay 依赖 `ModelAgent` 的 metadata key。循环须输出相同 key。实现后用 `scripts/smoke_v2.py` 跑一局 diff 前后日志确认。

### 步骤 2:LocalAgent 基类 + 可组合步骤

**新建 [src/aibattle/agents/local/base.py](src/aibattle/agents/local/base.py)**:

`HarnessAgent(Agent)`(`agent_type = "local"`)— 共享基类,持有 `client: ModelClient`、`template: GameTemplate`、`name`、`max_retries`,并提供可组合的步骤原语,各 harness 复用:

```python
class HarnessAgent(Agent):
    agent_type = "local"
    def __init__(self, *, client, template, name, max_retries=2, **harness_cfg): ...

    # 原语(供子类编排):
    async def _generate(self, prompt: str) -> GenerateResult: ...      # 调 client,归一化
    def _final_prompt(self, request) -> str: ...                       # template.render_prompt
    def _parse(self, text, request) -> Optional[Move]: ...             # template.parse
    def _vote(self, moves: list[Move]) -> Move: ...                    # 多数投票(并列取首个合法)
    def _compose(self, request, *, extra_context: str) -> str: ...     # 通用中间-prompt 拼接

    # 抽象:子类实现编排,返回 AgentResponse(把中间产物塞进 metadata)
    @abstractmethod
    async def act(self, request) -> AgentResponse: ...
```

中间产物(估计文本 / 候选列表 / 批评意见)统一塞进 `AgentResponse.metadata["harness"]`,供日志/replay 审查 harness 是否真的帮到决策。

### 步骤 3:四个 harness 子类

各占一文件,均 `agent_type="local"`,复用步骤 2 原语 + 步骤 1 基石:

- **[src/aibattle/agents/local/cot.py](src/aibattle/agents/local/cot.py) `StructuredCoTAgent`** — 在 `template.render_prompt` 末尾追加通用结构化指令("先逐项分析:你的牌力、底池赔率、对手可能的动作/范围,再在最后一行给出动作"),单次 generate,`run_template_loop` 的 parse/repair 兜底。参数:`cot_instructions`(可覆盖)。
- **[src/aibattle/agents/local/self_consistency.py](src/aibattle/agents/local/self_consistency.py) `SelfConsistencyAgent`** — 同 prompt 并发跑 `n` 次(`temperature` 可配),各自 parse 出 Move,`_vote` 取多数;全失败则退回单次 + repair。参数:`n`(默认 5)、`temperature`(默认 0.7)。metadata 记录票型分布。
- **[src/aibattle/agents/local/two_stage.py](src/aibattle/agents/local/two_stage.py) `TwoStageAgent`** — Gen-1 用通用"估计提示"(默认:"基于公开信息和行动历史,估计对手可能的牌力/范围,简要说明")→ `_compose` 把估计拼进 → Gen-2 走最终决策步 + parse/repair。参数:`estimate_prompt`(可覆盖)。metadata 记录 Gen-1 估计文本。
- **[src/aibattle/agents/local/self_refine.py](src/aibattle/agents/local/self_refine.py) `SelfRefineAgent`** — Gen-1 初步动作+理由 → Gen-2 批评(通用"审视该动作 EV/是否更优"提示)→ Gen-3 修正 + parse/repair。参数:`rounds`(默认 1 轮批评)、`critique_prompt`(可覆盖)。metadata 记录每轮初稿/批评。

所有中间提示**游戏无关**(只引用 `observation.rendered` / `history` / `legal_actions`),最终步永远复用 `GameTemplate`,因此自动适用全部游戏。

### 步骤 4:registry + loader + 默认参数

- **改 [src/aibattle/agents/registry.py](src/aibattle/agents/registry.py)**:新增 `_build_local_agent(cfg, game_name, seed)`,按 `cfg["harness"]` 在子注册表 `{cot, self_consistency, two_stage, self_refine}` 选类;复用 `make_client(cfg["model"])` + `make_template(game_name)` 构造;把 `cfg.get("harness_args", {})` 透传给 harness。`make_agent` 增加 `local` 分支。
- **改 [src/aibattle/config/loader.py](src/aibattle/config/loader.py) 第 83 行**:类型白名单加 `"local"`,错误信息同步。
- 每个 harness 的可配置参数(`n`/`temperature`/`rounds`/`*_prompt`)都有合理默认,YAML 仅覆盖需要的项。

### 步骤 5:pyproject.toml

**改 [pyproject.toml](pyproject.toml)**:`dev` 加 `pytest-asyncio>=0.23`;新增 `[tool.pytest.ini_options]`(`asyncio_mode="auto"`、`testpaths=["tests"]`、`markers`)。harness 复用现有 `openai`/`anthropic` client,无新运行时依赖。

### 实现顺序

1. `template_loop.py` + 改 `model_agent.py` → smoke 验证 metadata 不变。
2. `local/base.py`(`HarnessAgent` + 原语)。
3. 四个 harness 子类(cot → self_consistency → two_stage → self_refine)。
4. registry `local` 分发 + loader 白名单。
5. `pyproject.toml` + pytest 配置。
6. TDD 测试套件。

### 示例 YAML

```yaml
players:
  player_0:
    agent:
      type: local
      harness: two_stage
      name: deepseek-twostage
      model: { provider: fireworks, model_id: accounts/fireworks/models/deepseek-v4-pro,
               api_key_env: FIREWORKS_API_KEY, temperature: 0.0, max_tokens: 16384 }
      harness_args: { estimate_prompt: "先估计对手在当前行动线下最可能的牌力区间" }
  player_1:
    agent:
      type: local
      harness: self_consistency
      model: { ... }
      harness_args: { n: 7, temperature: 0.8 }
```

---

## 测试部分(TDD,先写测试钉契约)

新建 `tests/` 布局:

```
tests/
  conftest.py
  agents/
    test_template_loop.py     # 共享基石
    test_harness_cot.py
    test_harness_self_consistency.py
    test_harness_two_stage.py
    test_harness_self_refine.py
  config/
    test_registry_and_loader.py
  integration/
    test_runner_e2e.py        # 经真实 Runner 端到端(离线)
```

### conftest.py 共享 fixtures

- **`make_request`**(工厂)— 不启动 game 构造最小 `AgentRequest`+`Observation`,discrete/numeric 两变体,可覆盖 `legal_actions`/`decision_seed`/`match`。
- **`FakeModelClient`** — 子类化 `ModelClient`,`generate()` **返回真实 `ModelOutput`**(字符串自动包装),记录每次 `prompt`/`temperature`/`max_tokens` 到 `self.calls`,**支持按调用序脚本化**多个输出(harness 多步必需),脚本耗尽 `AssertionError`(抓过度调用)。
  > 关键:`ModelAgent`/harness 依赖 `out.full_text()`,fake 必须返 `ModelOutput`。
- **`real_kuhn`** — `make_game("kuhn_poker")` + `Runner` + `MatchLogger(None)`(零文件写),离线端到端。

### 各测试要点

**`test_template_loop.py`**(全离线,`FakeModelClient` + 真实 `KuhnTemplate`/`HoldemTemplate`):
- 首次成功→`attempts==1`、只调一次且 prompt 是渲染结果。
- 一次 repair 后成功→第二次 prompt 等于 `repair_prompt(request, bad)`、`attempts==2`。
- 耗尽→`INVALID`、`attempts==max_retries+1`、`invalid==True`。
- numeric 金额解析、缺金额触发 repair;metadata 透传 token/`truncated`/`has_reasoning`;`raw_output==full_text()`。
- **`ModelAgent` 委托一致性**:真实 `ModelAgent` 结果 == 直接调 helper(防回归锚点)。

**`test_harness_cot.py`**:
- render 出的 prompt 含结构化指令;最终能 parse 出动作;garbage→repair→INVALID 链路正常。
- `cot_instructions` 覆盖生效(自定义文本出现在发给 client 的 prompt 里)。

**`test_harness_self_consistency.py`**:
- 脚本 `n` 个输出(如 `["bet","bet","check","bet","check"]`)→ 投票得 `bet`;`metadata["harness"]` 含票型分布。
- `n`/`temperature` 透传(断言 `FakeModelClient.calls` 调了 n 次、temperature 正确)。
- 全部不可 parse → 退回 repair → 最终 INVALID。
- 并列(2 vs 2 vs ...)取首个合法 Move(确定性,便于复现)。

**`test_harness_two_stage.py`**:
- 两次调用:第一次 prompt 含"估计"指令、第二次 prompt **含第一次的估计文本**(断言拼接);最终 parse 出动作。
- `metadata["harness"]["estimate"]` 记录 Gen-1 文本。
- `estimate_prompt` 覆盖生效。
- Gen-2 garbage→repair。

**`test_harness_self_refine.py`**:
- 三步调用顺序正确:初稿→批评→修正;第三次 prompt 含批评内容(断言拼接)。
- `rounds` 控制批评轮数(rounds=2 → 调用次数相应增加)。
- `metadata["harness"]` 记录每轮初稿/批评。
- 修正后仍不可 parse → repair/INVALID 兜底。

**`test_registry_and_loader.py`**:
- `make_agent({"type":"local","harness":"two_stage","model":{...}}, game_name="kuhn_poker")` 构造出对应 harness;`harness_args` 透传;`game_name` 解析到正确 template。
- 未知 `harness` 名 / 缺 `model` → 清晰报错。
- `load_config`(`tmp_path` 临时 YAML)接受 `type: local`、拒绝未知 type、仍要求 `agent.type`。

**`test_runner_e2e.py`**(`@pytest.mark.integration`,离线):
- 用 `FakeModelClient` 驱动一个 harness agent 打完整 Kuhn match(`MatchLogger(None)`、`episode_dir=None`)→ `episodes` 数对、`failures==0`、`returns` 零和。
- harness 返回 `INVALID` 时 `fallback` 策略生效、match 完成。
- harness 是 drop-in `Agent`:与 builtin/random 对打跑通。

### Marker / CI

- **默认快速层**(仅新增 `pytest-asyncio`):template_loop、四个 harness、registry/loader——全离线、用 `FakeModelClient`。
- **`@pytest.mark.integration`**:runner e2e,离线但稍慢。
- 除 loader 用 `tmp_path` 外无测试写仓库文件系统;所有 match 用 `MatchLogger(None)`。

---

## Verification(端到端验证)

1. **测试**:`uv pip install -e ".[dev]"` → `pytest`(全绿,离线)。
2. **metadata 回归**:`python scripts/smoke_v2.py`(需 `.fireworks`)确认 `ModelAgent` 重构后日志 metadata key 不变。
3. **harness 手测**:写两个 `type: local` config(如 `two_stage` vs baseline `model`),`aibattle run` 一个 Kuhn/Holdem config 跑通,检查 `trajectories.json` 里 `metadata["harness"]` 有中间推理痕迹。
4. **对照实验**:同一模型 baseline(`model`)vs 各 harness 跑小规模锦标赛(复用 `scripts/*_tournament.py` 模式),看胜率/平均收益是否提升——验证 harness 实际效果。

---

## 关键文件

**新建**:
- [src/aibattle/agents/template_loop.py](src/aibattle/agents/template_loop.py) — 共享基石
- [src/aibattle/agents/local/base.py](src/aibattle/agents/local/base.py) — `HarnessAgent` + 可组合原语
- [src/aibattle/agents/local/cot.py](src/aibattle/agents/local/cot.py) / [self_consistency.py](src/aibattle/agents/local/self_consistency.py) / [two_stage.py](src/aibattle/agents/local/two_stage.py) / [self_refine.py](src/aibattle/agents/local/self_refine.py)
- `tests/`(conftest + 7 个测试模块)

**修改**:
- [src/aibattle/agents/model_agent.py](src/aibattle/agents/model_agent.py) — 委托 `run_template_loop`
- [src/aibattle/agents/registry.py](src/aibattle/agents/registry.py) — `local` 分发(子注册表 + `harness_args` 透传 + `game_name`)
- [src/aibattle/config/loader.py](src/aibattle/config/loader.py) — 类型白名单加 `local`
- [pyproject.toml](pyproject.toml) — `pytest-asyncio` + `[tool.pytest.ini_options]`

---

## 后续(本次不做,已规划)

- **跨手记忆型 harness**:对手建模(统计对手在各状态的下注/诈唬频率)、对局后反思(Reflexion 式 lesson memory)。需 agent 持有跨 `act()` 状态——架构上的下一步,有 [Readable Minds (arXiv:2604.04157)](https://arxiv.org/abs/2604.04157) 支撑。
- **工具增强**:接确定性 equity 计算器 / 底池赔率工具,弥补 LLM 算术短板(扑克杀手级应用)。
- **harness 组合**:如 CoT+投票、两阶段+self-refine(共享基类的可组合步骤已为此预留)。
