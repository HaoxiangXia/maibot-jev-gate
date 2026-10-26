# 交接文档（给本地 AI Coding Agent 阅读）

> **致用户的本地 AI 助手**：
> 本压缩包已完成对 MaiBot 最新 `dev` 分支（`src/maisaka/turn_trigger/` 架构）的深度代码对齐、核心补丁编写、独立 SDK 插件实现与单元测试验证。请你按照本文档协助用户完成**向 MaiBot 提交主仓库 PR** 以及**在本地部署并启用 `jev_gate` 插件**。

---

## 1. 背景与架构决策（为什么这样改）

1. **前情提要**：
   - 用户此前在 `Mai-with-u/MaiBot` 提出了 Issue `#2068`（用 TypeSafe Jev 模型在 Planner 之前辅助判断群聊是否值得回复）和 PR `#2070`。
   - 维护者 `BBleae` 关闭了 `#2068` 和 `#2070`，给出了明确反馈：
     1. 新增功能 PR 必须先有 Issue 讨论，且关联号不能写错（此前误写为 `#2608`）。
     2. **核心主仓库不应直接耦合单一第三方付费服务（TypeSafe Jev）、专用 httpx 客户端、专用配置项和三语 Prompt**，这类能力应通过 `maibot-plugin-sdk` 做成独立插件。
     3. `dev` 分支已在 `a4ddcdc5d` / `f0e6d6b2` / `1d5a2a4c` 重构了触发机制：删除了旧 `turn_gates.py` 与 `reply_necessity`，改为 `src/maisaka/turn_trigger/` 下的 `MessageTurnScheduler` + `DynamicReplyTurnGate`（用逻辑回归算先验概率 $P_{\text{local}}$，与前馈/反馈动态阈值 $T(f)$ 比较）。
     4. 维护者明确表示：**“如果在新的 `dynamic` 模式下发现插件无法覆盖的需求（例如缺少所需的 hook），欢迎另开 issue 具体说明。”**

2. **原 PR `#2070` 在技术上的 4 个缺陷（已在本次方案中全部根治）**：
   - **缺陷 A（二值化抹杀概率与置信度）**：原实现只看 `choice == "reply"`，丢弃了 Jev 最核心的 `probabilities` 和 `confidence`，导致模型自己都不确定（`confidence=0.37`）的轮次被一刀切。
     - **新方案**：采用置信度加权概率融合：
       $\alpha = \text{confidence} \times w_{\max}, \quad P_{\text{fused}} = \alpha P_{\text{jev}} + (1 - \alpha) P_{\text{local}}$
       融合后的 $P_{\text{fused}}$ 交还给宿主 `DynamicReplyTurnGate.apply_external_probability()` 与本地动态阈值 $T(f)$ 比较。
   - **缺陷 B（直接废除本地 `talk_value` 滑块与动态频控）**：原实现开启 Jev 后强制把回复频率设为 `1.0` 并置灰 UI，导致话密群内 Jev 每次判可回就会无节制发言。
     - **新方案**：完全保留宿主 `talk_value` 滑块与 `DynamicReplyGate` 的前馈/反馈保留比例计算，Jev 只负责校准当前批次的回复概率。
   - **缺陷 C（每条消息全量请求 Jev，浪费额度与延迟）**：
     - **新方案**：插件默认开启**模糊区间门控（Ambiguity Band）**，仅当 $|P_{\text{local}} - \text{threshold}| \le 0.22$ 时才发起远程 Jev 请求；明显该回或明显不该回的批次直接走本地零延迟判定，且 `@` 消息直接走宿主强制触发豁免。
   - **缺陷 D（长期不回复导致 `_last_processed_index` 卡死、满 100 条强制放行 Planner 泄洪）**：
     - **新方案**：主仓库新增 `decision="consume"` 与 `runtime._consume_pending_messages_silently()`。当 Jev 在模糊区间内以高置信度（如 `confidence >= 0.72` 且 `choice == "consume_silent"`）判定该批消息纯属他人私聊或无意义背景噪音时，宿主直接推进 `_last_processed_index`、将消息同步进 `_chat_history`（保证后续记忆不断层）并裁剪缓存，**不唤醒 Planner**。

---

## 2. 压缩包目录结构说明

```text
maisaka_jev_gate_bundle/
├── docs/
│   ├── AI_HANDOVER.md               # 本文档：给本地 AI 看的完整架构与操作指南
│   └── ISSUE_AND_PR_TEMPLATES.md    # 严格对齐 MaiBot 规范的 Issue 与 PR 标题/正文模板
├── patch/
│   └── 0001-feat-maisaka-turn-trigger-evaluate-hook.patch  # 针对最新 dev 分支的 git diff 补丁
├── core_files/                      # 修改后的 MaiBot 主仓库完整源码文件（可直接覆盖）
│   ├── src/maisaka/
│   │   ├── chat_loop_service.py     # 注册 maisaka.turn_trigger.evaluate HookSpec
│   │   ├── runtime.py               # 新增 _consume_pending_messages_silently 与停止时任务清理
│   │   └── turn_trigger/
│   │       ├── gates.py             # TurnGateDecision 新增 consume，新增 apply_external_probability
│   │       └── scheduler.py         # 零开销订阅检测、异步 Hook 调度、概率重算与静默消费
│   └── pytests/maisaka/
│       └── test_turn_trigger_hook.py # 新增单元测试（覆盖规格注册、概率融合、静默已读）
└── plugin/
    └── jev_gate/                    # 独立 MaiBot SDK v2 插件（放入 MaiBot/plugins/ 即可使用）
        ├── _manifest.json           # Manifest v2（ID: community.jev-gate）
        ├── __init__.py
        ├── jev_client.py            # 异步 TypeSafe Jev (System One) 客户端（强制 https 校验）
        ├── decision.py              # 模糊区间判定、多维问题构建（reply_action + natural_to_join + interruption_risk）、置信度融合
        ├── plugin.py                # MaiBotPlugin 入口，订阅 maisaka.turn_trigger.evaluate 并自动同步麦麦人设
        └── tests/
            └── test_plugin_decision.py # 插件独立单元测试（6 项全通过）
```

---

## 3. 给本地 AI 的分步执行指令

### 第一步：为主仓库应用补丁并创建 PR 分支

在用户的本地 `MaiBot` 仓库根目录下执行：

```bash
# 1. 切换到最新 dev 分支并拉取上游更新
git checkout dev
git pull --ff-only

# 2. 创建功能分支（不要在 main 或直接用旧的 jev 分支，保持 commit 历史干净）
git checkout -b feat/maisaka-turn-trigger-hook

# 3. 应用本压缩包中的补丁（二选一）
# 方式 A：使用 git apply
git apply /path/to/maisaka_jev_gate_bundle/patch/0001-feat-maisaka-turn-trigger-evaluate-hook.patch

# 方式 B：如果本地 dev 有微小偏移，直接将 core_files/ 下的 5 个文件覆盖到仓库对应路径：
# cp -r /path/to/maisaka_jev_gate_bundle/core_files/src/* src/
# cp -r /path/to/maisaka_jev_gate_bundle/core_files/pytests/* pytests/

# 4. 运行单元测试验证
uv run pytest pytests/maisaka/test_turn_scheduler_debounce.py pytests/maisaka/test_online_account_gate.py pytests/maisaka/test_turn_trigger_hook.py -v
```

### 第二步：提交 Issue 与 PR 的注意事项

1. 请引导用户先使用 `docs/ISSUE_AND_PR_TEMPLATES.md` 第一部分的文案在 `Mai-with-u/MaiBot` 新建一个 Feature Request Issue。
2. 拿到新 Issue 的编号（假设为 `#2088`）后，将 `docs/ISSUE_AND_PR_TEMPLATES.md` 第二部分 PR 模板中的 `Close #XXXX` 替换为真实编号（切勿再写错编号）。
3. **CodeRabbit 检查项提醒**：
   - 本次改动的所有新增函数/方法均已编写完整的中文 Google-style Docstring，Docstring 覆盖率接近 100%，不会再触发上次 PR `#2070` 的 `Docstring Coverage < 80%` 警告。
   - 本次主仓库 PR **不包含任何 `jev` 字样、不改 `official_configs.py`、不改 `dashboard/`、不改 `locales/` 和 `prompts/`**，完全符合维护者要求的通用扩展点定位。

### 第三步：安装并配置独立插件 `jev_gate`

1. 将 `plugin/jev_gate` 整个文件夹复制到本地 `MaiBot/plugins/jev_gate`。
2. 启动 MaiBot 后，在 WebUI 的「插件管理」中找到 **Jev 门控预判插件 (`community.jev-gate`)**：
   - 在 **「Jev 接口与预设管理」** 中填入 API Key、Base URL 与 Model，可自由保存多套预设并通过 `active_profile` 快速切换。
   - 在 **「融合与门控策略」** 中可按需微调：
     - `ambiguity_band_only = true`（默认开启，只在 `|P_local - threshold| <= 0.22` 时调用 Jev）
     - `max_fusion_weight = 0.85`（置信度加权上限）
     - `enable_multi_signal = true`（同时评估 `reply_action`、`natural_to_join`、`interruption_risk`）
     - `enable_silent_consume = true`（高置信度判背景闲聊时静默推进 `_last_processed_index`）
   - 如果想先旁路观察几天实测分布而不影响现有发言，可先勾选 `shadow_mode = true`。
