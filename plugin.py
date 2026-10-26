"""Jev 门控预判插件 (Jev Gate Plugin) 主入口。

架构设计：
1. 首选模式（主仓库已合入 `maisaka.turn_trigger.evaluate` Hook 补丁）：
   订阅 `maisaka.turn_trigger.evaluate`，在进入 Planner 前、DynamicReplyGate 算出本地先验概率后，
   仅对处于模糊区间 `|P_local - threshold| <= ambiguity_margin` 的消息发起 Jev 结构化评估，
   按置信度加权融合概率 `P_fused = alpha * P_jev + (1 - alpha) * P_local`，
   并在高置信度判定为背景闲聊时返回 `decision="consume"` 静默推进 `_last_processed_index` 游标。

2. 多预设支持（Profiles）：
   支持在 WebUI 中保留多家服务商或多个端点的配置，通过 `active_profile` 快速无损切换，
   同时向下兼容旧版单一平铺字段。
"""

from __future__ import annotations

from typing import Any

from pydantic import model_validator

from maibot_sdk import Field, HookHandler, MaiBotPlugin, ON_BOT_CONFIG_RELOAD, PluginConfigBase
from maibot_sdk.types import ErrorPolicy, HookMode, HookOrder

from .decision import (
    DEFAULT_DECISION_PROMPT,
    GatePolicyConfig,
    build_jev_questions,
    build_jev_state,
    fuse_jev_evaluation,
    should_call_jev,
)
from .jev_client import JevClient, JevConfigError


class PluginSection(PluginConfigBase):
    """插件开关配置。"""

    __ui_label__ = "基础设置"
    __ui_icon__ = "sliders"
    __ui_order__ = 0

    enabled: bool = Field(
        default=True,
        json_schema_extra={
            "label": "启用 Jev 门控预判插件",
            "hint": "开启后将在群聊中通过 Jev 模型辅助门控裁决",
        },
        description="是否启用 Jev 门控预判插件",
    )
    config_version: str = Field(
        default="1.0.0",
        json_schema_extra={
            "label": "配置文件版本号",
            "hidden": True,
        },
        description="配置文件版本号",
    )
    shadow_mode: bool = Field(
        default=False,
        json_schema_extra={
            "label": "影子观测模式",
            "hint": "仅旁路调用 Jev 并在日志输出预判详情，不干预原本群聊触发判定",
        },
        description="影子观测模式：仅旁路调用 Jev 并记录融合日志，不改变宿主原有的触发判定",
    )


PluginSection.model_rebuild()


class JevProfileConfig(PluginConfigBase):
    """单个 Jev 服务端点/模型预设配置。"""

    name: str = Field(
        default="TypeSafe 官方",
        json_schema_extra={
            "label": "预设名称",
            "placeholder": "例如: TypeSafe 官方 / 自建代理",
        },
        description="预设唯一标识名称，用于在 active_profile 中选择切换",
    )
    enabled: bool = Field(
        default=True,
        json_schema_extra={
            "label": "启用此预设",
        },
        description="是否启用此预设（若多个预设均启用，优先采用上方指定项）",
    )
    api_key: str = Field(
        default="",
        json_schema_extra={
            "label": "API Key",
            "placeholder": "输入 TypeSafe Jev API Key（Bearer Token）",
            "input_type": "password",
        },
        description="Jev API Key（Bearer Token）",
    )
    base_url: str = Field(
        default="https://api.typesafe.ai/v1",
        json_schema_extra={
            "label": "API 基础地址",
            "placeholder": "https://api.typesafe.ai/v1",
        },
        description="Jev API 基础地址（必须为 https://）",
    )
    model: str = Field(
        default="jev-latest",
        json_schema_extra={
            "label": "模型名称",
            "placeholder": "jev-latest",
        },
        description="Jev 模型名称",
    )
    timeout_seconds: float = Field(
        default=3.5,
        json_schema_extra={
            "label": "请求超时秒数",
            "step": 0.5,
        },
        description="单次 Jev HTTP 请求超时秒数",
    )


JevProfileConfig.model_rebuild()


class JevApiSection(PluginConfigBase):
    """Jev 接口与多预设配置。"""

    __ui_label__ = "Jev 接口与预设管理"
    __ui_icon__ = "globe"
    __ui_order__ = 1

    active_profile: str = Field(
        default="TypeSafe 官方",
        json_schema_extra={
            "label": "当前激活预设名称",
            "placeholder": "填写下方预设名称，例如: TypeSafe 官方",
            "hint": "填写下方列表中想要生效的预设名称；留空时自动选用列表中首个启用的预设",
        },
        description="当前激活生效的预设名称（与 profiles 列表中的某项 name 对应）",
    )
    profiles: list[JevProfileConfig] = Field(
        default_factory=lambda: [JevProfileConfig()],
        json_schema_extra={
            "label": "Jev 服务配置预设列表",
            "hint": "可添加多家服务商或多个端点，随时切换无需重复输入",
        },
        description="Jev 服务配置预设列表：可保留多家服务商配置，切换时仅改 active_profile 即可",
    )
    context_message_count: int = Field(
        default=12,
        json_schema_extra={
            "label": "附带最近聊天历史条数",
            "step": 1,
        },
        description="附带给 Jev 的最近聊天历史条数",
    )

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_flat_fields(cls, data: Any) -> Any:
        """平滑向下兼容旧版平铺配置，将其自动迁移收敛进 profiles 列表。"""
        if isinstance(data, dict):
            if not data.get("profiles") and (data.get("api_key") or data.get("base_url")):
                data["profiles"] = [
                    {
                        "name": "默认预设",
                        "enabled": True,
                        "api_key": data.get("api_key", ""),
                        "base_url": data.get("base_url", "https://api.typesafe.ai/v1"),
                        "model": data.get("model", "jev-latest"),
                        "timeout_seconds": data.get("timeout_seconds", 3.5),
                    }
                ]
                data["active_profile"] = "默认预设"
        return data


JevApiSection.model_rebuild()


class FusionPolicySection(PluginConfigBase):
    """概率融合与静默消费策略。"""

    __ui_label__ = "融合与门控策略"
    __ui_icon__ = "activity"
    __ui_order__ = 2

    ambiguity_band_only: bool = Field(
        default=True,
        json_schema_extra={
            "label": "仅在模糊区间调用 Jev",
            "hint": "仅当本地先验概率靠近动态阈值时才请求 Jev，大幅节省额度与延迟",
        },
        description="仅在本地概率靠近阈值的模糊区间内调用 Jev（大幅节省调用次数与延迟）",
    )
    ambiguity_margin: float = Field(
        default=0.22,
        json_schema_extra={
            "label": "模糊区间范围 (半宽)",
            "step": 0.01,
        },
        description="模糊区间半宽：仅当 |P_local - threshold| <= 该值时才请求 Jev",
    )
    max_fusion_weight: float = Field(
        default=0.85,
        json_schema_extra={
            "label": "最大融合权重",
            "step": 0.05,
        },
        description="Jev 概率的最大融合权重（实际权重 alpha = confidence * max_fusion_weight）",
    )
    min_confidence: float = Field(
        default=0.20,
        json_schema_extra={
            "label": "最低生效置信度",
            "step": 0.05,
        },
        description="Jev 最低生效置信度：低于该值时完全回退本地先验概率",
    )
    enable_multi_signal: bool = Field(
        default=True,
        json_schema_extra={
            "label": "启用多维信号协同判定",
            "hint": "同时评估回复必要性、搭话自然度与打扰风险三个维度",
        },
        description="同时向 Jev 询问 natural_to_join 与 interruption_risk 两个辅助维度",
    )
    enable_silent_consume: bool = Field(
        default=True,
        json_schema_extra={
            "label": "背景闲聊静默已读 (consume)",
            "hint": "高置信度判定为背景闲聊时静默推进已读游标，消除消息积压",
        },
        description="当 Jev 高置信度判定为背景闲聊时，返回 decision=consume 静默推进已读游标",
    )
    silent_consume_prob_ceiling: float = Field(
        default=0.20,
        json_schema_extra={
            "label": "静默已读回复概率上限",
            "step": 0.05,
        },
        description="触发静默已读允许的最大 Jev 回复概率上限",
    )
    silent_consume_min_confidence: float = Field(
        default=0.72,
        json_schema_extra={
            "label": "静默已读最小置信度",
            "step": 0.02,
        },
        description="触发静默已读所需的最小 Jev 置信度",
    )
    custom_prompt: str = Field(
        default="",
        json_schema_extra={
            "label": "自定义评估提示词模板",
            "placeholder": "留空使用内置默认模板（自动注入麦麦人设与行为风格）",
            "rows": 4,
        },
        description="自定义 Jev 评估提示词模板（留空使用内置默认模板，自动注入麦麦人设与行为风格）",
    )


FusionPolicySection.model_rebuild()


class JevGateConfig(PluginConfigBase):
    """Jev 门控预判插件完整配置。"""

    plugin: PluginSection = Field(default_factory=PluginSection)
    jev: JevApiSection = Field(default_factory=JevApiSection)
    policy: FusionPolicySection = Field(default_factory=FusionPolicySection)


JevGateConfig.model_rebuild()
MaisakaJevGateConfig = JevGateConfig


class JevGatePlugin(MaiBotPlugin):
    """Jev 门控预判插件 (Jev Gate Plugin)。"""

    config_model = JevGateConfig
    config_reload_subscriptions = (ON_BOT_CONFIG_RELOAD,)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._bot_name_cache: str = ""
        self._personality_cache: str = ""
        self._behavior_style_cache: str = ""

    async def on_load(self) -> None:
        await self._refresh_bot_persona_cache()
        self.ctx.logger.info("Jev 门控预判插件已成功加载并初始化。")
        return None
    async def on_unload(self) -> None:
        return None

    async def on_config_update(self, scope: str, config_data: dict[str, object], version: str) -> None:
        del config_data, version
        if scope == ON_BOT_CONFIG_RELOAD:
            await self._refresh_bot_persona_cache()

    async def _refresh_bot_persona_cache(self) -> None:
        """从 Host 配置能力读取麦麦昵称、人格与说话风格，换人设时自动同步。"""

        try:
            self._bot_name_cache = str(await self.ctx.config.get("bot.nickname", "麦麦") or "麦麦").strip()
            self._personality_cache = str(await self.ctx.config.get("personality.personality", "") or "").strip()
            self._behavior_style_cache = str(await self.ctx.config.get("personality.reply_style", "") or "").strip()
        except Exception:
            self._bot_name_cache = self._bot_name_cache or "麦麦"

    def _build_policy_config(self) -> GatePolicyConfig:
        cfg: JevGateConfig = self.config
        return GatePolicyConfig(
            ambiguity_band_only=bool(cfg.policy.ambiguity_band_only),
            ambiguity_margin=float(cfg.policy.ambiguity_margin),
            max_fusion_weight=float(cfg.policy.max_fusion_weight),
            min_confidence=float(cfg.policy.min_confidence),
            enable_multi_signal=bool(cfg.policy.enable_multi_signal),
            enable_silent_consume=bool(cfg.policy.enable_silent_consume),
            silent_consume_prob_ceiling=float(cfg.policy.silent_consume_prob_ceiling),
            silent_consume_min_confidence=float(cfg.policy.silent_consume_min_confidence),
            shadow_mode=bool(cfg.plugin.shadow_mode),
        )

    def _resolve_active_jev_profile(self) -> tuple[str, str, str, float, str]:
        """解析当前激活状态的 Jev 配置。

        Returns:
            tuple[str, str, str, float, str]: (api_key, base_url, model, timeout_seconds, profile_name)
        """
        cfg: JevGateConfig = self.config
        jev_sec = cfg.jev
        target_name = (jev_sec.active_profile or "").strip().lower()

        # 1. 优先在 profiles 列表中按 active_profile 名称匹配
        if jev_sec.profiles:
            for p in jev_sec.profiles:
                if (p.name or "").strip().lower() == target_name:
                    return (p.api_key, p.base_url, p.model, p.timeout_seconds, p.name)

            # 2. 检查 profiles 中是否有明确启用的项 (enabled == True)
            for p in jev_sec.profiles:
                if p.enabled:
                    return (p.api_key, p.base_url, p.model, p.timeout_seconds, p.name)

            # 3. 回退使用第一个预设
            first = jev_sec.profiles[0]
            return (first.api_key, first.base_url, first.model, first.timeout_seconds, first.name)

        # 4. 终极兜底默认值
        return ("", "https://api.typesafe.ai/v1", "jev-latest", 3.5, "default")

    def _build_jev_client(self) -> tuple[JevClient, str]:
        """构建当前激活预设对应的 JevClient 实例。"""
        api_key, base_url, model, timeout_seconds, profile_name = self._resolve_active_jev_profile()
        return (
            JevClient(
                api_key=api_key,
                base_url=base_url,
                model=model,
                timeout_seconds=timeout_seconds,
            ),
            profile_name,
        )

    @HookHandler(
        "maisaka.turn_trigger.evaluate",
        name="jev_turn_trigger_gate",
        description="在 Maisaka 动态门控模糊区间内调用 TypeSafe Jev 融合回复概率或静默推进已读游标",
        mode=HookMode.BLOCKING,
        order=HookOrder.NORMAL,
        timeout_ms=5000,
        error_policy=ErrorPolicy.SKIP,
    )
    async def handle_turn_trigger_evaluate(
        self,
        session_id: str = "",
        is_group_session: bool = True,
        frequency: float = 1.0,
        probability: float = 0.0,
        threshold: float = 0.0,
        decision: str = "wait",
        detail: str = "",
        delay_seconds: float = 0.0,
        consumed_count: int = 0,
        pending_messages: list[dict[str, Any]] | None = None,
        recent_messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del session_id, frequency, delay_seconds, kwargs
        cfg: MaisakaJevGateConfig = self.config
        if not cfg.plugin.enabled or not is_group_session:
            return {"action": "continue"}

        pending_list = pending_messages if isinstance(pending_messages, list) else []
        recent_list = recent_messages if isinstance(recent_messages, list) else []
        policy = self._build_policy_config()

        should_call, skip_reason = should_call_jev(
            base_probability=probability,
            threshold=threshold,
            pending_messages=pending_list,
            policy=policy,
        )
        if not should_call:
            self.ctx.logger.info(f"门控预判决策: {skip_reason}")
            return {"action": "continue"}

        try:
            client, profile_name = self._build_jev_client()
        except JevConfigError as exc:
            self.ctx.logger.warning(f"Jev 配置未就绪，回退本地门控: {exc}")
            return {"action": "continue"}
        self.ctx.logger.info(f"命中预判规则 [{profile_name}]: {skip_reason}，正在向 Jev 发起结构化评估...")

        if not self._bot_name_cache:
            await self._refresh_bot_persona_cache()

        prompt_template = cfg.policy.custom_prompt.strip() or DEFAULT_DECISION_PROMPT
        state = build_jev_state(
            bot_name=self._bot_name_cache,
            personality=self._personality_cache,
            behavior_style=self._behavior_style_cache,
            recent_messages=recent_list,
            pending_messages=pending_list,
            base_probability=probability,
            threshold=threshold,
            context_limit=cfg.jev.context_message_count,
        )
        questions = build_jev_questions(
            bot_name=self._bot_name_cache,
            personality=self._personality_cache,
            behavior_style=self._behavior_style_cache,
            prompt_template=prompt_template,
            enable_multi_signal=policy.enable_multi_signal,
        )

        try:
            evaluation = await client.evaluate(state=state, questions=questions)
            outcome = fuse_jev_evaluation(
                evaluation=evaluation,
                base_probability=probability,
                threshold=threshold,
                base_decision=decision,
                policy=policy,
            )
        except Exception as exc:
            self.ctx.logger.warning(f"Jev 远程评估失败，安全回退本地门控判定: {exc}")
            return {"action": "continue"}

        self.ctx.logger.info(f"Jev 门控预判完成 [{profile_name}]: {outcome.detail}")
        if policy.shadow_mode:
            return {
                "action": "continue",
                "modified_kwargs": {
                    "detail": f"{detail} {outcome.detail}".strip(),
                },
            }

        modified: dict[str, Any] = {
            "probability": outcome.fused_probability,
            "threshold": outcome.threshold,
            "detail": f"{detail} {outcome.detail}".strip(),
        }
        if outcome.decision == "consume":
            modified["decision"] = "consume"
            modified["consumed_count"] = consumed_count or len(pending_list)

        return {
            "action": "continue",
            "modified_kwargs": modified,
        }


MaisakaJevGatePlugin = JevGatePlugin


def create_plugin() -> JevGatePlugin:
    """创建 Jev 门控预判插件实例。"""
    return JevGatePlugin()
