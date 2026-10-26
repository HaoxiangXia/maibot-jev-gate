"""Jev 门控融合策略与多维度结构化提问构建。

解决原 PR #2070 的四大核心缺陷：
1. 概率与置信度融合：不再把 Jev 当成二值开关，而是用 confidence 加权融合 P_jev 与 P_local：
   P_fused = alpha * P_jev + (1 - alpha) * P_local，继续交由本地 DynamicReplyGate 与 talk_value 阈值裁决。
2. 静默已读游标推进（consume）：当 Jev 高置信度判定当前消息为纯背景噪音/他人私下对话且无需等待下文时，
   返回 decision="consume" 静默推进 _last_processed_index，从根源消除 100 条积压泄洪。
3. 模糊区间选择性调用（Ambiguity Band）：仅在 |P_local - threshold| <= ambiguity_margin 时才请求 Jev，
   明显该回或明显不该回的轮次零远程开销、零额外延迟。
4. 失败优雅回退：Jev 超时或报错时直接沿用本地 DynamicReplyGate 的原判定，不阻塞、不丢消息。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .jev_client import JevAnswer, JevEvaluationResult

DEFAULT_DECISION_PROMPT = """你是{bot_name}的群聊社交直觉评估器，只负责评估{bot_name}此刻是否适合参与群聊，不直接生成回复正文。

{bot_name}的人设：
{personality}

{bot_name}的说话与行动风格：
{behavior_style}

输入说明：
- recent_messages：近期群聊上下文历史。
- pending_messages：刚收到的待评估新消息，at_bot=true 表示直接 @ 了{bot_name}。
- local_gate：宿主本地逻辑回归门控给出的先验回复概率 base_probability 与当前放行阈值 threshold。

判断原则：
- 若新消息在提问、点名、接{bot_name}刚才的话茬，或话题是{bot_name}感兴趣/能自然搭腔的公共话题，倾向于 reply。
- 若群友正在进行连续的上下文铺垫（话刚说一半），倾向于 wait（等待更多消息再看）。
- 若消息明显是其他群友之间的私密/特定两人对话、无意义刷屏、纯表情符号，且后续也无需{bot_name}插嘴，倾向于 consume_silent（标为已读不回）。
"""


@dataclass(frozen=True, slots=True)
class GatePolicyConfig:
    """门控融合策略参数。"""

    ambiguity_band_only: bool = True
    ambiguity_margin: float = 0.22
    max_fusion_weight: float = 0.85
    min_confidence: float = 0.20
    enable_multi_signal: bool = True
    enable_silent_consume: bool = True
    silent_consume_prob_ceiling: float = 0.20
    silent_consume_min_confidence: float = 0.72
    shadow_mode: bool = False


@dataclass(frozen=True, slots=True)
class FusedGateOutcome:
    """Jev 评估与本地概率融合后的输出。"""

    called_jev: bool
    decision: str
    fused_probability: float
    threshold: float
    jev_reply_probability: float
    jev_confidence: float
    jev_action: str
    detail: str


def should_call_jev(
    *,
    base_probability: float,
    threshold: float,
    pending_messages: Sequence[Mapping[str, Any]],
    policy: GatePolicyConfig,
) -> tuple[bool, str]:
    """判断当前批次是否需要发起远程 Jev 请求。"""

    if not pending_messages:
        return False, "无待处理外部消息"

    for msg in pending_messages:
        if msg.get("is_at") or msg.get("at_bot"):
            return False, "存在直接 @ 消息，走宿主强制触发豁免"

    if not policy.ambiguity_band_only:
        return True, "全量评估模式"

    distance = abs(float(base_probability) - float(threshold))
    if distance <= max(0.0, float(policy.ambiguity_margin)):
        return True, f"命中模糊区间(|{base_probability:.2f}-{threshold:.2f}|={distance:.2f}<={policy.ambiguity_margin:.2f})"

    return False, f"超出模糊区间(|{base_probability:.2f}-{threshold:.2f}|={distance:.2f}>{policy.ambiguity_margin:.2f})，沿用本地门控"


def build_jev_state(
    *,
    bot_name: str,
    personality: str,
    behavior_style: str,
    recent_messages: Sequence[Mapping[str, Any]],
    pending_messages: Sequence[Mapping[str, Any]],
    base_probability: float,
    threshold: float,
    context_limit: int = 12,
    pending_limit: int = 30,
) -> dict[str, Any]:
    """构建发往 TypeSafe Jev 的结构化状态对象。"""

    normalized_recent = []
    for item in list(recent_messages)[-max(1, int(context_limit)) :]:
        text = str(item.get("text") or item.get("processed_plain_text") or "").strip()
        if not text:
            continue
        normalized_recent.append(
            {
                "sender": str(item.get("sender") or item.get("sender_name") or ""),
                "text": text,
                "is_self": bool(item.get("is_self", False)),
            }
        )

    normalized_pending = []
    for msg in list(pending_messages)[-max(1, int(pending_limit)) :]:
        text = str(msg.get("processed_plain_text") or msg.get("text") or "").strip()
        user_info = ((msg.get("message_info") or {}).get("user_info") or {}) if isinstance(msg.get("message_info"), dict) else {}
        sender = (
            msg.get("sender")
            or msg.get("sender_name")
            or user_info.get("user_cardname")
            or user_info.get("user_nickname")
            or user_info.get("user_id")
            or ""
        )
        normalized_pending.append(
            {
                "sender": str(sender),
                "text": text,
                "at_bot": bool(msg.get("is_at") or msg.get("at_bot")),
                "mentioned_bot": bool(msg.get("is_mentioned") or msg.get("mentioned_bot")),
            }
        )

    return {
        "bot_name": bot_name or "麦麦",
        "persona": {
            "personality": personality,
            "behavior_style": behavior_style,
        },
        "local_gate": {
            "base_probability": round(float(base_probability), 4),
            "threshold": round(float(threshold), 4),
        },
        "recent_messages": normalized_recent,
        "pending_messages": normalized_pending,
    }


def build_jev_questions(
    *,
    bot_name: str,
    personality: str,
    behavior_style: str,
    prompt_template: str = DEFAULT_DECISION_PROMPT,
    enable_multi_signal: bool = True,
) -> dict[str, dict[str, Any]]:
    """构造 Jev 结构化问题表（支持主动作 choice + 辅助维度 noul）。"""

    rendered_prompt = prompt_template.format(
        bot_name=bot_name or "麦麦",
        personality=personality or "友善、自然的群聊伙伴",
        behavior_style=behavior_style or "像真人一样自然接话，不刷屏、不强行插嘴。",
    ).strip()

    criteria = {
        "reply": "现在进入规划器思考并参与发言（被提问、接话自然、有梗或能提供有效信息）",
        "wait": "暂时潜水等待后续消息（话题刚起头、上下文未说完整，稍后再看）",
        "consume_silent": "静默已读并跳过本批消息（纯粹是他人之间的私下交流、无意义刷屏或完全接不上话）",
    }
    questions: dict[str, dict[str, Any]] = {
        "reply_action": {
            "type": "choice",
            "instructions": rendered_prompt,
            "criteria": criteria,
        }
    }

    if enable_multi_signal:
        questions["natural_to_join"] = {
            "type": "noul",
            "instructions": f"从群聊气氛看，{bot_name or '麦麦'}此刻插话是否自然且不突兀？",
        }
        questions["interruption_risk"] = {
            "type": "noul",
            "instructions": f"{bot_name or '麦麦'}此刻发言是否会打断别人正在进行的严肃或私密两人对话？",
        }
    return questions


def fuse_jev_evaluation(
    *,
    evaluation: JevEvaluationResult,
    base_probability: float,
    threshold: float,
    base_decision: str,
    policy: GatePolicyConfig,
) -> FusedGateOutcome:
    """将 Jev 返回的概率分布、置信度与多维信号融合为最终门控决策。"""

    action_ans: JevAnswer | None = evaluation.answers.get("reply_action")
    if action_ans is None or action_ans.answer_type != "choice":
        raise ValueError("Jev 响应缺少有效的 reply_action choice 答案")

    probs = action_ans.probabilities or {}
    p_reply = float(probs.get("reply", 1.0 if action_ans.choice == "reply" else 0.0))
    p_wait = float(probs.get("wait", 1.0 if action_ans.choice == "wait" else 0.0))
    p_consume = float(probs.get("consume_silent", 1.0 if action_ans.choice == "consume_silent" else 0.0))
    confidence = min(1.0, max(0.0, float(action_ans.confidence)))

    # 结合多维辅助信号微调 p_reply
    p_jev = p_reply
    natural_ans = evaluation.answers.get("natural_to_join")
    interrupt_ans = evaluation.answers.get("interruption_risk")
    if natural_ans is not None and natural_ans.answer_type == "noul":
        p_jev = 0.75 * p_jev + 0.25 * float(natural_ans.noul)
    if interrupt_ans is not None and interrupt_ans.answer_type == "noul":
        interrupt_penalty = float(interrupt_ans.noul) * max(0.3, float(interrupt_ans.confidence))
        p_jev = p_jev * (1.0 - 0.35 * interrupt_penalty)

    p_jev = min(1.0, max(0.0, p_jev))

    # 置信度过低时完全回退本地概率
    if confidence < policy.min_confidence:
        alpha = 0.0
    else:
        alpha = min(1.0, max(0.0, confidence * float(policy.max_fusion_weight)))

    fused_prob = min(1.0, max(0.0, alpha * p_jev + (1.0 - alpha) * float(base_probability)))

    # Shadow Mode 仅记录日志，不改变实际决策与概率
    if policy.shadow_mode:
        detail = (
            f"[Jev-Shadow action={action_ans.choice} p_jev={p_jev:.2f} conf={confidence:.2f} "
            f"p_fused={fused_prob:.2f} (未生效)]"
        )
        return FusedGateOutcome(
            called_jev=True,
            decision=base_decision,
            fused_probability=float(base_probability),
            threshold=float(threshold),
            jev_reply_probability=p_jev,
            jev_confidence=confidence,
            jev_action=action_ans.choice,
            detail=detail,
        )

    # 判断是否触发静默已读（consume）：高置信度认为应当静默跳过，且融合后概率远低于阈值
    should_silent_consume = (
        policy.enable_silent_consume
        and confidence >= policy.silent_consume_min_confidence
        and (
            action_ans.choice == "consume_silent"
            or (p_consume >= max(p_reply, p_wait) and p_jev <= policy.silent_consume_prob_ceiling)
        )
        and fused_prob < float(threshold)
    )

    if should_silent_consume:
        decision = "consume"
    elif fused_prob >= float(threshold):
        decision = "trigger"
    else:
        decision = "wait"

    detail = (
        f"[Jev action={action_ans.choice} p_local={base_probability:.2f}->p_fused={fused_prob:.2f} "
        f"(p_jev={p_jev:.2f}, conf={confidence:.2f}, alpha={alpha:.2f}) 阈值={threshold:.2f} => {decision}]"
    )
    return FusedGateOutcome(
        called_jev=True,
        decision=decision,
        fused_probability=fused_prob,
        threshold=float(threshold),
        jev_reply_probability=p_jev,
        jev_confidence=confidence,
        jev_action=action_ans.choice,
        detail=detail,
    )
