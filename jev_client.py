"""TypeSafe Jev (System One) 结构化决策客户端。

仅使用 httpx / urllib 发起异步请求，不依赖 MaiBot 主仓库内部模块。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping
import asyncio
import json
import urllib.request

try:
    import httpx
except ImportError:  # pragma: no cover - fallback to stdlib urllib in thread
    httpx = None  # type: ignore[assignment]

def resolve_evaluate_url(base_url: str) -> str:
    """自动兼容 Liquid AI (/decisions) 与 TypeSafe (/v1) 等不同格式的 Base URL。"""
    cleaned = str(base_url or "").strip().rstrip("/")
    if cleaned.endswith("/systemone"):
        return cleaned
    if cleaned.endswith("/v1"):
        return f"{cleaned}/systemone"
    return f"{cleaned}/v1/systemone"

class JevConfigError(ValueError):
    """Jev 配置缺失或不合法时抛出。"""


@dataclass(frozen=True, slots=True)
class JevAnswer:
    """单个 Jev 问题的类型化答案。"""

    answer_type: str
    choice: str = ""
    probabilities: Mapping[str, float] = field(default_factory=dict)
    noul: float = 0.0
    score: float = 0.0
    confidence: float = 0.0


@dataclass(frozen=True, slots=True)
class JevEvaluationResult:
    """一次 Jev 评估的完整结果。"""

    model: str
    answers: Mapping[str, JevAnswer]
    input_tokens: int = 0
    output_tokens: int = 0


class JevClient:
    """调用 TypeSafe Jev 结构化决策模型的轻量异步客户端。"""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.typesafe.ai/v1",
        model: str = "jev-latest",
        timeout_seconds: float = 4.0,
    ) -> None:
        cleaned_key = str(api_key or "").strip()
        if not cleaned_key:
            raise JevConfigError("尚未配置 Jev API Key")

        cleaned_url = str(base_url or "").strip().rstrip("/")
        if not cleaned_url:
            raise JevConfigError("尚未配置 Jev API Base URL")
        if not cleaned_url.lower().startswith("https://"):
            raise JevConfigError(f"Jev API 地址必须使用 https，当前为: {cleaned_url}")

        self._api_key = cleaned_key
        self._base_url = cleaned_url
        self._model = str(model or "jev-latest").strip() or "jev-latest"
        self._timeout_seconds = max(0.5, float(timeout_seconds))

    async def evaluate(
        self,
        *,
        state: str | Mapping[str, Any] | list[Any],
        questions: Mapping[str, Mapping[str, Any]],
    ) -> JevEvaluationResult:
        """把 state 与 questions 交给 Jev 评估，并返回结构化结果。"""

        request_payload: dict[str, Any] = {
            "state": state,
            "model": self._model,
            "questions": dict(questions),
        }
        url = resolve_evaluate_url(self._base_url)
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        if httpx is not None:
            async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
                response = await client.post(url, headers=headers, json=request_payload)
                response.raise_for_status()
                response_payload = response.json()
        else:
            response_payload = await asyncio.to_thread(
                self._post_with_urllib,
                url=url,
                headers=headers,
                payload=request_payload,
            )

        return parse_evaluation_payload(response_payload)

    def _post_with_urllib(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
    ) -> Any:
        raw_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url, data=raw_body, headers=dict(headers), method="POST")
        with urllib.request.urlopen(req, timeout=self._timeout_seconds) as resp:  # noqa: S310
            text = resp.read().decode("utf-8")
        return json.loads(text)


def parse_evaluation_payload(response_payload: Any) -> JevEvaluationResult:
    """解析 Jev 响应体为结构化结果。"""

    if not isinstance(response_payload, dict):
        raise ValueError(f"Jev 响应体不是 JSON 对象: {type(response_payload).__name__}")

    raw_answers = response_payload.get("answers")
    if not isinstance(raw_answers, dict) or not raw_answers:
        raise ValueError("Jev 响应体缺少 answers 字段")

    usage = response_payload.get("usage")
    usage_dict = usage if isinstance(usage, dict) else {}
    return JevEvaluationResult(
        model=str(response_payload.get("model") or ""),
        answers={
            str(question_id): _parse_answer(str(question_id), raw_answer)
            for question_id, raw_answer in raw_answers.items()
        },
        input_tokens=_read_int_token(usage_dict.get("input_tokens")),
        output_tokens=_read_int_token(usage_dict.get("output_tokens")),
    )


def _parse_answer(question_id: str, raw_answer: Any) -> JevAnswer:
    if not isinstance(raw_answer, dict):
        raise ValueError(f"Jev 问题 {question_id} 的答案不是 JSON 对象")

    answer_type = str(raw_answer.get("type") or "").strip()
    if answer_type == "choice":
        return JevAnswer(
            answer_type=answer_type,
            choice=str(raw_answer.get("choice") or ""),
            probabilities=_parse_probabilities(raw_answer.get("probabilities")),
            confidence=_read_float(raw_answer.get("confidence")),
        )
    if answer_type == "noul":
        return JevAnswer(
            answer_type=answer_type,
            noul=_read_float(raw_answer.get("noul")),
            confidence=_read_float(raw_answer.get("confidence")),
        )
    if answer_type == "score":
        return JevAnswer(
            answer_type=answer_type,
            score=_read_float(raw_answer.get("score")),
            confidence=_read_float(raw_answer.get("confidence")),
        )
    raise ValueError(f"Jev 问题 {question_id} 返回了未知答案类型: {answer_type!r}")


def _parse_probabilities(raw_probabilities: Any) -> dict[str, float]:
    if not isinstance(raw_probabilities, dict):
        return {}
    return {str(option_name): _read_float(probability) for option_name, probability in raw_probabilities.items()}


def _read_float(raw_value: Any) -> float:
    if raw_value is None:
        return 0.0
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        raise ValueError(f"Jev 响应中的数值字段类型非法: {raw_value!r}")
    return float(raw_value)


def _read_int_token(raw_value: Any) -> int:
    if raw_value is None:
        return 0
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        raise ValueError(f"Jev 响应中的 Token 字段类型非法: {raw_value!r}")
    return int(raw_value)
