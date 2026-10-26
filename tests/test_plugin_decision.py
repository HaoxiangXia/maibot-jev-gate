import unittest

from jev_gate.decision import (
    GatePolicyConfig,
    build_jev_questions,
    build_jev_state,
    fuse_jev_evaluation,
    should_call_jev,
)
from jev_gate.jev_client import JevConfigError, JevClient, parse_evaluation_payload
from jev_gate.plugin import JevGateConfig, JevGatePlugin, JevProfileConfig


class TestMaisakaJevGateDecision(unittest.TestCase):
    def test_ambiguity_band_gating(self) -> None:
        policy = GatePolicyConfig(ambiguity_band_only=True, ambiguity_margin=0.22)
        pending = [{"message_id": "1", "processed_plain_text": "大家觉得呢", "is_at": False}]

        # Inside band: |0.42 - 0.55| = 0.13 <= 0.22 -> call Jev
        should_call, reason = should_call_jev(
            base_probability=0.42,
            threshold=0.55,
            pending_messages=pending,
            policy=policy,
        )
        self.assertTrue(should_call, reason)

        # Outside band: |0.10 - 0.55| = 0.45 > 0.22 -> skip Jev
        should_call_far, _ = should_call_jev(
            base_probability=0.10,
            threshold=0.55,
            pending_messages=pending,
            policy=policy,
        )
        self.assertFalse(should_call_far)

        # Direct @ -> skip Jev (handled by forced turn)
        should_call_at, _ = should_call_jev(
            base_probability=0.50,
            threshold=0.55,
            pending_messages=[{"message_id": "2", "processed_plain_text": "@麦麦", "is_at": True}],
            policy=policy,
        )
        self.assertFalse(should_call_at)

    def test_probability_fusion_triggers_when_jev_confident_reply(self) -> None:
        payload = {
            "model": "jev-latest",
            "answers": {
                "reply_action": {
                    "type": "choice",
                    "choice": "reply",
                    "probabilities": {"reply": 0.90, "wait": 0.08, "consume_silent": 0.02},
                    "confidence": 0.85,
                },
                "natural_to_join": {"type": "noul", "noul": 0.92, "confidence": 0.80},
                "interruption_risk": {"type": "noul", "noul": 0.05, "confidence": 0.80},
            },
            "usage": {"input_tokens": 120, "output_tokens": 15},
        }
        eval_res = parse_evaluation_payload(payload)
        policy = GatePolicyConfig()
        outcome = fuse_jev_evaluation(
            evaluation=eval_res,
            base_probability=0.40,
            threshold=0.55,
            base_decision="wait",
            policy=policy,
        )
        self.assertEqual(outcome.decision, "trigger")
        self.assertGreaterEqual(outcome.fused_probability, 0.55)

    def test_silent_consume_when_jev_confident_background_noise(self) -> None:
        payload = {
            "model": "jev-latest",
            "answers": {
                "reply_action": {
                    "type": "choice",
                    "choice": "consume_silent",
                    "probabilities": {"reply": 0.03, "wait": 0.12, "consume_silent": 0.85},
                    "confidence": 0.88,
                },
                "natural_to_join": {"type": "noul", "noul": 0.05, "confidence": 0.85},
                "interruption_risk": {"type": "noul", "noul": 0.80, "confidence": 0.85},
            },
        }
        eval_res = parse_evaluation_payload(payload)
        policy = GatePolicyConfig()
        outcome = fuse_jev_evaluation(
            evaluation=eval_res,
            base_probability=0.38,
            threshold=0.50,
            base_decision="wait",
            policy=policy,
        )
        self.assertEqual(outcome.decision, "consume")
        self.assertLess(outcome.fused_probability, 0.50)

    def test_low_confidence_falls_back_to_local_probability(self) -> None:
        payload = {
            "model": "jev-latest",
            "answers": {
                "reply_action": {
                    "type": "choice",
                    "choice": "reply",
                    "probabilities": {"reply": 0.95, "wait": 0.05, "consume_silent": 0.0},
                    "confidence": 0.10,  # Below min_confidence=0.20
                }
            },
        }
        eval_res = parse_evaluation_payload(payload)
        policy = GatePolicyConfig(min_confidence=0.20, enable_multi_signal=False)
        outcome = fuse_jev_evaluation(
            evaluation=eval_res,
            base_probability=0.35,
            threshold=0.50,
            base_decision="wait",
            policy=policy,
        )
        self.assertAlmostEqual(outcome.fused_probability, 0.35)
        self.assertEqual(outcome.decision, "wait")

    def test_client_https_validation(self) -> None:
        with self.assertRaises(JevConfigError):
            JevClient(api_key="sk-test", base_url="http://api.typesafe.ai/v1")

    def test_state_and_questions_builder(self) -> None:
        state = build_jev_state(
            bot_name="麦麦",
            personality="活泼",
            behavior_style="简短",
            recent_messages=[{"sender": "A", "text": "早", "is_self": False}],
            pending_messages=[{"sender": "B", "processed_plain_text": "今天好冷", "is_at": False}],
            base_probability=0.45,
            threshold=0.50,
        )
        self.assertEqual(state["bot_name"], "麦麦")
        self.assertEqual(len(state["pending_messages"]), 1)
        questions = build_jev_questions(
            bot_name="麦麦",
            personality="活泼",
            behavior_style="简短",
            enable_multi_signal=True,
        )
        self.assertIn("reply_action", questions)
        self.assertIn("natural_to_join", questions)
        self.assertIn("interruption_risk", questions)

    def test_multi_profile_switching(self) -> None:
        plugin = JevGatePlugin()
        plugin._plugin_config_instance = JevGateConfig.model_validate({
            "jev": {
                "active_profile": "服务商B",
                "profiles": [
                    {
                        "name": "服务商A",
                        "api_key": "key_a",
                        "base_url": "https://api.a.com/v1",
                        "model": "model-a",
                        "timeout_seconds": 2.0,
                    },
                    {
                        "name": "服务商B",
                        "api_key": "key_b",
                        "base_url": "https://api.b.com/v1",
                        "model": "model-b",
                        "timeout_seconds": 4.0,
                    },
                ],
            }
        })
        resolved = plugin._resolve_active_jev_profile()
        self.assertEqual(resolved, ("key_b", "https://api.b.com/v1", "model-b", 4.0, "服务商B"))

        # 切换至服务商A
        plugin.config.jev.active_profile = "服务商A"
        switched = plugin._resolve_active_jev_profile()
        self.assertEqual(switched, ("key_a", "https://api.a.com/v1", "model-a", 2.0, "服务商A"))

    def test_multi_profile_fallback_to_first_when_not_found(self) -> None:
        plugin = JevGatePlugin()
        plugin._plugin_config_instance = JevGateConfig.model_validate({
            "jev": {
                "active_profile": "不存在的预设",
                "profiles": [
                    {
                        "name": "默认预设",
                        "api_key": "default_key",
                        "base_url": "https://api.typesafe.ai/v1",
                        "model": "jev-latest",
                        "timeout_seconds": 3.5,
                    }
                ],
            }
        })
        resolved = plugin._resolve_active_jev_profile()
        self.assertEqual(resolved, ("default_key", "https://api.typesafe.ai/v1", "jev-latest", 3.5, "默认预设"))

    def test_legacy_flat_config_fallback(self) -> None:
        plugin = JevGatePlugin()
        plugin._plugin_config_instance = JevGateConfig.model_validate({
            "jev": {
                "api_key": "legacy_key",
                "base_url": "https://legacy.com/v1",
                "model": "legacy-jev",
                "timeout_seconds": 5.0,
            }
        })
        resolved = plugin._resolve_active_jev_profile()
        self.assertEqual(resolved, ("legacy_key", "https://legacy.com/v1", "legacy-jev", 5.0, "默认预设"))


if __name__ == "__main__":
    unittest.main()
