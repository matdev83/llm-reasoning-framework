import unittest
from src.llm_config import LLMConfig

class TestLLMConfig(unittest.TestCase):

    def test_default_initialization(self):
        config = LLMConfig()
        self.assertEqual(config.temperature, 1.0)
        self.assertEqual(config.top_p, 1.0)
        self.assertEqual(config.top_k, 0)
        self.assertEqual(config.frequency_penalty, 0.0)
        self.assertEqual(config.presence_penalty, 0.0)
        self.assertEqual(config.repetition_penalty, 1.0)
        self.assertIsNone(config.seed)
        self.assertIsNone(config.max_tokens)
        self.assertIsNone(config.stop)
        self.assertIsNone(config.logit_bias)
        self.assertIsNone(config.response_format)
        self.assertEqual(config.reasoning_effort, "high")
        self.assertEqual(config.provider_specific_params, {})

    def test_custom_initialization(self):
        config = LLMConfig(
            temperature=0.5,
            top_p=0.9,
            top_k=50,
            max_tokens=100,
            seed=123,
            reasoning_effort="low",
            provider_specific_params={"custom_key": "custom_value"}
        )
        self.assertEqual(config.temperature, 0.5)
        self.assertEqual(config.top_p, 0.9)
        self.assertEqual(config.top_k, 50)
        self.assertEqual(config.max_tokens, 100)
        self.assertEqual(config.seed, 123)
        self.assertEqual(config.reasoning_effort, "low")
        self.assertEqual(config.provider_specific_params, {"custom_key": "custom_value"})

    def test_to_payload_dict_defaults(self):
        config = LLMConfig()
        payload = config.to_payload_dict()
        expected_payload = {
            "temperature": 1.0,
            "top_p": 1.0,
            # top_k is 0, so not included
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "repetition_penalty": 1.0,
            "reasoning": {"effort": "high"}
        }
        self.assertEqual(payload, expected_payload)

    def test_to_payload_dict_custom_values(self):
        config = LLMConfig(
            temperature=0.7,
            top_p=0.8,
            top_k=20, # Should be included
            frequency_penalty=0.1,
            presence_penalty=-0.1,
            repetition_penalty=1.2,
            seed=42,
            max_tokens=200,
            stop=["\n", "stop"],
            logit_bias={"123": 0.5, "456": -0.5},
            response_format={"type": "json_object"},
            reasoning_effort="medium",
            provider_specific_params={"ext_param": "ext_val"}
        )
        payload = config.to_payload_dict()
        expected_payload = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "frequency_penalty": 0.1,
            "presence_penalty": -0.1,
            "repetition_penalty": 1.2,
            "seed": 42,
            "max_tokens": 200,
            "stop": ["\n", "stop"],
            "logit_bias": {"123": 0.5, "456": -0.5},
            "response_format": {"type": "json_object"},
            "reasoning": {"effort": "medium"},
            "ext_param": "ext_val"
        }
        self.assertEqual(payload, expected_payload)

    def test_to_payload_dict_none_values(self):
        config = LLMConfig(
            temperature=None, # Explicitly None
            top_p=None,
            top_k=0, # Default, not included
            frequency_penalty=None,
            presence_penalty=None,
            repetition_penalty=None, # Explicitly None, different from default 1.0
            reasoning_effort=None,
            provider_specific_params={}
        )
        payload = config.to_payload_dict()
        # Even if set to None, if the default in to_payload_dict isn't None, it might get included
        # The current to_payload_dict() only includes if not None, or if it's a default like top_k=0
        # repetition_penalty default is 1.0. If set to None, it should be excluded by "if self.repetition_penalty is not None"
        expected_payload = {
            # No temperature, top_p, etc.
            # top_k = 0 is not included
        }
        # Check what is actually produced for Nones
        # Based on current LLMConfig.to_payload_dict():
        # temperature=None -> not in payload
        # top_p=None -> not in payload
        # top_k=0 -> not in payload
        # frequency_penalty=None -> not in payload
        # presence_penalty=None -> not in payload
        # repetition_penalty=None -> not in payload.
        # reasoning_effort=None -> "reasoning" key not in payload.
        self.assertEqual(payload, expected_payload)

    def test_to_payload_dict_top_k_zero(self):
        config = LLMConfig(top_k=0)
        payload = config.to_payload_dict()
        self.assertNotIn("top_k", payload)

        config = LLMConfig(top_k=1)
        payload = config.to_payload_dict()
        self.assertIn("top_k", payload)
        self.assertEqual(payload["top_k"], 1)

    def test_to_payload_dict_repetition_penalty_default_vs_custom(self):
        # Default repetition_penalty is 1.0
        config_default = LLMConfig()
        payload_default = config_default.to_payload_dict()
        self.assertEqual(payload_default["repetition_penalty"], 1.0)

        # Custom repetition_penalty
        config_custom = LLMConfig(repetition_penalty=1.5)
        payload_custom = config_custom.to_payload_dict()
        self.assertEqual(payload_custom["repetition_penalty"], 1.5)

        # Explicitly None repetition_penalty (should not be in payload)
        config_none = LLMConfig(repetition_penalty=None)
        payload_none = config_none.to_payload_dict()
        self.assertNotIn("repetition_penalty", payload_none)


if __name__ == '__main__':
    unittest.main()
