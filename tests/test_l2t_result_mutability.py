import unittest
from unittest.mock import MagicMock
from src.l2t.dataclasses import L2TResult
from src.aot.dataclasses import LLMCallStats

class TestL2TResultMutability(unittest.TestCase):
    def test_l2t_result_update_in_mock(self):
        result_obj = L2TResult()
        stats = LLMCallStats(completion_tokens=1, prompt_tokens=1, call_duration_seconds=0.01)

        def mock_update_effect(res, st):
            res.total_llm_calls += 1
            res.total_completion_tokens += st.completion_tokens
            res.total_prompt_tokens += st.prompt_tokens
            res.total_llm_interaction_time_seconds += st.call_duration_seconds
            print(f"Inside mock: res.total_completion_tokens = {res.total_completion_tokens}")
            print(f"Inside mock: id(res) = {id(res)}")

        mock_function = MagicMock(side_effect=mock_update_effect)

        print(f"Before mock call: result_obj.total_completion_tokens = {result_obj.total_completion_tokens}")
        print(f"Before mock call: id(result_obj) = {id(result_obj)}")

        mock_function(result_obj, stats)

        print(f"After mock call: result_obj.total_completion_tokens = {result_obj.total_completion_tokens}")
        print(f"After mock call: id(result_obj) = {id(result_obj)}")

        self.assertEqual(result_obj.total_completion_tokens, 1)
