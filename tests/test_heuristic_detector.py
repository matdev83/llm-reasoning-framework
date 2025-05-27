import unittest
import os
from src.heuristic_detector import HeuristicDetector

class TestHeuristicDetector(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.oneshot_prompts_dir = "conf/tests/prompts/heuristic/oneshot"
        cls.complex_prompts_dir = "conf/tests/prompts/heuristic/complex"
        cls.oneshot_prompts = []
        cls.complex_prompts = []

        for i in range(1, 21):
            oneshot_file_path = os.path.join(cls.oneshot_prompts_dir, f"prompt_{i:02d}.txt")
            with open(oneshot_file_path, "r") as f:
                cls.oneshot_prompts.append(f.read().strip())

            complex_file_path = os.path.join(cls.complex_prompts_dir, f"prompt_{i:02d}.txt")
            with open(complex_file_path, "r") as f:
                cls.complex_prompts.append(f.read().strip())

    def test_oneshot_prompts_do_not_trigger_complex_process(self):
        for i, prompt in enumerate(self.oneshot_prompts):
            with self.subTest(f"One-shot Prompt {i+1}"):
                self.assertFalse(
                    HeuristicDetector.should_trigger_complex_process_heuristically(prompt),
                    f"One-shot Prompt '{prompt}' unexpectedly triggered complex process heuristic."
                )

    def test_complex_prompts_do_trigger_complex_process(self):
        for i, prompt in enumerate(self.complex_prompts):
            with self.subTest(f"Complex Prompt {i+1}"):
                self.assertTrue(
                    HeuristicDetector.should_trigger_complex_process_heuristically(prompt),
                    f"Complex Prompt '{prompt}' unexpectedly did NOT trigger complex process heuristic."
                )

if __name__ == '__main__':
    unittest.main()
