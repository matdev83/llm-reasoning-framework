import unittest

from src.l2t.prompt_generator import L2TPromptGenerator
from src.l2t.dataclasses import L2TConfig
from src.l2t.constants import DEFAULT_L2T_X_FMT_DEFAULT, DEFAULT_L2T_X_EVA_DEFAULT


class TestL2TPromptGenerator(unittest.TestCase):
    def setUp(self):
        self.config = L2TConfig()
        self.prompt_generator = L2TPromptGenerator(l2t_config=self.config)

    def test_construct_l2t_initial_prompt_default_templates(self):
        problem_text = "What is the meaning of life?"
        prompt = self.prompt_generator.construct_l2t_initial_prompt(problem_text)

        self.assertIn(problem_text, prompt)
        self.assertIn(self.config.x_fmt_default, prompt)
        self.assertIn(self.config.x_eva_default, prompt)
        # Keywords from conf/prompts/l2t_initial.txt
        self.assertIn("Problem:", prompt)
        self.assertIn("You are starting a step-by-step reasoning process.", prompt)
        self.assertIn("Format constraints for each thought:", prompt)
        self.assertIn("Evaluation criteria for each thought:", prompt)
        self.assertIn("Provide your very first thought", prompt)
        self.assertIn("Your thought:", prompt)

    def test_construct_l2t_initial_prompt_custom_templates(self):
        problem_text = "Solve x + 5 = 10"
        custom_fmt = "Be very precise."
        custom_eva = "Check your math."
        prompt = self.prompt_generator.construct_l2t_initial_prompt(
            problem_text, x_fmt=custom_fmt, x_eva=custom_eva
        )

        self.assertIn(problem_text, prompt)
        self.assertIn(custom_fmt, prompt)
        self.assertIn(custom_eva, prompt)
        self.assertNotIn(self.config.x_fmt_default, prompt)
        self.assertNotIn(self.config.x_eva_default, prompt)
        self.assertIn("Your thought:", prompt)

    def test_construct_l2t_node_classification_prompt_default_template(self):
        graph_context = "Path: A -> B. Current node: B."
        node_content = "This is thought B."
        prompt = self.prompt_generator.construct_l2t_node_classification_prompt(
            graph_context, node_content
        )

        self.assertIn(graph_context, prompt)
        self.assertIn(node_content, prompt)
        self.assertIn(self.config.x_eva_default, prompt)
        # Keywords from conf/prompts/l2t_node_classification.txt
        self.assertIn("Current reasoning context:", prompt)
        self.assertIn("Thought to classify:", prompt)
        self.assertIn("Evaluation criteria for thoughts:", prompt)
        self.assertIn("classify this thought. Choose one of the following categories:", prompt)
        self.assertIn("- CONTINUE:", prompt)
        self.assertIn("- TERMINATE_BRANCH:", prompt)
        self.assertIn("- FINAL_ANSWER:", prompt)
        self.assertIn("- BACKTRACK:", prompt)
        self.assertIn("Your classification (must be one of the exact category names", prompt)

    def test_construct_l2t_node_classification_prompt_custom_template(self):
        graph_context = "Path: X -> Y."
        node_content = "This is thought Y."
        custom_eva = "Is it useful?"
        prompt = self.prompt_generator.construct_l2t_node_classification_prompt(
            graph_context, node_content, x_eva=custom_eva
        )

        self.assertIn(graph_context, prompt)
        self.assertIn(node_content, prompt)
        self.assertIn(custom_eva, prompt)
        self.assertNotIn(self.config.x_eva_default, prompt)
        self.assertIn("Your classification", prompt)


    def test_construct_l2t_thought_generation_prompt_default_templates(self):
        graph_context = "Current thought is A, which was good."
        parent_node_content = "Thought A: The sky is blue."
        prompt = self.prompt_generator.construct_l2t_thought_generation_prompt(
            graph_context, parent_node_content
        )

        self.assertIn(graph_context, prompt)
        self.assertIn(parent_node_content, prompt)
        self.assertIn(self.config.x_fmt_default, prompt)
        self.assertIn(self.config.x_eva_default, prompt)
        # Keywords from conf/prompts/l2t_thought_generation.txt
        self.assertIn("Current reasoning context:", prompt)
        self.assertIn("Parent thought (classified as CONTINUE):", prompt)
        self.assertIn("Format constraints for your new thought:", prompt)
        self.assertIn("Evaluation criteria for your new thought:", prompt)
        self.assertIn("Generate the next single thought", prompt)
        self.assertIn("Your new thought:", prompt)

    def test_construct_l2t_thought_generation_prompt_custom_templates(self):
        graph_context = "Current thought is B."
        parent_node_content = "Thought B: Water is wet."
        custom_fmt = "Be creative."
        custom_eva = "Is it novel?"
        prompt = self.prompt_generator.construct_l2t_thought_generation_prompt(
            graph_context, parent_node_content, x_fmt=custom_fmt, x_eva=custom_eva
        )

        self.assertIn(graph_context, prompt)
        self.assertIn(parent_node_content, prompt)
        self.assertIn(custom_fmt, prompt)
        self.assertIn(custom_eva, prompt)
        self.assertNotIn(self.config.x_fmt_default, prompt)
        self.assertNotIn(self.config.x_eva_default, prompt)
        self.assertIn("Your new thought:", prompt)

if __name__ == "__main__":
    unittest.main()
