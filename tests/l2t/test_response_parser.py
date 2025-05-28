import unittest
import logging

from src.l2t.response_parser import L2TResponseParser
from src.l2t.dataclasses import L2TNodeCategory

# Suppress logging output during tests for cleaner test results
logging.disable(logging.CRITICAL)


class TestL2TResponseParser(unittest.TestCase):
    def test_parse_l2t_initial_response_success(self):
        response_text = "Some preamble.\nYour thought: This is the actual thought."
        expected_thought = "This is the actual thought."
        self.assertEqual(
            L2TResponseParser.parse_l2t_initial_response(response_text),
            expected_thought,
        )

    def test_parse_l2t_initial_response_success_case_insensitive(self):
        response_text = "YOUR THOUGHT: Case insensitive test."
        expected_thought = "Case insensitive test."
        self.assertEqual(
            L2TResponseParser.parse_l2t_initial_response(response_text),
            expected_thought,
        )

    def test_parse_l2t_initial_response_success_no_preamble(self):
        response_text = "Your thought: Direct thought."
        expected_thought = "Direct thought."
        self.assertEqual(
            L2TResponseParser.parse_l2t_initial_response(response_text),
            expected_thought,
        )
    
    def test_parse_l2t_initial_response_success_extra_whitespace(self):
        response_text = "Your thought:   Thought with spaces.   "
        expected_thought = "Thought with spaces."
        self.assertEqual(
            L2TResponseParser.parse_l2t_initial_response(response_text),
            expected_thought,
        )

    def test_parse_l2t_initial_response_keyword_not_found(self):
        response_text = "This response does not contain the keyword."
        self.assertIsNone(
            L2TResponseParser.parse_l2t_initial_response(response_text)
        )

    def test_parse_l2t_initial_response_keyword_present_no_thought(self):
        response_text = "Your thought:"
        self.assertIsNone(
            L2TResponseParser.parse_l2t_initial_response(response_text)
        )

    def test_parse_l2t_initial_response_empty_string(self):
        response_text = ""
        self.assertIsNone(
            L2TResponseParser.parse_l2t_initial_response(response_text)
        )

    def test_parse_l2t_node_classification_response_success_all_categories(self):
        categories = {
            "CONTINUE": L2TNodeCategory.CONTINUE,
            "TERMINATE_BRANCH": L2TNodeCategory.TERMINATE_BRANCH,
            "FINAL_ANSWER": L2TNodeCategory.FINAL_ANSWER,
            "BACKTRACK": L2TNodeCategory.BACKTRACK,
        }
        for cat_str, cat_enum in categories.items():
            response_text = f"Some text before. Your classification: {cat_str}"
            with self.subTest(category=cat_str):
                self.assertEqual(
                    L2TResponseParser.parse_l2t_node_classification_response(
                        response_text
                    ),
                    cat_enum,
                )

    def test_parse_l2t_node_classification_response_success_case_insensitive_marker(self):
        response_text = "YOUR CLASSIFICATION: CONTINUE"
        self.assertEqual(
            L2TResponseParser.parse_l2t_node_classification_response(response_text),
            L2TNodeCategory.CONTINUE,
        )
    
    def test_parse_l2t_node_classification_response_success_category_case_insensitive(self):
        response_text = "Your classification: continue"
        self.assertEqual(
            L2TResponseParser.parse_l2t_node_classification_response(response_text),
            L2TNodeCategory.CONTINUE,
        )

    def test_parse_l2t_node_classification_response_success_extra_whitespace(self):
        response_text = "Your classification:  FINAL_ANSWER  "
        self.assertEqual(
            L2TResponseParser.parse_l2t_node_classification_response(response_text),
            L2TNodeCategory.FINAL_ANSWER,
        )
    
    def test_parse_l2t_node_classification_response_success_with_reasoning(self):
        response_text = "Your classification: TERMINATE_BRANCH because it's a dead end."
        self.assertEqual(
            L2TResponseParser.parse_l2t_node_classification_response(response_text),
            L2TNodeCategory.TERMINATE_BRANCH,
        )

    def test_parse_l2t_node_classification_response_invalid_category(self):
        response_text = "Your classification: INVALID_CATEGORY"
        self.assertIsNone(
            L2TResponseParser.parse_l2t_node_classification_response(
                response_text
            )
        )

    def test_parse_l2t_node_classification_response_keyword_not_found(self):
        response_text = "No classification here."
        self.assertIsNone(
            L2TResponseParser.parse_l2t_node_classification_response(
                response_text
            )
        )

    def test_parse_l2t_node_classification_response_keyword_present_no_category(self):
        response_text = "Your classification:"
        self.assertIsNone(
            L2TResponseParser.parse_l2t_node_classification_response(
                response_text
            )
        )
    
    def test_parse_l2t_node_classification_response_empty_string(self):
        response_text = ""
        self.assertIsNone(
            L2TResponseParser.parse_l2t_node_classification_response(response_text)
        )

    def test_parse_l2t_thought_generation_response_success(self):
        response_text = "Preamble. Your new thought: This is the new generated thought."
        expected_thought = "This is the new generated thought."
        self.assertEqual(
            L2TResponseParser.parse_l2t_thought_generation_response(response_text),
            expected_thought,
        )
    
    def test_parse_l2t_thought_generation_response_success_case_insensitive(self):
        response_text = "YOUR NEW THOUGHT: New thought, case insensitive."
        expected_thought = "New thought, case insensitive."
        self.assertEqual(
            L2TResponseParser.parse_l2t_thought_generation_response(response_text),
            expected_thought,
        )

    def test_parse_l2t_thought_generation_response_success_no_preamble(self):
        response_text = "Your new thought: Direct new thought."
        expected_thought = "Direct new thought."
        self.assertEqual(
            L2TResponseParser.parse_l2t_thought_generation_response(response_text),
            expected_thought,
        )

    def test_parse_l2t_thought_generation_response_success_extra_whitespace(self):
        response_text = "Your new thought:   New thought with spaces.   "
        expected_thought = "New thought with spaces."
        self.assertEqual(
            L2TResponseParser.parse_l2t_thought_generation_response(response_text),
            expected_thought,
        )

    def test_parse_l2t_thought_generation_response_keyword_not_found(self):
        response_text = "This response does not contain the new thought keyword."
        self.assertIsNone(
            L2TResponseParser.parse_l2t_thought_generation_response(
                response_text
            )
        )

    def test_parse_l2t_thought_generation_response_keyword_present_no_thought(self):
        response_text = "Your new thought:"
        self.assertIsNone(
            L2TResponseParser.parse_l2t_thought_generation_response(
                response_text
            )
        )
    
    def test_parse_l2t_thought_generation_response_empty_string(self):
        response_text = ""
        self.assertIsNone(
            L2TResponseParser.parse_l2t_thought_generation_response(response_text)
        )


if __name__ == "__main__":
    # Re-enable logging if you run this file directly and want to see logs
    # logging.disable(logging.NOTSET) 
    unittest.main()
