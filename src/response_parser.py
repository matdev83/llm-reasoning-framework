import re
import logging
from typing import List, Optional

from src.aot_dataclasses import ParsedLLMOutput

class ResponseParser:
    SKIP_KEYWORDS = [
        "verify", "confirm", "check", "review", "ensure",
        "correct", "matches", "requirement", "final", "total", "result"
    ]
    @staticmethod
    def _is_verification_step(line_content: str) -> bool:
        return any(kw in line_content.lower() for kw in ResponseParser.SKIP_KEYWORDS)
    @staticmethod
    def parse_llm_output(response_text: str) -> ParsedLLMOutput:
        parsed_output = ParsedLLMOutput()
        parsing_final_answer_block = False
        lines = response_text.splitlines()
        for line_content_raw in lines:
            line = line_content_raw.strip()
            if not line: continue
            if "There are no more unique reasoning steps left." in line:
                parsed_output.ran_out_of_steps_signal = True
            step_match = re.match(r"^(?:###\s*)?Step\s*(\d+):\s*(.*)", line, re.IGNORECASE)
            if step_match:
                parsing_final_answer_block = False
                full_step_line_content = line
                step_content_after_prefix = step_match.group(2).strip()
                parsed_output.all_lines_from_model_for_context.append(full_step_line_content)
                if not ResponseParser._is_verification_step(step_content_after_prefix):
                    parsed_output.valid_steps_for_trace.append(full_step_line_content)
                    m_ca = re.search(r"\((?:Current answer|Current state):([^)]+)\)", step_content_after_prefix, re.IGNORECASE)
                    if m_ca: parsed_output.last_current_answer = m_ca.group(1).strip()
                else: # Log skipped verification step
                    logging.debug(f"Skipping verification-like step: {full_step_line_content}")
                continue
            normalized_line_start = line.lower().lstrip('#').lstrip().lstrip(':').lstrip()
            if not parsing_final_answer_block and normalized_line_start.startswith("final answer"):
                parts = re.split(r":\s*", line, maxsplit=1)
                if len(parts) > 1 and parts[1].strip(): parsed_output.final_answer_text = parts[1].strip()
                else: parsed_output.final_answer_text = "" # Indicates final answer block started but no text yet
                parsed_output.is_final_answer_marked_done = True
                parsing_final_answer_block = True
                continue
            if parsing_final_answer_block:
                if line.startswith("---") or line.startswith("===") or "LLM call duration:" in line or \
                   re.match(r"^(?:###\s*)?Step\s*(\d+):", line, re.IGNORECASE): # Heuristics to stop final answer block
                    parsing_final_answer_block = False
                else:
                    if parsed_output.final_answer_text == "": parsed_output.final_answer_text = line_content_raw # First line of multiline
                    elif parsed_output.final_answer_text is not None: parsed_output.final_answer_text += "\n" + line_content_raw
        if parsed_output.final_answer_text is not None:
            parsed_output.final_answer_text = parsed_output.final_answer_text.strip()
            if not parsed_output.final_answer_text: # If empty after strip
                parsed_output.final_answer_text = None # Reset if it was just whitespace
        return parsed_output
