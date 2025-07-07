import re
import logging
from typing import List, Optional

from src.aot.dataclasses import ParsedLLMOutput

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
        parsing_initial_answer_block = False
        parsing_reflection_block = False
        parsing_refined_answer_block = False
        
        lines = response_text.splitlines()
        for line_content_raw in lines:
            line = line_content_raw.strip()
            if not line: continue
            
            # Legacy support for old step-based format
            if "There are no more unique reasoning steps left." in line:
                parsed_output.ran_out_of_steps_signal = True
            step_match = re.match(r"^(?:###\s*)?Step\s*(\d+):\s*(.*)", line, re.IGNORECASE)
            if step_match:
                parsing_final_answer_block = False
                parsing_initial_answer_block = False
                parsing_reflection_block = False
                parsing_refined_answer_block = False
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
            
            # Parse Initial Answer
            if not parsing_initial_answer_block and normalized_line_start.startswith("initial answer"):
                parts = re.split(r":\s*", line, maxsplit=1)
                if len(parts) > 1 and parts[1].strip(): 
                    parsed_output.initial_answer = parts[1].strip()
                else: 
                    parsed_output.initial_answer = "" # Indicates initial answer block started but no text yet
                parsed_output.is_initial_answer_provided = True
                parsing_initial_answer_block = True
                parsing_final_answer_block = False
                parsing_reflection_block = False
                parsing_refined_answer_block = False
                continue
            
            # Parse Reflection
            if not parsing_reflection_block and normalized_line_start.startswith("reflection"):
                parts = re.split(r":\s*", line, maxsplit=1)
                if len(parts) > 1 and parts[1].strip(): 
                    parsed_output.reflection_text = parts[1].strip()
                else: 
                    parsed_output.reflection_text = "" # Indicates reflection block started but no text yet
                parsed_output.is_reflection_provided = True
                parsing_reflection_block = True
                parsing_final_answer_block = False
                parsing_initial_answer_block = False
                parsing_refined_answer_block = False
                continue
            
            # Parse Refined Answer
            if not parsing_refined_answer_block and normalized_line_start.startswith("refined answer"):
                parts = re.split(r":\s*", line, maxsplit=1)
                if len(parts) > 1 and parts[1].strip(): 
                    parsed_output.refined_answer = parts[1].strip()
                else: 
                    parsed_output.refined_answer = "" # Indicates refined answer block started but no text yet
                parsed_output.is_refined_answer_provided = True
                parsing_refined_answer_block = True
                parsing_final_answer_block = False
                parsing_initial_answer_block = False
                parsing_reflection_block = False
                continue
            
            # Parse Final Answer
            if not parsing_final_answer_block and normalized_line_start.startswith("final answer"):
                parts = re.split(r":\s*", line, maxsplit=1)
                if len(parts) > 1 and parts[1].strip(): 
                    parsed_output.final_answer_text = parts[1].strip()
                else: 
                    parsed_output.final_answer_text = "" # Indicates final answer block started but no text yet
                parsed_output.is_final_answer_marked_done = True
                parsing_final_answer_block = True
                parsing_initial_answer_block = False
                parsing_reflection_block = False
                parsing_refined_answer_block = False
                continue
            
            # Handle multi-line content for each block
            if parsing_initial_answer_block:
                if line.startswith("---") or line.startswith("===") or "LLM call duration:" in line or \
                   re.match(r"^(?:###\s*)?(?:Reflection|Refined Answer|Final Answer):", line, re.IGNORECASE):
                    parsing_initial_answer_block = False
                else:
                    if parsed_output.initial_answer == "": 
                        parsed_output.initial_answer = line_content_raw
                    elif parsed_output.initial_answer is not None: 
                        parsed_output.initial_answer += "\n" + line_content_raw
            elif parsing_reflection_block:
                if line.startswith("---") or line.startswith("===") or "LLM call duration:" in line or \
                   re.match(r"^(?:###\s*)?(?:Initial Answer|Refined Answer|Final Answer):", line, re.IGNORECASE):
                    parsing_reflection_block = False
                else:
                    if parsed_output.reflection_text == "": 
                        parsed_output.reflection_text = line_content_raw
                    elif parsed_output.reflection_text is not None: 
                        parsed_output.reflection_text += "\n" + line_content_raw
            elif parsing_refined_answer_block:
                if line.startswith("---") or line.startswith("===") or "LLM call duration:" in line or \
                   re.match(r"^(?:###\s*)?(?:Initial Answer|Reflection|Final Answer):", line, re.IGNORECASE):
                    parsing_refined_answer_block = False
                else:
                    if parsed_output.refined_answer == "": 
                        parsed_output.refined_answer = line_content_raw
                    elif parsed_output.refined_answer is not None: 
                        parsed_output.refined_answer += "\n" + line_content_raw
            elif parsing_final_answer_block:
                if line.startswith("---") or line.startswith("===") or "LLM call duration:" in line or \
                   re.match(r"^(?:###\s*)?(?:Initial Answer|Reflection|Refined Answer):", line, re.IGNORECASE):
                    parsing_final_answer_block = False
                else:
                    if parsed_output.final_answer_text == "": 
                        parsed_output.final_answer_text = line_content_raw
                    elif parsed_output.final_answer_text is not None: 
                        parsed_output.final_answer_text += "\n" + line_content_raw
        
        # Clean up parsed content
        for field_name in ['initial_answer', 'reflection_text', 'refined_answer', 'final_answer_text']:
            field_value = getattr(parsed_output, field_name)
            if field_value is not None:
                cleaned_value = field_value.strip()
                if not cleaned_value:  # If empty after strip
                    setattr(parsed_output, field_name, None)
                else:
                    setattr(parsed_output, field_name, cleaned_value)
        
        return parsed_output
