import logging
import re
from typing import Optional

from .dataclasses import L2TNodeCategory

logger = logging.getLogger(__name__)


class L2TResponseParser:
    @staticmethod
    def parse_l2t_initial_response(response_text: str) -> Optional[str]:
        """
        Parses the raw text response from the LLM after an initial prompt.
        Looks for various thought markers and extracts the following text.
        Uses multiple fallback strategies for robustness.
        """
        try:
            # Multiple possible markers to look for
            markers = [
                "Your thought:",
                "**First Thought:**",
                "First Thought:",
                "My thought:",
                "Thought:",
                "Initial thought:"
            ]
            
            # Try each marker
            for marker in markers:
                marker_idx = response_text.lower().rfind(marker.lower())
                if marker_idx != -1:
                    thought = response_text[marker_idx + len(marker):].strip()
                    if thought:
                        logger.info(f"Successfully parsed initial thought using marker: '{marker}'")
                        return thought
            
            # Fallback 1: Look for patterns like "**[anything]:**" at the end
            pattern = r'\*\*[^*]+:\*\*\s*(.*?)(?:\n\n|\Z)'
            matches = re.findall(pattern, response_text, re.DOTALL | re.IGNORECASE)
            if matches:
                thought = matches[-1].strip()
                if thought:
                    logger.info("Successfully parsed initial thought using pattern fallback")
                    return thought
            
            # Fallback 2: Take the last substantial paragraph
            paragraphs = [p.strip() for p in response_text.split('\n\n') if p.strip()]
            if paragraphs:
                # Filter out very short paragraphs (likely not the main thought)
                substantial_paragraphs = [p for p in paragraphs if len(p) > 50]
                if substantial_paragraphs:
                    thought = substantial_paragraphs[-1]
                    logger.info("Successfully parsed initial thought using paragraph fallback")
                    return thought
                else:
                    # If no substantial paragraphs, take the last one anyway
                    thought = paragraphs[-1]
                    logger.info("Successfully parsed initial thought using last paragraph fallback")
                    return thought
            
            # Fallback 3: Take the entire response if it's reasonable length
            if 20 <= len(response_text.strip()) <= 1000:
                logger.info("Using entire response as initial thought")
                return response_text.strip()
                
            logger.warning(
                "Could not parse initial thought from response: '%s'",
                response_text[:200] + "..." if len(response_text) > 200 else response_text
            )
            return None
            
        except Exception as e:
            logger.warning(
                "Error parsing initial LLM response: '%s'. Error: %s",
                response_text[:200] + "..." if len(response_text) > 200 else response_text,
                e,
                exc_info=True,
            )
            return None

    @staticmethod
    def parse_l2t_node_classification_response(
        response_text: str,
    ) -> Optional[L2TNodeCategory]:
        """
        Parses the raw text response from the LLM after a node classification prompt.
        Looks for classification markers and extracts the category string.
        Uses multiple fallback strategies for robustness.
        """
        try:
            # Multiple possible markers to look for
            markers = [
                "Your classification:",
                "Classification:",
                "My classification:",
                "Category:",
                "Answer:"
            ]
            
            # Try each marker
            for marker in markers:
                marker_idx = response_text.lower().rfind(marker.lower())
                if marker_idx != -1:
                    category_str = response_text[marker_idx + len(marker):].strip()
                    if category_str:
                        category = L2TResponseParser._extract_category(category_str)
                        if category:
                            logger.info(f"Successfully parsed classification using marker: '{marker}'")
                            return category
            
            # Fallback 1: Look for any of the category names anywhere in the response
            category_names = ["CONTINUE", "TERMINATE_BRANCH", "FINAL_ANSWER", "BACKTRACK"]
            for category_name in category_names:
                if category_name.lower() in response_text.lower():
                    try:
                        logger.info(f"Successfully parsed classification using direct search: '{category_name}'")
                        return L2TNodeCategory[category_name]
                    except (KeyError, ValueError):
                        continue
            
            # Fallback 2: Look for partial matches
            response_upper = response_text.upper()
            if "CONTINUE" in response_upper:
                return L2TNodeCategory.CONTINUE
            elif "TERMINATE" in response_upper or "STOP" in response_upper:
                return L2TNodeCategory.TERMINATE_BRANCH
            elif "FINAL" in response_upper or "ANSWER" in response_upper:
                return L2TNodeCategory.FINAL_ANSWER
            elif "BACK" in response_upper or "RETURN" in response_upper:
                return L2TNodeCategory.BACKTRACK
                
            logger.warning(
                "Could not parse classification from response: '%s'",
                response_text[:200] + "..." if len(response_text) > 200 else response_text
            )
            return None
            
        except Exception as e:
            logger.warning(
                "Error parsing node classification LLM response: '%s'. Error: %s",
                response_text[:200] + "..." if len(response_text) > 200 else response_text,
                e,
                exc_info=True,
            )
            return None

    @staticmethod
    def _extract_category(category_str: str) -> Optional[L2TNodeCategory]:
        """Helper method to extract category from a string."""
        try:
            # Try exact match first
            potential_category = category_str.split()[0].upper()
            return L2TNodeCategory[potential_category]
        except (KeyError, ValueError, IndexError):
            # Try to find any category name in the string
            category_names = ["CONTINUE", "TERMINATE_BRANCH", "FINAL_ANSWER", "BACKTRACK"]
            category_str_upper = category_str.upper()
            for category_name in category_names:
                if category_name in category_str_upper:
                    try:
                        return L2TNodeCategory[category_name]
                    except (KeyError, ValueError):
                        continue
            return None

    @staticmethod
    def parse_l2t_thought_generation_response(
        response_text: str,
    ) -> Optional[str]:
        """
        Parses the raw text response from the LLM after a thought generation prompt.
        Looks for various thought markers and extracts the following text.
        Uses multiple fallback strategies for robustness.
        """
        try:
            # Multiple possible markers to look for
            markers = [
                "Your new thought:",
                "New thought:",
                "Next thought:",
                "My thought:",
                "Thought:",
                "Generated thought:"
            ]
            
            # Try each marker
            for marker in markers:
                marker_idx = response_text.lower().rfind(marker.lower())
                if marker_idx != -1:
                    thought = response_text[marker_idx + len(marker):].strip()
                    if thought:
                        logger.info(f"Successfully parsed thought generation using marker: '{marker}'")
                        return thought
            
            # Fallback 1: Look for patterns like "**[anything]:**" at the end
            pattern = r'\*\*[^*]+:\*\*\s*(.*?)(?:\n\n|\Z)'
            matches = re.findall(pattern, response_text, re.DOTALL | re.IGNORECASE)
            if matches:
                thought = matches[-1].strip()
                if thought:
                    logger.info("Successfully parsed thought generation using pattern fallback")
                    return thought
            
            # Fallback 2: Take the last substantial paragraph
            paragraphs = [p.strip() for p in response_text.split('\n\n') if p.strip()]
            if paragraphs:
                # Filter out very short paragraphs
                substantial_paragraphs = [p for p in paragraphs if len(p) > 30]
                if substantial_paragraphs:
                    thought = substantial_paragraphs[-1]
                    logger.info("Successfully parsed thought generation using paragraph fallback")
                    return thought
                else:
                    thought = paragraphs[-1]
                    logger.info("Successfully parsed thought generation using last paragraph fallback")
                    return thought
            
            # Fallback 3: Take the entire response if it's reasonable length
            if 10 <= len(response_text.strip()) <= 1000:
                logger.info("Using entire response as generated thought")
                return response_text.strip()
                
            logger.warning(
                "Could not parse thought generation from response: '%s'",
                response_text[:200] + "..." if len(response_text) > 200 else response_text
            )
            return None
            
        except Exception as e:
            logger.warning(
                "Error parsing thought generation LLM response: '%s'. Error: %s",
                response_text[:200] + "..." if len(response_text) > 200 else response_text,
                e,
                exc_info=True,
            )
            return None
