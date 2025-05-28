DEFAULT_L2T_CLASSIFICATION_MODEL_NAMES = ["meta-llama/llama-3.3-8b-instruct:free"]
DEFAULT_L2T_THOUGHT_GENERATION_MODEL_NAMES = ["tngtech/deepseek-r1t-chimera:free"]
DEFAULT_L2T_INITIAL_PROMPT_MODEL_NAMES = ["tngtech/deepseek-r1t-chimera:free"]
DEFAULT_L2T_CLASSIFICATION_TEMPERATURE = 0.1
DEFAULT_L2T_THOUGHT_GENERATION_TEMPERATURE = 0.2
DEFAULT_L2T_INITIAL_PROMPT_TEMPERATURE = 0.2
DEFAULT_L2T_MAX_STEPS = 10
DEFAULT_L2T_MAX_TOTAL_NODES = 50
DEFAULT_L2T_MAX_TIME_SECONDS = 120
DEFAULT_L2T_X_FMT_DEFAULT = """\
**Format Constraints**

*   **Clarity and Conciseness:** Thoughts should be clear, concise, and easy to understand. Avoid jargon where possible, or explain it if necessary.
*   **Logical Flow:** Ensure that thoughts follow a logical sequence. Each new thought should build upon the previous ones, contributing to the overall reasoning process.
*   **Actionable Insights:** Where appropriate, thoughts should lead to actionable insights or next steps. This is particularly important for problem-solving or decision-making tasks.
*   **Supporting Evidence:** If a thought involves a claim or assertion, it should be backed by evidence or reasoning. Specify sources if applicable.
*   **Alternative Perspectives:** Encourage the exploration of alternative perspectives or solutions. This demonstrates a comprehensive understanding of the problem.
*   **Structured Output:** Present thoughts in a structured manner, perhaps using bullet points or numbered lists for complex ideas. This aids readability and comprehension.
*   **Language:** Use precise and unambiguous language. Avoid colloquialisms or slang that might be misinterpreted.
*   **Tone:** Maintain a neutral and objective tone. This is crucial for unbiased reasoning and analysis.
"""
DEFAULT_L2T_X_EVA_DEFAULT = """\
**Evaluation Criteria**

*   **Relevance:** How relevant is the thought to the problem or question at hand? Does it directly address the core issue?
*   **Coherence:** Does the thought make sense in the context of previous thoughts? Is it logically consistent with the overall reasoning chain?
*   **Depth of Analysis:** Does the thought demonstrate a deep understanding of the subject matter? Does it go beyond surface-level observations?
*   **Originality:** Does the thought offer new insights or perspectives? Does it contribute novel ideas to the discussion?
*   **Feasibility:** If the thought proposes an action or solution, how feasible is it? Are there any practical constraints that need to be considered?
*   **Evidence Base:** Is the thought well-supported by evidence or logical reasoning? Are the sources credible and reliable?
*   **Clarity of Expression:** Is the thought expressed clearly and unambiguously? Is it easy to understand the intended meaning?
*   **Contribution to Goal:** How well does the thought contribute to achieving the overall goal or answering the main question?
"""
