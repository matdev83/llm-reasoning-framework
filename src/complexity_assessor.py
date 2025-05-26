import logging
import re # Added for heuristic functions
from typing import List, Tuple

from src.aot_dataclasses import LLMCallStats
from src.aot_enums import AssessmentDecision
from src.llm_client import LLMClient
from src.prompt_generator import PromptGenerator

# --- Helper Functions for Specific Complex Patterns (Copied from aot_heuristic_function.py) ---

def _check_architect_pattern(prompt_lower: str) -> bool:
    """
    Checks for "architect a [system/solution] for [complex subject] that [handles complexity]".
    The subject itself needs to be somewhat descriptive.
    """
    pattern = r"\barchitect (?:a|an|the) (?:solution|system|platform|application|framework|infrastructure) for ([\w\s\-':,.()]+?) (?:that (?:handles|supports|addresses|scales to|processes|manages)|considering (?:[\w\s,]+? and [\w\s,]+?|scalability|security|performance)|with requirements for|which needs to|which must support|to achieve|capable of)\b"
    match = re.search(pattern, prompt_lower)
    if match:
        subject_description = match.group(1).strip()
        if subject_description and len(subject_description.split()) > 3:
            return True
    return False

def _check_complex_implementation_pattern(prompt_lower: str) -> bool:
    """
    Checks for "implement [item] that [has multiple/complex features/requirements]".
    Looks for conjunctions, lists of features, or keywords indicating complexity in requirements.
    """
    pattern = r"\b(implement|develop|create|build) (?:an?|the) ([\w\s\-()':,.]+?) (?:that|which|to|for) (.*(?:(?:and|or|,){2,}|handles (?:multiple|complex|various)|interacts with (?:multiple|several)|processes large|requires (?:complex|advanced|detailed|multiple|integration of|coordination between)|supports (?:[\w\s,]+?,){2,}[\w\s,]+?|featuring (?:[\w\s,]+?,){2,}[\w\s,]+?|covers (?:[\w\s,]+?,){2,}[\w\s,]+?))\b"
    match = re.search(pattern, prompt_lower)
    if match:
        item_implemented = match.group(2).strip()
        requirements_desc = match.group(3).strip()
        if not re.search(r"\b(simple|basic|small|trivial|dummy|example)\b", item_implemented, re.IGNORECASE):
            if len(requirements_desc.split()) > 4 or re.search(r"(?:and|or|,.*,)|multiple|complex|large|advanced|integration|coordination", requirements_desc, re.IGNORECASE):
                return True
    return False

def should_use_aot_heuristically(prompt_text: str) -> bool:
    """
    Analyzes the prompt text using deterministic heuristics to decide if it's
    VERY HIGHLY LIKELY to require an AoT (Algorithm of Thought) process.
    """
    prompt_lower = prompt_text.lower()

    explicit_decomposition_keywords = [
        r"\bstep(?:-| )by(?:-| )step\b",
        r"\bdetailed steps (?:to|for|on|how to)\b",
        r"\boutline the steps (?:to|for|involved in)\b",
        r"\bbreak down the (?:process|task|problem|steps) (?:of|for|to|into)\b",
        r"\bwalk me through (?:how to|the process of|the steps to)\b",
        r"\bstepwise (?:solution|guide|approach|instructions) (?:for|to)\b",
        r"\b(provide|create|develop|formulate|outline|devise|draft) a (?:detailed|comprehensive|strategic) (?:plan|strategy|roadmap|blueprint|procedure|method|approach|framework) (?:to|for|on|that addresses|covering)\b",
        r"\b(give|provide|write|list|generate) (?:detailed|full|comprehensive) instructions (?:for|on|to)\b",
        r"\bshow (?:all|your) work (?:for|in solving|to arrive at)?\b",
        r"\b(explain|describe|detail) (?:your reasoning|the thinking process|the logical steps|how you arrived at this) (?:clearly|thoroughly|in detail)\b"
    ]
    if any(re.search(keyword, prompt_lower) for keyword in explicit_decomposition_keywords):
        return True

    design_architecture_keywords = [
        r"\bdesign (?:the|a(?:n)?) (?:software|system|application|database|network|api|microservice|data pipeline|cloud|solution|it|technical|overall|end-to-end|secure|scalable|resilient|high-performance|distributed|fault-tolerant) architecture for\b",
        r"\b(propose|define|detail|outline|create|develop) an architecture for a system (?:that|which|to|for) (?:[\w\s]+ (?:and|or) [\w\s]+|handles|supports|meets requirements)\b",
        r"\b(create|develop|write|draft) a system design document (?:for|to|detailing|that covers)\b",
        r"\b(detail|define|specify) the (?:high-level|low-level|detailed|overall|complete) design (?:for|of)\b",
        r"\b(recommend|suggest|propose) a(?:n suitable|n appropriate| suitable| appropriate)? architecture for a system (?:that|which) (?:needs to|must support|will handle|requires (?:significant|complex|multiple))\b",
        r"\b(discuss|explain|apply|implement|choose|select) (?:appropriate|suitable|relevant) (?:design patterns|architectural patterns) (?:for|to|in|when building|to solve|to address)\b",
        r"\bexplain the (?:architecture|system design|internals|detailed workings) of (?:a|an|the|this) (?:complex|large-scale|distributed|fault-tolerant|enterprise-grade|mission-critical|existing|proposed) system\b",
        r"\b(what would be|propose|design) a (?:robust|scalable|efficient|secure|comprehensive|resilient) solution architecture for\b"
    ]
    if any(re.search(keyword, prompt_lower) for keyword in design_architecture_keywords):
        return True
    if _check_architect_pattern(prompt_lower):
        return True

    in_depth_explanation_phrases = [
        r"\bexplain in (?:great|full|thorough) detail\b",
        r"\b(provide|give|offer) a comprehensive (?:explanation|overview|analysis|breakdown) of\b",
        r"\banalyze (?:the|its|their) (?:causes and effects|pros and cons|trade-offs|implications|impact|benefits and drawbacks|root cause(?:s)?) (?:of|for|on) (?:[\w\s]+?) (?:in detail|comprehensively|thoroughly|from multiple perspectives|considering all factors)\b",
        r"\bperform a detailed (?:analysis|review|examination|investigation|audit) of\b",
        r"\b(thoroughly|critically|deeply) (?:examine|discuss|evaluate|review|investigate|analyze|scrutinize)\b",
        r"\b(explore|delve into|unpack|dissect) the (?:complexities|nuances|intricacies|underlying principles|full scope) of\b"
    ]
    if any(re.search(phrase, prompt_lower) for phrase in in_depth_explanation_phrases):
        return True

    if _check_complex_implementation_pattern(prompt_lower):
        return True

    specific_complex_coding_keywords = [
        r"\b(write|create|develop|build) (?:code|a script|a program|software|an application) (?:to|for|that) (?:integrate (?:[\w\s()\-]+?) (?:with|and|into) (?:[\w\s()\-]+?)(?: and (?:[\w\s()\-]+?))?|automate a multi-step (?:process|workflow|pipeline)|build a full-stack (?:[\w\s]+?) application (?:with|including|featuring)|parse (?:and|to) transform complex data (?:from|into) (?:multiple formats|a structured schema))",
        r"\bdevelop an algorithm for (?:solving (?:a specific|an?) (?:optimization problem|[\w\s]+? problem with (?:multiple|complex|specific|conflicting) constraints)|pathfinding in a (?:complex|large|dynamic|multi-layered) graph|[\w\s]+? under (?:multiple|specific|complex|stringent) constraints)",
        r"\brefactor (?:this|a|an|the) (?:large|complex|legacy|monolithic|spaghetti) codebase (?:to|for|in order to) (?:improve performance significantly|achieve better modularity|separate concerns|enhance security substantially|address critical vulnerabilities|migrate to a new architecture|reduce technical debt across several modules)",
        r"\bdebug (?:a|an|the|this) (?:complex|multi-threaded|distributed|concurrent|large-scale|performance-critical|memory leak in|race condition in|deadlock in|hard-to-reproduce|intermittent|cascading) (?:code|system|application|bug|issue|problem|failure)",
        r"\bintegrate (?:([\w\s()\-]+?) with ([\w\s()\-]+?)(?: and ([\w\s()\-]+?))?(?: using (?:[\w\s()\-]+? (?:protocol|api|sdk|library)))?|(?:a|this|an) (?:third-party|external) API (?:into|with) (?:our|an|the) existing (?:platform|system|application) to (?:achieve|provide|enable) (?:[\w\s]+? functionality|specific outcomes|new capabilities involving multiple steps))\b",
        r"\b(create|set up|implement|design|configure) a (?:ci/cd pipeline|continuous integration (?:and|&) (?:deployment|delivery) setup) for (?:a microservices architecture|a multi-environment application|a complex project with multiple dependencies|automated testing and release management)",
        r"\b(dockerize|containerize) (?:a multi-container application|this legacy system|an application with multiple services) and orchestrate (?:its deployment )?with (?:kubernetes|k8s|docker swarm|ecs|eks|gke)(?: including (?:persistent storage|networking configuration|secrets management|auto-scaling|service discovery|load balancing|monitoring and logging))",
        r"\bimplement (?:the )?(?:circuit breaker|saga|event sourcing|cqrs|strangler fig|repository pattern|unit of work|dependency injection|publish-subscribe|command pattern|chain of responsibility|leader election|sharding|raft|paxos|two-phase commit) pattern(?: for| to handle| in| to address)\b",
        r"\b(write|develop|implement|create|design) a smart contract (?:that|to|for) (?:manages (?:complex assets|multi-party agreements)|governs (?:a dao|decentralized voting)|automates (?:intricate financial transactions|supply chain logic)|facilitates (?:secure data exchange|tokenized systems))\b"
    ]
    if any(re.search(keyword, prompt_lower) for keyword in specific_complex_coding_keywords):
        return True

    data_algo_tasks = [
        r"\bdesign a custom data structure (?:for|to handle|to store) (?:[\w\s]+?) (?:that efficiently supports (?:multiple|specific|complex) operations like (?:[\w\s,]+, [\w\s,]+, and [\w\s,]+)|optimized for (?:specific criteria such as|both [\w\s]+ and [\w\s]+|a scenario involving (?:[\w\s]+? (?:and|or) [\w\s]+?)))\b",
        r"\bdevelop an (?:efficient|optimized|advanced|novel) algorithm to (?:solve (?:the )?([\w\s()\-.'’]+? problem)(?: for instance| e\.g\., traveling salesman| knapsack| max flow min cut| set cover)?|find the optimal strategy in (?:a complex game|resource allocation with (?:multiple|dynamic) constraints)|perform (?:[\w\s]+?) on (?:streaming|large-scale|real-time|high-dimensional|noisy|heterogeneous|complex) data (?:with (?:high accuracy|low latency|robustness)|efficiently|for anomaly detection|for predictive modeling))\b"
    ]
    if any(re.search(keyword, prompt_lower) for keyword in data_algo_tasks):
        return True

    multi_part_complex = [
        r"\bcompare and contrast (?:[\w\s()\-.'’]+?) and (?:[\w\s()\-.'’]+?)(?: and (?:[\w\s()\-.'’]+?))? (?:(?:across|on|based on) (?:several|multiple|various|\d+) (?:dimensions|criteria|aspects|factors|points)|in terms of (?:[\w\s,]+, [\w\s,]+, and [\w\s,]+)|considering aspects such as (?:[\w\s,]+, [\w\s,]+, and [\w\s,]+))\b",
        r"\bevaluate the (?:pros and cons|advantages and disadvantages|benefits and drawbacks|strengths and weaknesses) of (?:([\w\s()\-.'’]+?) versus ([\w\s()\-.'’]+?)|using (?:[\w\s()\-.'’]+?) (?:compared to|over|vs\.?) (?:[\w\s()\-.'’]+?)) (?:(?:for|in the context of|regarding) (?:[\w\s]+?) )?(?:providing a detailed justification for each|with a clear recommendation based on the analysis|thoroughly discussing each point|considering factors like (?:[\w\s,]+, [\w\s,]+, and [\w\s,]+))\b",
        r"\bdiscuss (?:([\w\s()\-.'’]+?)(?:, (?:then |and )?([\w\s()\-.'’]+?)){1,}, and (?:then )?([\w\s()\-.'’]+?)) (?:in the context of|regarding|concerning) (?:[\w\s]+?) (?:detailing|considering|analyzing) (?:their (?:interdependencies|interactions|relationships|synergies|trade-offs)|potential (?:conflicts|issues|challenges)|individual roles and combined effect)\b",
        r"\bexplain (?:how|why) (?:[\w\s()\-.'’]+?) (?:relates to|differs from|impacts|interacts with|influences|complements|contradicts) (?:[\w\s()\-.'’]+?), and then (?:propose|analyze|detail|discuss|suggest|recommend|elaborate on|what are the implications for|how can this be applied to|derive a strategy based on this)\b",
        r"\b(firstly|first|to begin with)(?:.*)(secondly|second|next|then)(?:.*)(thirdly|third|furthermore|additionally|also|in addition|subsequently|finally|lastly)\b"
    ]
    if any(re.search(keyword, prompt_lower) for keyword in multi_part_complex):
        return True

    complex_conditional_keywords = [
        r"\b(what are|analyze|discuss|detail|assess|evaluate|predict|forecast) the (?:potential|likely|possible|significant|major|far-reaching|cascading) (?:failures|implications|effects|consequences|ramifications|risks and benefits|ethical considerations|downstream impacts) (?:if|should|when|of a scenario where|in the event that) (?:[\w\s]+?) (?:were to|should|does|fails|is adopted|changes|occurs|were implemented|interacts with)\b",
        r"\banalyze the (?:full|overall|potential|detailed|comprehensive|systemic) impact of a scenario where (?:[\w\s]+?) (?:occurs|happens)(?: and (?:[\w\s]+?) (?:also occurs|also happens|is also true))?, considering (?:multiple aspects such as|various factors including|technical, business, (?:and|or) operational (?:aspects|factors|implications)|at least (?:[\w\s]+, [\w\s]+, and [\w\s]+) dimensions|short-term and long-term effects)\b",
        r"\b(devise|develop|create|outline|propose) a comprehensive (?:contingency plan|disaster recovery strategy|mitigation plan|risk management strategy|response plan|policy proposal) (?:for|to address|in case of|to handle|to govern) (?:a scenario (?:where|involving)|a situation (?:where|involving)|[\w\s]+? (?:failure|breach|crisis|disruption|ethical dilemma|complex challenge)) (?:including|detailing|covering|outlining) (?:mitigation (?:steps|strategies), recovery (?:procedures|steps)|multiple response levels|specific actions for (?:[\w\s]+, [\w\s]+, and [\w\s]+)|preventative measures and corrective actions|communication protocols|stakeholder engagement plans|long-term sustainability)\b",
    ]
    if any(re.search(keyword, prompt_lower) for keyword in complex_conditional_keywords):
        return True

    problem_solving_keywords = [
        r"\b(solve|address|tackle|resolve|find a solution for) the following (?:complex|multi-step|intricate|challenging|non-trivial|multi-constraint|multi-faceted|ill-defined|wicked) (?:problem|scenario|case study|puzzle|challenge|issue)(?::|\b)",
        r"\b(find|propose|develop|create|design|formulate) a solution (?:to|for) (?:a|an|the|this) (?:[\w\s()\-.'’]+?) (?:problem|challenge|issue|scenario) (?:that (?:satisfies|meets|addresses|balances) (?:all of )?(?:the following|these|multiple|specific|a set of|N) (?:constraints|requirements|conditions|criteria|objectives)|given (?:these|multiple|competing|conflicting) (?:objectives|goals|factors)|optimizing for (?:[\w\s]+?) while (?:respecting|adhering to|considering|balancing|minimizing|maximizing) (?:[\w\s]+?)(?: and (?:[\w\s]+?))?)\b"
    ]
    if any(re.search(keyword, prompt_lower) for keyword in problem_solving_keywords):
        return True

    math_logic_proof_keywords = [
        r"\bprove (?:that|the following(?: theorem| statement| conjecture| lemma| proposition)?|the correctness of (?:this algorithm|this method|this approach)|this (?:mathematical )?(?:statement|assertion|claim|identity|inequality)|the proposition that)\b",
        r"\bderive (?:the (?:complete|general )?formula for|a set of equations (?:describing|for|that model)|the (?:statistical )?properties of|the relationship between (?:[\w\s]+ and [\w\s]+)|an expression for|the general solution to)\b"
    ]
    if any(re.search(keyword, prompt_lower) for keyword in math_logic_proof_keywords):
        return True

    creative_writing_complex = [
        r"\b(write|create|compose|develop|draft|outline) a (?:story|script|novel|play|screenplay|technical whitepaper|research paper|detailed proposal|long-form essay|book|narrative|report|treatment|synopsis|manifesto|treatise) (?:on|about|featuring|set in|that explores|arguing for|analyzing) (?:[\w\s()\-.'’]+?) (?:that (?:must (?:include|contain|address|feature)|incorporates|features|contains|details) (?:(?:several|multiple|specific|the following) (?:elements|characters|plot points|details|requirements|themes|sections|chapters|viewpoints|arguments|twists|constraints)|(?:[\w\s]+, [\w\s]+, and [\w\s]+(?: at least)?))|which (?:follows|adheres to) (?:a specific (?:narrative arc|structure|format|outline|template|set of rules)|the provided (?:guidelines|outline|structure|constraints))|requiring (?:multiple (?:perspectives|viewpoints|character arcs)|a detailed plot with (?:specific turning points|multiple subplots|intricate backstories)|exploration of (?:[\w\s]+, [\w\s]+, and [\w\s]+)|integration of diverse sources))\b",
        r"\b(compose|write|develop) an in-depth essay (?:arguing (?:for|against) (?:[\w\s()\-.'’]+?) (?:by addressing|and refuting) (?:counterarguments|multiple viewpoints|alternative perspectives|potential objections|underlying assumptions)|analyzing the multifaceted impact of (?:[\w\s()\-.'’]+?) (?:from various angles|considering (?:[\w\s]+, [\w\s]+, and [\w\s]+)|on different stakeholders)|critiquing (?:[\w\s()\-.'’]+?) from (?:multiple|different|various) perspectives (?:such as [\w\s]+ and [\w\s]+)?|that explores (?:a complex topic|[\w\s]+?) through (?:lenses|perspectives) (?:[\w\s]+, [\w\s]+, and [\w\s]+)|synthesizing information from multiple sources to propose)\b"
    ]
    if any(re.search(pattern, prompt_lower) for pattern in creative_writing_complex):
        return True

    simulation_modeling_keywords = [
        r"\b(simulate|run a simulation of|develop a simulation for|model the simulation of) the (?:behavior of|process of|workflow for|operation of|response of|dynamics of) (?:a complex system|a network under (?:various conditions|heavy load|attack scenarios|stochastic inputs)|a financial model (?:under various scenarios|with multiple variables|for risk assessment|stress testing)|interactions between (?:multiple|many) (?:agents|entities|components|particles|actors)|an ecological system (?:with inputs X, Y, Z|under climate change|with feedback loops)|a chemical reaction pathway|a quantum system)\b",
        r"\b(model|create a model of|develop a model for|build a mathematical model of|construct a simulation environment for) the (?:interactions between|data flow (?:in|through)|state transitions of|dynamic behavior of|relationships among|evolution of|feedback mechanisms in) (?:multiple (?:components|systems|agents|variables|species|layers)|a (?:distributed|complex|stochastic|dynamical|non-linear|adaptive) system (?:with properties A, B, C|subject to constraints X, Y)|a business process (?:involving (?:several|multiple) (?:departments|stages|actors|queues)|with multiple dependencies|for optimization under uncertainty)|an economic scenario (?:with factors X, Y, Z|to predict Z|with endogenous variables)|a physical phenomenon (?:like turbulence|crystal growth)|a biological pathway)\b"
    ]
    if any(re.search(keyword, prompt_lower) for keyword in simulation_modeling_keywords):
        return True

    return False

class ComplexityAssessor:
    def __init__(self, llm_client: LLMClient, small_model_names: List[str], temperature: float, use_heuristic_shortcut: bool = True):
        self.llm_client = llm_client
        self.small_model_names = small_model_names
        self.temperature = temperature
        self.use_heuristic_shortcut = use_heuristic_shortcut # New parameter

    def assess(self, problem_text: str) -> Tuple[AssessmentDecision, LLMCallStats]:
        # Check heuristic shortcut first
        if self.use_heuristic_shortcut:
            if should_use_aot_heuristically(problem_text):
                logging.info("Heuristic shortcut triggered: Problem classified as AOT.")
                # Return a dummy LLMCallStats as no LLM call was made
                dummy_stats = LLMCallStats(
                    model_name="heuristic_shortcut",
                    prompt_tokens=0,
                    completion_tokens=0,
                    call_duration_seconds=0.0
                )
                return AssessmentDecision.AOT, dummy_stats

        logging.info(f"--- Initial Complexity Assessment using models: {', '.join(self.small_model_names)} ---")
        assessment_prompt = PromptGenerator.construct_assessment_prompt(problem_text)
        response_content, stats = self.llm_client.call(
            prompt=assessment_prompt, models=self.small_model_names, temperature=self.temperature
        )
        logging.debug(f"Assessment model ({stats.model_name}) raw response: '{response_content.strip()}'")
        logging.info(f"Assessment call: {stats.model_name}, Duration: {stats.call_duration_seconds:.2f}s, Tokens (C:{stats.completion_tokens}, P:{stats.prompt_tokens})")
        
        decision = AssessmentDecision.AOT # Default decision
        if response_content.startswith("Error:"):
            logging.warning(f"Assessment model call failed. Defaulting to AOT. Error: {response_content}")
            decision = AssessmentDecision.ERROR # Specific error state
        else:
            cleaned_response = response_content.strip().upper()
            if cleaned_response == AssessmentDecision.ONESHOT.value:
                decision = AssessmentDecision.ONESHOT
                logging.info("Assessment: Problem classified as ONESHOT.")
            elif cleaned_response == AssessmentDecision.AOT.value:
                decision = AssessmentDecision.AOT
                logging.info("Assessment: Problem classified as AOT.")
            else:
                logging.warning(f"Assessment model output ('{response_content.strip()}') was not '{AssessmentDecision.ONESHOT.value}' or '{AssessmentDecision.AOT.value}'. Defaulting to AOT.")
                # decision remains AOT (the default)
        return decision, stats
