# Learn to Think: Bootstrapping LLM Reasoning Capability Through Graph Representation Learning

**Hang Gao, Chenhao Zhang<sup>1,2,3*</sup>, Tie Wang<sup>4†</sup>, Junsuo Zhao<sup>1,2,3</sup>, Fengge Wu<sup>1,2,3†</sup>, Changwen Zheng<sup>1,2,3</sup>, Huaping Liu<sup>5</sup>**

<sup>1</sup>Institute of Software, Chinese Academy of Sciences  
<sup>2</sup>National Key Laboratory of Space Integrated Information System  
<sup>3</sup>University of Chinese Academy of Sciences  
<sup>4</sup>—  
<sup>5</sup>—

---

## Abstract

Large Language Models (LLMs) have achieved remarkable success across various domains. However, they still face significant challenges, including high computational costs for training and limitations in solving complex reasoning problems. Although existing methods have extended the reasoning capabilities of LLMs through structured paradigms, these approaches often rely on task-specific prompts and predefined reasoning processes, which constrain their flexibility and generalizability. 

To address these limitations, we propose a novel framework that leverages graph learning to enable more flexible and adaptive reasoning capabilities for LLMs. Specifically, this approach models the reasoning process of a problem as a graph and employs LLM-based graph learning to guide the adaptive generation of each reasoning step. To further enhance the adaptability of the model, we introduce a Graph Neural Network (GNN) module to perform representation learning on the generated reasoning process, enabling real-time adjustments to both the model and the prompt.

Experimental results demonstrate that this method significantly improves reasoning performance across multiple tasks without requiring additional training or task-specific prompt design.

Code: [https://github.com/zch65458525/L2T](https://github.com/zch65458525/L2T)

---

## 1. Introduction

In recent years, LLMs [Radford et al., 2018] have achieved remarkable success in fields such as natural language processing [Brown et al., 2022], machine translation [Jiao et al., 2022], and code generation [Ni et al., 2022]. However, training these models requires substantial computational resources and energy, resulting in high costs and environmental impacts [Patterson et al., 2022]. Efficiently utilizing LLMs has thus become a key research focus, with prompt engineering emerging as a critical technique [Liu et al., 2023; Zhou et al., 2022; Sun et al., 2022].

By designing effective prompts, it is possible to optimize model performance without additional training, making it a cost-effective and straightforward approach. Notably, the Chain-of-Thought (CoT) method [Wei et al., 2022] has demonstrated significant improvements in tasks such as mathematical reasoning and logical inference by guiding models through step-by-step reasoning processes. CoT works by crafting prompts that break down complex problems into logical steps, enabling the model to solve them incrementally.

Based on Chain of Thought, numerous related methods have been proposed in recent years, including:

- **Tree of Thoughts (ToT)** [Chu et al., 2024],
- **Graph of Thoughts (GoT)** [Besta et al., 2024],
- **Thread of Thoughts (ThoT)** [Zhou et al., 2023b].

These methods introduce more complex thinking paradigms—such as tree structures, graph structures, and thread-based reasoning—thereby further extending the reasoning capabilities of LLMs. Compared to chain-based reasoning structures, these approaches have significantly enhanced the breadth and depth of the cognition of LLMs [Qiao et al., 2023] and have played an active role in optimizing the performance of LLMs [Hadi et al., 2024].

However, these methods still face several critical challenges:

1. **Lack of Adaptability:** Existing approaches are often unable to make real-time adjustments to models and prompts in response to dynamic changes in scenarios, resulting in limited flexibility and robustness when addressing diverse tasks [Chu et al., 2024].
2. **Task-Specific Prompt Design:** These methods often require task-specific prompt design to handle different tasks effectively, particularly for those involving more complex reasoning processes.
3. **Limited Generalizability:** This heavy dependence on task-specific prompts severely undermines the generalizability of such methods.

A possible solution is to collect task-specific data and use fine-tuning methods to train the LLM, but this approach incurs significant costs and is not feasible for cases where only API access is available.

---

**Key Question:**  
Is there a way to address different types of problems in a unified manner without requiring LLM training or additional prompt design, while also allowing the model to flexibly adjust based on the problem and reasoning process?

---

### Our Approach: Learn to Think (L2T)

We propose **Learn to Think (L2T)**, which guides the LLM to “think” based on graph learning. This method:

- Employs graphs to unify the representation of the reasoning process of LLMs across different tasks.
- Utilizes LLM-based graph learning to adaptively guide reasoning strategies for various scenarios.
- Introduces a GNN-based reasoning mode selection module for real-time adjustments during the reasoning process, refined within a reinforcement learning framework.

#### **Contributions**

- **A reasoning framework** for LLMs that can adapt to different problems and develop reasoning pathways without requiring task-specific prompts.
- **A GNN-based reasoning mode selection module** that enables real-time adjustment of LLM reasoning strategies, further optimized through reinforcement learning.
- **Extensive experiments** to thoroughly validate and analyze the proposed method.

---

## 2. Related Works

### Prompt Engineering

Prompt engineering for LLMs has seen significant advancements, introducing innovative techniques aimed at enhancing reasoning and reliability:

- **Chain-of-Thought (CoT)** [Wei et al., 2022] for reasoning capabilities via intermediate steps,
- **Self-consistency** [Wang et al., 2023] for reliability via output aggregation,
- **Interactive QA** [Yao et al., 2023b; Masson et al., 2024] for adaptive reasoning,
- **Retrieval-Augmented Generation (RAG)** [Lewis et al., 2020] to ensure factual accuracy,
- **Chain-of-Verification (CoVe)** [Dhuliawala et al., 2024], **Chain-of-Note (CoN)** [Yu et al., 2023], and **Chain-of-Knowledge (CoK)** for robust step-by-step validation,
- Prompting research for **user intent understanding** [Diao et al., 2024], **autonomous prompt selection** [Zhou et al., 2023a], **tool integration** [Paranjape et al., 2023], and **emotional control** [Li et al., 2023].

### Logic and Reasoning within LLM Prompting

To enhance logic and reasoning in LLM prompting, several methods have emerged:

- **Auto-CoT** [Zhang et al., 2023]: Automated reasoning chain generation,
- **Logical CoT (LogiCoT)** [Zhao et al., 2024]: Symbolic logic for step-by-step verification,
- **Prompt Sketching** [Beurer-Kellner et al., 2024]: Output constraint to logical structures,
- **Tree of Thoughts (ToT)** [Yao et al., 2023a] and **Graph of Thoughts (GoT)** [Besta et al., 2024]: Hierarchical and graph-based structures,
- **Algorithm of Thoughts (AoT)** [Sel et al., 2024]: In-context algorithmic examples,
- **Thread of Thought (ThoT)** [Zhou et al., 2023b]: Structured thought threads.

These typically follow predefined reasoning processes and heavily depend on task-specific prompts—limiting adaptability and generalizability. Our method addresses these limitations by enabling more flexible and adaptive reasoning capabilities for LLMs.

---

## 3. Method

Our method consists of the following parts:

1. **Representing the complete logical reasoning process** of the LLM as a specifically designed graph.
2. **Automatically generating the format and evaluation criteria** of the reasoning process, then employing a graph learning framework to process the reasoning process graph—facilitating flexible and adaptive multi-step problem-solving without task-specific prompts.
3. **Iteratively refining the reasoning model through reinforcement learning.**

### 3.1 Reasoning Process Graph

The conversation with the LLM consists of user messages (prompts) and the LLM’s responses (thoughts). Various structures have been proposed for organizing prompts and thoughts, such as chains [Wei et al., 2022], trees [Yao et al., 2023a], graphs [Besta et al., 2024], etc. Graphs can represent most existing models, as trees, chains, and other structures are special cases of graphs.

#### **Definition**

The entire reasoning process of an LLM is represented as a **reasoning process graph** \( G = (V, E) \):

- \( V \): Set of nodes (each node \( v \in V \) is a thought generated by the LLM)
- \( E \): Set of edges (each edge \( e \in E \) is a connection from one thought to its subsequent thought)

\( V \) can be partitioned into two subsets:
- \( V_{pres} \): Nodes corresponding to unprocessed thoughts (basis for generating subsequent thoughts)
- \( V_{hist} \): Nodes that have already been processed and will no longer be revisited

Each node \( v \in V_{pres} \) is assigned a category label \( Y_v \in \{1,2,3,4\} \):

- **1:** Reasoning should not proceed based on node \( v \)
- **2:** Reasoning should continue based on node \( v \)
- **3:** Node \( v \) should be output as the final result
- **4:** Backtracking: reasoning continues based on its parent node

To assign labels, we utilize **LLM-based graph learning for node classification**, and these labels guide the thought generation process—eliminating the need for task-specific prompts.

---

### 3.2 Thought Generation Framework

Since the reasoning process is step by step, we detail how reasoning is performed at the **first step**, **intermediate k-th step**, and **final step**.

#### **First Step**

- Obtain the initial state.
- Use the LLM to generate:
  - Initial reasoning process graph \( G^{(1)} = (V^{(1)}, E^{(1)}) \) with \( |V^{(1)}| = 1, E^{(1)} = \emptyset \)
  - Constraint format and examples (\( X_{fmt} \))
  - Evaluation criteria for generated thoughts (\( X_{eva} \))
- Node in \( V_{pres}^{(1)} \) contains the task description.

#### **k-th Step**

Generate subsequent thoughts to construct \( G^{(k)} \) based on \( G^{(k-1)} \):

1. **Reasoning Process Graph Node Classification**
    - For each \( v \in V_{pres}^{(k-1)} \), extract subgraph \( G_{v}^{(k-1)} \) (all nodes with paths to \( v \) of length less than \( \beta \))
    - Topological relationships and node attributes are annotated as text and input to the LLM for node classification:

      \[
      \hat{Y}_v^{(k)} = f(S_{node}, \tau(\{ x_u \mid u \in V_v^{(k-1)} \}), G_v^{(k-1)})
      \]
      Where \( S_{node} \) is the node classification prompt, \( \tau \) converts graph info to text.

2. **GNN-Based Reasoning Mode Selection**

    - The GNN \( g(\cdot) \) processes \( G^{(k-1)} \) and outputs node features.
    - We use a one-layer GCN [Kipf & Welling, 2017] + two-layer MLP.
    - Each node \( v \) in \( V_{pres}^{(k)} \) gets a vector \( a_v^{(k)} \), containing:
      - Prompt-related params (e.g., number of branches)
      - LLM hyperparameters (e.g., temperature)

      \[
      a_v^{(k)} = A(g[v](G^{(k-1)}))
      \]

    - \( A(\cdot) \) acts as the Actor in Actor-Critic RL.

3. **Thought Generation**

    - For label 2 nodes, generate new child nodes using:

      \[
      x_u = f(S_{gen}, x_v^{(k-1)}, X_{fmt}, a_v^{(k)})
      \]

      Where \( S_{gen} \) is the data generation prompt. \( a_v^{(k)} \) also sets LLM hyperparameters.

#### **Final Step**

The process ends when \( V_{pres} \) contains a node with label 3 (final result) or if all are labeled 1 (then thoughts are regenerated; if still 1, terminate).

---

### 3.3 Update: Actor-Critic RL Optimization

We employ the **Actor-Critic algorithm** (PPO [Schulman et al., 2017]) to optimize the GNN-based reasoning mode selection module.

- At each step \( k \), treat \( a_v^{(k)} \) as an action (choosing reasoning mode) based on GNN output.
- The Actor \( A(\cdot) \) generates \( a_v^{(k)} \) from policy \( \pi(a_v^{(k)} | g[v](G^{(k-1)}); \theta_{actor}) \):

  \[
  a_v^{(k)} \sim \pi(a_v^{(k)} | g[v](G^{(k-1)}); \theta_{actor})
  \]

- Immediate reward \( r_k \):
  - 100 if final result, otherwise 0-10 (scored by LLM with \( G^{(k)}, X_{eva} \))
- Critic estimates state value \( V(g[v](G^{(k-1)})) \)
- Use PPO to optimize Actor and Critic networks.

For graphs with multiple pending nodes (\( |V_{pres}^{(k)}| > 1 \)), each node is processed sequentially.

---

## 4. Experiments

### 4.1 Comparison with State-of-the-Art Methods

**Baselines:**  
Experiments used GPT-4o as the base model.

- Compared **L2T** with:
  - **IO** (original GPT-4o)
  - **CoT** [Wei et al., 2022] (zero-shot, few-shot)
  - **ToT** [Yao et al., 2023a]
  - **GoT** [Besta et al., 2024]
  - **AoT** [Sel et al., 2024]

#### **Tables (Summaries)**

- **Table 1:** Results for Sudoku performance
- **Table 2:** Results for Game of 24
- **Table 3:** Results for TruthQuest
- **Table 4:** Creative Writing (relative score to L2T)

*Note: Results italicized when copied directly from methods that do not use task-specific prompts.*

#### **Tasks**

Evaluated on four tasks:
- **Sudoku:** 3×3, 4×4, 5×5 grids ([Long, 2023])
- **Game of 24:** Use four numbers and basic operations to reach 24 ([Yao et al., 2023a])
- **TruthQuest:** Knights and Knaves puzzles ([Mondorf & Plank, 2024])
- **Creative Writing:** Writing challenges to assess logic and coherence

L2T was tested using identical prompts for all tasks.

#### **Settings**

- Used GPT-4o API for all experiments and baselines.
- L2T w/o GNN: L2T without the GNN-based reasoning mode selection (no training required).
- Also tested baselines with task-specific components removed.

#### **Results**

- L2T consistently outperforms others, especially **without task-specific prompts**.
- Even without the GNN-based module, performance remains strong.
- L2T achieves higher or equivalent creative writing scores in over 80% of cases.
- Fewer reasoning steps required with the GNN-based module.

#### **Additional Tables**

- **Table 5:** Comparison of accuracy and number of generated nodes
- **Table 6:** Comparison of prompt tokens per thought, generate tokens per thought, and tokens per case
- **Table 7:** Comparison of LLM access counts (min value bolded)

#### **Computational Consumption Analysis**

- L2T’s token usage is comparable to other methods, outperforming GoT.
- Achieves complex reasoning without excessive computational resources.

#### **Process Analysis**

- **Figure 4:** Correlation of temperature and top-p values output by the GNN-based module.
- For creative writing, temperature and top-p are inversely related; for Game of 24, directly related—showing task-specific adaptation.

---

## 5. Conclusion

This paper proposes a novel LLM reasoning method, **L2T**, which utilizes a graph-based framework to represent the reasoning process and applies graph learning techniques to learn and analyze this graph, generating corresponding reasoning strategies. L2T incorporates graph learning via LLMs and GNNs, eliminating the need for specifically designed prompts and integrating reinforcement learning to continuously self-optimize during problem solving. Extensive experiments demonstrate the effectiveness of L2T.

---

## References

*(Full references as in the PDF, e.g.:)*

- [Besta et al., 2024] Maciej Besta, Nils Blach, Ales Kubicek, et al. *Graph of Thoughts: Solving elaborate problems with large language models.* AAAI 2024.
- [Wei et al., 2022] Jason Wei, Xuezhi Wang, Dale Schuurmans, et al. *Chain-of-thought prompting elicits reasoning in large language models.* NeurIPS 2022.
- [Kipf & Welling, 2017] Thomas N. Kipf, Max Welling. *Semi-supervised classification with graph convolutional networks.* ICLR 2017.
- [Schulman et al., 2017] John Schulman, Filip Wolski, Prafulla Dhariwal, et al. *Proximal Policy Optimization Algorithms.* arXiv preprint arXiv:1707.06347, 2017.

*(…and so on; see the full reference list in the PDF for all entries.)*

---

## Appendices

### A. Implementation and Experimental Details

#### A.1 Actor-Critic Algorithm Implementation

- Implemented using Proximal Policy Optimization (PPO) [Schulman et al., 2017]
- PPO stabilizes policy optimization with clipped surrogate objective

#### A.2 Hyperparameters

- Learning rate: \(5 \times 10^{-3}\)
- RL training: 20 epochs
- PPO clip parameter: 0.2
- Max gradient norm: 0.5
- Path hyperparameter \( \beta = 2 \)

#### A.3 Node Feature Extraction

The node feature extraction function \( \tau(\cdot) \) outputs:  
“The former generated thoughts are: {…}, {…}, …”

#### A.4 Property Adjust Vector Details

The vector \( a_v^{(k)} \) contains parameters for prompt adjustment (e.g., number of branches) and LLM fine-tuning (e.g., temperature, top-p).

#### A.5 Prompt Implementations

- **Format Generation Prompt (\( X_{fmt} \))**: Generates content format and step-by-step examples.
- **Evaluation Information Prompt (\( X_{eva} \))**: Criteria for assessing each step.
- **Evaluation Prompt**: Rates the helpfulness of a result from 0 to 10.
- **Prompt \( S_{node} \)**: For node classification (terminate, continue, complete, backtrack).
- **Prompt \( S_{gen} \)**: For generating new thoughts.

---

### B. Tasks

#### B.1 Game of 24

Use 4 numbers and basic arithmetic operations (\( +, -, \times, \div \)) to obtain 24; dataset from [Yao et al., 2023a].

#### B.2 Sudoku Puzzle

Fill 1 to \( n \) in an \( n \times n \) grid, no repeats in any row or column; benchmark from [Long, 2023].

#### B.3 Truth Quest

Knights and Knaves puzzles: deduce identity of each character based on their statements; test on 3, 4, 5 characters ([Mondorf & Plank, 2024]).

#### B.4 Creative Writing

Two creative writing tasks:
1. Expand several words into sentences, combine into a paragraph.
2. Expand four short sentences into small paragraphs, combine into a larger paragraph.

---

### C. Reasoning Processes

**C.1 Game of 24**  
*(Example output of step-by-step thoughts with labels)*

- Thought 0 → Thoughts 1-5:  
  Input: [10, 9, 2, 3], Plan: 10 + 2 = 12, Output: [9, 3, 12], Label: 2 (Continue)  
  …(other examples continue as per PDF)…

- …[Further steps and labeling as in the text above]…

**C.2 Creative Writing**  
*(Example output for the creative writing task, showing the progressive combination of sentences as described)*

---

### D. Further Backgrounds

#### D.1 Graph Neural Networks (GNNs)

- GNNs update node representations by aggregating information from neighbors.

  \[
  h_v^{(k)} = \mathrm{AGGREGATE}(\{ h_u^{(k-1)} : u \in N(v) \})
  \]
  \[
  h_v^{(k)} = \sigma(W^{(k)} \cdot [h_v^{(k-1)} \oplus h_v^{(k)}] + b^{(k)})
  \]

- For graph-level tasks:

  \[
  h_G = \mathrm{POOL}(\{ h_v^{(K)} : v \in V \})
  \]

- Node classification loss:

  \[
  L = -\sum_{v \in V} y_v \log \hat{y}_v
  \]

#### D.2 Actor-Critic Algorithm

Main steps:

1. Initialize policy (Actor) and value (Critic) networks.
2. At each step, action \( a_t = \pi_\theta(s_t) \).
3. Observe reward \( r_t \), state \( s_{t+1} \).
4. Critic TD error: \( \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t) \).
5. Critic update:

   \[
   \theta_{critic} \leftarrow \theta_{critic} + \alpha_{critic} \delta_t \nabla_{\theta_{critic}} V(s_t)
   \]

6. Actor update:

   \[
   \theta_{actor} \leftarrow \theta_{actor} + \alpha_{actor} \delta_t \nabla_{\theta_{actor}} \log \pi_\theta(s_t, a_t)
   \]

#### D.3 PPO Algorithm

- Clipped objective:

  \[
  L_{CLIP}(\theta) = \hat{\mathbb{E}}_t [\min (r_t(\theta) \hat{A}_t, \mathrm{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t )]
  \]

- Importance ratio: \( r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} \)
- Advantage function (GAE):

  \[
  \hat{A}_t = \delta_t + (\gamma \lambda)\delta_{t+1} + \dots + (\gamma \lambda)^{T-t+1} \delta_T
  \]

- PPO objective:

  \[
  L_{PPO}(\theta) = \hat{\mathbb{E}}_t [\min (r_t(\theta) \hat{A}_t, \mathrm{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t )]
  \]

---

**Acknowledgments:**  
We would like to express our sincere gratitude to the reviewers of this paper, as well as the Program Committee and Area Chairs, for their valuable comments and suggestions. This work is supported by the CAS Project for Young Scientists in Basic Research, Grant No. YSBR-040.

---
