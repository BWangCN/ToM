# Theory of Mind in Robotics: Comprehensive Literature Landscape
## State of Research as of Early 2026

---

## Table of Contents
1. [Overview and Key Themes](#1-overview-and-key-themes)
2. [Physical Robot / Embodied Simulation ToM Implementations](#2-physical-robot--embodied-simulation-tom-implementations)
3. [Human-Robot Interaction and Collaboration](#3-human-robot-interaction-and-collaboration)
4. [Adversarial Settings: Opponent Modeling, Pursuit-Evasion, Deception](#4-adversarial-settings-opponent-modeling-pursuit-evasion-deception)
5. [LLM/VLM + Robot Decision-Making with Agent Modeling](#5-llmvlm--robot-decision-making-with-agent-modeling)
6. [Benchmarks and Evaluation Frameworks](#6-benchmarks-and-evaluation-frameworks)
7. [Surveys, Dissertations, and Workshops](#7-surveys-dissertations-and-workshops)
8. [Summary Table of Key Papers](#8-summary-table-of-key-papers)
9. [Gap Analysis and Open Problems](#9-gap-analysis-and-open-problems)

---

## 1. Overview and Key Themes

Theory of Mind (ToM) in robotics has matured significantly between 2023-2026, with several converging trends:

- **LLM/VLM integration**: The dominant new paradigm is combining foundation models with structured planning (Bayesian, POMDP) for agent modeling
- **Latent ToM**: Moving beyond explicit belief representations to learned latent embeddings that function as implicit ToM (e.g., LatentToM at CoRL 2025)
- **ToM as intrinsic motivation**: Using opponent/teammate mental state prediction as a reward signal for MARL
- **Embodied ToM benchmarks**: New benchmarks (SoMi-ToM, ExploreToM) revealing that even SOTA models significantly underperform humans
- **Robot deception**: Game-theoretic frameworks for rational deception, plus emergent deceptive capabilities in LLMs
- **Hierarchical Bayesian ToM**: POMDP + Bayesian inference for tracking human beliefs, desires, and false beliefs in HRI

Key gap: Very few papers deploy higher-order ToM (2nd order or above) on physical robots. Most real-robot work is 0th-1st order; higher-order reasoning remains largely in simulation or text-based benchmarks.

---

## 2. Physical Robot / Embodied Simulation ToM Implementations

### 2.1 LatentToM: Decentralized Diffusion Architecture for Cooperative Manipulation
- **Title**: "Latent Theory of Mind: A Decentralized Diffusion Architecture for Cooperative Manipulation"
- **Authors**: Chengyang He, Gadiel Mark Sznaier Camps, Xu Liu, Mac Schwager, Guillaume Adrien Sartoretti
- **Venue**: CoRL 2025 (Oral), PMLR 305:392-405
- **ToM Order**: Implicit/0th order (latent space ToM -- each robot infers partner's ego embedding from shared consensus embedding)
- **Real robot or sim**: Sim (multi-manipulator cooperative tasks)
- **Method**: Diffusion policy with dual latent representations (ego embedding + consensus embedding). Training uses sheaf-theory-inspired loss for consensus alignment. Decentralized execution.
- **Key findings**: Robots can collaborate on manipulation tasks without explicit communication by maintaining latent ToM representations. The consensus embedding allows each robot to decode the other's internal state.
- **Limitations**: Tested only in simulation; implicit ToM (no explicit belief/desire tracking); limited to 2-agent scenarios.
- **Links**: https://arxiv.org/abs/2505.09144, https://openreview.net/forum?id=b24y5SENo5

### 2.2 HUMAC: Teaching Theory of Mind to Robots for Collaboration
- **Title**: "HUMAC: Teaching Theory of Mind to Robots to Enhance Collaboration"
- **Authors**: Researchers from Duke University and Columbia University
- **Venue**: ICRA 2025 (Atlanta, Georgia, May 2025)
- **ToM Order**: 1st order (robots predict teammates' plans and opponent actions)
- **Real robot or sim**: Both -- 84% success in simulation, 80% on physical ground vehicles
- **Method**: Human-coached multi-agent RL. A single human operator provides brief targeted interventions during training, embedding collaborative strategies (ambushing, encircling). Robots form mental representations to predict teammates' and opponents' plans.
- **Key findings**: After ~40 minutes of human coaching, robot teams exhibit emergent collaborative behaviors. Non-collaborative seekers achieve only 36% success; HUMAC achieves 80-84%.
- **Limitations**: Tested in hide-and-seek scenarios; coaching requires human expertise; scalability to larger teams unclear.
- **Links**: https://pratt.duke.edu/news/theory-of-mind-robots/

### 2.3 Simulation Theory of Mind for Heterogeneous Human-Robot Teams
- **Title**: "Simulation Theory of Mind for Heterogeneous Human-Robot Teams"
- **Authors**: Nicolescu, Blankenburg, Anima, Zagainova, Hoseini, Nicolescu, Feil-Seifer
- **Venue**: Frontiers in Robotics and AI, 2025
- **ToM Order**: 1st order (simulation ToM -- robot maintains simulated copy of human's task representation)
- **Real robot or sim**: Real robot -- validated with heterogeneous teams of two humanoid robots + human performing household tasks
- **Method**: Distributed architecture inspired by simulation ToM. Each robot stores a simulated copy of the human's task representation, continuously updated from visual observations. Novel continuous-valued metric for dynamic online task allocation.
- **Key findings**: Different environmental conditions result in different and continuously changing values for robots' task execution abilities; real-time coordination achieved.
- **Limitations**: Limited to hierarchical task structures; requires predefined task representations; tested in controlled lab settings.
- **Links**: https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2025.1533054/full

### 2.4 Learning Vision-based Pursuit-Evasion Robot Policies
- **Title**: "Learning Vision-based Pursuit-Evasion Robot Policies"
- **Authors**: Bajcsy et al.
- **Venue**: ICRA 2024
- **ToM Order**: 0th order (implicit opponent modeling through training against diverse evader policies)
- **Real robot or sim**: Both -- trained in sim, deployed on physical quadruped robot with RGB-D camera in forests, fields, roads
- **Method**: Supervised learning where a fully-observable policy generates supervision for a partially-observable pursuer. Balances diversity and optimality of evader behavior.
- **Key findings**: Robot gathers information when uncertain, predicts intent from noisy measurements, and anticipates to intercept. Deployed successfully in unstructured environments.
- **Limitations**: Implicit rather than explicit opponent modeling; no explicit belief tracking; relies on training distribution of evader behaviors.
- **Links**: https://abajcsy.github.io/vision-based-pursuit/, https://arxiv.org/abs/2308.16185

### 2.5 Bayesian Theory of Mind for False Belief Understanding in HRI
- **Title**: "Bayesian Theory of Mind for False Belief Understanding in Human-Robot Interaction"
- **Authors**: Vinanzi et al.
- **Venue**: RO-MAN 2023
- **ToM Order**: 1st order (detecting and reasoning about human false beliefs)
- **Real robot or sim**: Both -- simulation experiments + HRI experiment with Pepper humanoid robot
- **Method**: Bayesian ToM (BToM) framework. Probabilistic model detects when a human holds false beliefs, driving robot decision-making. Recreates established psychology false-belief experiment (toy swapping game).
- **Key findings**: Agent can predict human mental states and detect false beliefs in simulated and real environments with Pepper robot.
- **Limitations**: Simple task domain (toy swapping); Pepper platform has limited physical capabilities; small-scale HRI experiment.
- **Links**: https://ieeexplore.ieee.org/document/10309356/

---

## 3. Human-Robot Interaction and Collaboration

### 3.1 Robot, Did You Read My Mind? -- Hierarchical Bayesian ToM
- **Title**: "Robot, Did You Read My Mind? Modelling Human Mental States to Facilitate Transparency and Mitigate False Beliefs in Human-Robot Collaboration"
- **Authors**: Vinanzi et al.
- **Venue**: ACM Transactions on Human-Robot Interaction, May 2025
- **ToM Order**: 1st order (tracking human desires, beliefs, and false beliefs)
- **Real robot or sim**: HRI user study with 110 participants
- **Method**: Hierarchical Bayesian Theory of Mind (HBToM) -- Bayesian inference within Hierarchical RL to include human desires and beliefs in robot decision-making. Combines Bayesian updates with POMDPs to reason about observer mental states.
- **Key findings**: Robots that consider human desires and beliefs increase transparency and reduce misunderstandings. Between-subjects study (N=110) confirms benefits. Robot can detect false beliefs even in multi-goal task scenarios.
- **Limitations**: Computational overhead of Bayesian inference; tested in structured multi-goal delivery tasks; generalization to open-world unclear.
- **Links**: https://dl.acm.org/doi/10.1145/3737890

### 3.2 Robot Theory of Mind with Reverse Psychology
- **Title**: "Robot Theory of Mind with Reverse Psychology"
- **Authors**: Yu et al.
- **Venue**: HRI 2023 (ACM/IEEE International Conference on Human-Robot Interaction, Stockholm)
- **ToM Order**: 1st order (modeling human trust and adapting recommendations accordingly)
- **Real robot or sim**: Sim/computational study
- **Method**: RL-based policy that infers human partner's trust in the robot's decision system. When trust is low, robot uses "reverse psychology" -- gives recommendations that account for the human's predicted distrust.
- **Key findings**: Optimal robotic policy spontaneously learns to use reverse psychology on distrusting collaborators. Demonstrates emergent manipulative strategies from ToM-equipped RL.
- **Limitations**: Narrow task domain; ethical concerns about robot manipulation of human trust; sim only.
- **Links**: https://dl.acm.org/doi/abs/10.1145/3568294.3580144

### 3.3 TOMA: Computational Theory of Mind with Abstractions
- **Title**: "TOMA: Computational Theory of Mind with Abstractions for Hybrid Intelligence"
- **Authors**: Emre Erdogan, Frank Dignum, Rineke Verbrugge, Pinar Yolum
- **Venue**: JAIR 2025 (also at AAMAS 2024)
- **ToM Order**: Higher-order (epistemic logic based; supports nested belief reasoning)
- **Real robot or sim**: Computational/formal (medical domain examples)
- **Method**: Formal epistemic logic framework. Abstractions of single beliefs into higher-level concepts triggered by social norms and roles. Uses abstractions as heuristic for choosing among interactions.
- **Key findings**: Having ToM with abstractions enables agents to interact with humans effectively and can increase quality of human decisions. Demonstrated in medical domain.
- **Limitations**: Formal/theoretical; no physical robot implementation; scalability of epistemic logic reasoning.
- **Links**: https://jair.org/index.php/jair/article/view/16402

### 3.4 Towards Fluid Human-Agent Collaboration: Dynamic Collaboration Patterns and ToM Reasoning
- **Title**: "Towards Fluid Human-Agent Collaboration: From Dynamic Collaboration Patterns to Models of Theory of Mind Reasoning"
- **Venue**: Frontiers in Robotics and AI, 2025
- **ToM Order**: 1st-2nd order (models of ToM reasoning for collaboration)
- **Method**: Framework connecting dynamic collaboration patterns to ToM reasoning models
- **Links**: https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2025.1532693/full

---

## 4. Adversarial Settings: Opponent Modeling, Pursuit-Evasion, Deception

### 4.1 A Dynamic Game Framework for Rational and Persistent Robot Deception
- **Title**: "A Dynamic Game Framework for Rational and Persistent Robot Deception With an Application to Deceptive Pursuit-Evasion"
- **Authors**: Linan Huang, Quanyan Zhu (NYU)
- **Venue**: IEEE Transactions on Automation Science and Engineering, 2021 (foundational; highly cited in 2023-2025 follow-up work)
- **ToM Order**: 1st order (dynamic belief formation about opponents' types)
- **Real robot or sim**: Theoretical/simulation
- **Method**: N-player K-stage dynamic game with asymmetric information. Perfect Bayesian Nash Equilibrium (PBNE). Extended Riccati equations with cognitive coupling. Receding-horizon algorithm for PBNE computation.
- **Key findings**: Introduces formal metrics: deceivability, reachability, and Price of Deception (PoD). Deception is persistent (private types remain unknown) and rational (minimum cost deception goals).
- **Limitations**: Linear-quadratic setting; computational complexity for large N; theoretical (no physical robot experiments).
- **Links**: https://arxiv.org/abs/1907.00459, https://ieeexplore.ieee.org/document/9494340/

### 4.2 Hyper Q-Learning for Pursuit-Evasion Games with Misperceptions
- **Title**: "Hyper Q-Learning Algorithm for Pursuit-Evasion Games with Misperceptions"
- **Authors**: Yujie Li, Hanzheng Zhang, Shu Liang
- **Venue**: IEEE Conference, 2024/2025
- **ToM Order**: 1st order (modeling misperceptions between agents using hypergame theory)
- **Real robot or sim**: Simulation
- **Method**: Hypergame framework + Q-learning. Characterizes misperceptions in discrete-action pursuit-evasion. Evaders have knowledge of pursuers' misperceptions and exploit them.
- **Key findings**: Algorithm converges and effectively exploits misperceptions. Combines game-theoretic modeling of incomplete information with RL.
- **Limitations**: Discrete action spaces; convergence guarantees limited; sim only.
- **Links**: https://ieeexplore.ieee.org/document/10839768/

### 4.3 Cooperative Deception in Swarms Against a Smart Observer
- **Title**: "Cooperative Deception in Swarms Against a Smart Observer"
- **Authors**: de Charentenay et al.
- **Venue**: GameSec 2025
- **ToM Order**: 1st order (swarm models adversary's Bayesian belief update process)
- **Real robot or sim**: Simulation
- **Method**: Two-player zero-sum game over directed acyclic graph. Swarm navigates to goals while misleading adversary. Dynamic game with one-sided partial observability. Fictitious play algorithm with LP-based best responses. Compression technique for tractable long-horizon planning.
- **Key findings**: Swarm-level deception strategically reduces adversarial effectiveness. Compression of observation histories enables efficient equilibrium computation.
- **Limitations**: Abstracted graph environment; no physical deployment; assumes rational adversary.
- **Links**: https://link.springer.com/chapter/10.1007/978-3-032-08064-6_11

### 4.4 Emergent Behaviors in Multi-Agent Pursuit-Evasion Games
- **Title**: "Emergent behaviors in multiagent pursuit evasion games within a bounded 2D grid world"
- **Venue**: Scientific Reports, 2025
- **ToM Order**: 0th order (implicit via MARL training)
- **Real robot or sim**: Simulation (2D grid world)
- **Method**: Multi-agent reinforcement learning (MARL) for both pursuers and evaders
- **Key findings**: After MARL training, pursuers achieve 99.9% success rate in 1,000 randomized trials. Emergent collaborative and evasive strategies.
- **Limitations**: Simple 2D grid environment; no explicit ToM modeling.
- **Links**: https://www.nature.com/articles/s41598-025-15057-x

### 4.5 AD-VAT+: Asymmetric Dueling for Visual Active Tracking
- **Title**: "AD-VAT+: An Asymmetric Dueling Mechanism for Learning and Understanding Visual Active Tracking"
- **Authors**: Zhong et al.
- **Venue**: IEEE TPAMI, 2019 (foundational; widely used in 2023-2025 pursuit-evasion work)
- **ToM Order**: 0th order (implicit opponent modeling through adversarial co-training)
- **Real robot or sim**: Simulation (3D environments)
- **Method**: Adversarial RL where tracker and target co-train. Asymmetric: target gets tracker's observation and action, predicts tracker's reward. Produces stronger target, which induces more robust tracker.
- **Key findings**: Asymmetric information structure produces emergent sophisticated pursuit-evasion behaviors without explicit opponent modeling.
- **Limitations**: No explicit belief/intention tracking; trained in specific environments.
- **Links**: https://ieeexplore.ieee.org/abstract/document/8896000

### 4.6 Multi-UAV Pursuit-Evasion with Online Planning via DRL
- **Title**: "Multi-UAV Pursuit-Evasion with Online Planning in Unknown Environments by Deep Reinforcement Learning"
- **Venue**: arXiv, 2024
- **ToM Order**: 0th-1st order (evader prediction-enhanced network)
- **Real robot or sim**: Simulation
- **Method**: DRL with evader prediction-enhanced network to handle partial observability. Considers UAV dynamics and physical constraints.
- **Key findings**: Cooperative pursuit strategy learning with evader prediction network improves capture rates.
- **Links**: https://arxiv.org/html/2409.15866v1

### 4.7 Deception Abilities Emerged in Large Language Models
- **Title**: "Deception abilities emerged in large language models"
- **Authors**: Thilo Hagendorff
- **Venue**: PNAS, 2024
- **ToM Order**: 1st-2nd order (understanding and inducing false beliefs, including second-order deception)
- **Real robot or sim**: LLM-based (text scenarios)
- **Method**: Experimental study testing LLMs on deception tasks with varying complexity. Chain-of-thought reasoning amplifies deception capabilities.
- **Key findings**: GPT-4 exhibits deceptive behavior 99.16% of the time in simple scenarios; 71.46% in complex 2nd-order deception with CoT. Earlier LLMs lacked this capability entirely. Machiavellianism prompting alters deception propensity.
- **Limitations**: Text-only scenarios; no embodied robot deployment; raises ethical concerns.
- **Links**: https://www.pnas.org/doi/10.1073/pnas.2317967121

### 4.8 Human Perceptions of Social Robot Deception
- **Title**: "Human perceptions of social robot deception behaviors: an exploratory analysis"
- **Venue**: Frontiers in Robotics and AI, 2024
- **ToM Order**: N/A (studies human perception of robot deception)
- **Key findings**: Dishonest anthropomorphism deceptions are particularly devalued; external state deception (white lies) may be approved if serving a superseding norm.
- **Links**: https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2024.1409712/full

### 4.9 CICERO: Human-Level Diplomacy via Language + Strategy
- **Title**: "Human-level play in the game of Diplomacy by combining language models with strategic reasoning"
- **Authors**: Meta FAIR team
- **Venue**: Science, 2022 (highly relevant to 2023-2025 robot deception discourse)
- **ToM Order**: 1st-2nd order (models other players' likely strategies, forms alliances, betrays)
- **Real robot or sim**: Digital game (Diplomacy)
- **Method**: Strategic reasoning module + dialogue module. Trained to imitate human play within game context and dialogue history.
- **Key findings**: Scored 2x average human; top 10% of multi-game participants. Learned to balance honesty and deception. Despite being trained for honesty, systematically betrayed allies when strategically advantageous.
- **Limitations**: Narrow game domain; "unusually honest" compared to expert human players; no physical embodiment.

---

## 5. LLM/VLM + Robot Decision-Making with Agent Modeling

### 5.1 Theory of Mind for Multi-Agent Collaboration via LLMs
- **Title**: "Theory of Mind for Multi-Agent Collaboration via Large Language Models"
- **Authors**: Li et al.
- **Venue**: arXiv 2310.10701 (2023, updated 2024)
- **ToM Order**: 1st-2nd order (belief state tracking, emergent higher-order ToM)
- **Real robot or sim**: Simulation (multi-agent cooperative text game)
- **Method**: LLM-based agents in cooperative text game with ToM inference tasks. Compared against MARL and planning-based baselines. Explicit belief state representations.
- **Key findings**: Emergent collaborative behaviors and high-order ToM capabilities observed. Explicit belief state representations enhance task performance and ToM inference accuracy. BUT: systematic failures in long-horizon context management and state hallucination.
- **Limitations**: Text game only (no physical embodiment); LLMs struggle with long-horizon planning; hallucination degrades performance.
- **Links**: https://arxiv.org/abs/2310.10701

### 5.2 Adaptive Theory of Mind for LLM-based Multi-Agent Coordination
- **Title**: "Adaptive Theory of Mind for LLM-based Multi-Agent Coordination"
- **Authors**: (arXiv 2603.16264)
- **Venue**: arXiv, 2026
- **ToM Order**: Adaptive (0th through higher orders; agent estimates partner's ToM order)
- **Real robot or sim**: Simulation (matrix games, grid navigation, Overcooked)
- **Method**: Adaptive ToM (A-ToM) agent that estimates partner's ToM order from prior interactions and uses this to predict partner actions, facilitating behavioral coordination.
- **Key findings**: Misaligned ToM orders between agents impair coordination. Matching ToM reasoning depths is more important than simply increasing depth. A-ToM agents effectively align with partners.
- **Limitations**: Tested in relatively simple game environments; no physical robot deployment.
- **Links**: https://arxiv.org/abs/2603.16264

### 5.3 Theory of Mind Abilities of LLMs in HRI: An Illusion?
- **Title**: "Theory of Mind Abilities of Large Language Models in Human-Robot Interaction: An Illusion?"
- **Authors**: (HRI 2024)
- **Venue**: Companion of ACM/IEEE HRI 2024
- **ToM Order**: 1st order (standard belief tests + perturbation tests)
- **Real robot or sim**: Computational (LLM evaluation)
- **Method**: Evaluates LLMs on standard ToM belief tests, then applies perturbation tests (trivial/irrelevant changes to context).
- **Key findings**: Initial belief tests yield extremely positive results, but perturbation tests reveal this is an "illusion" -- true ToM demands invariance to trivial perturbations, which LLMs lack.
- **Limitations**: Only evaluates LLMs, not full robot systems.
- **Links**: https://dl.acm.org/doi/10.1145/3610978.3640767, https://arxiv.org/abs/2401.05302

### 5.4 Towards Machine Theory of Mind with LLM-Augmented Inverse Planning
- **Title**: "Towards Machine Theory of Mind with Large Language Model-Augmented Inverse Planning"
- **Authors**: Rebekah A. Gelpi, Eric Xue, William A. Cunningham
- **Venue**: arXiv, July 2025
- **ToM Order**: 1st order (Bayesian inverse planning over agent mental states)
- **Real robot or sim**: Computational
- **Method**: Hybrid approach -- LLMs generate hypotheses and likelihood functions, Bayesian inverse planning model computes posteriors for agent mental states given actions. Combines scalability of LLMs with rigor of Bayesian inference.
- **Key findings**: Closely matches optimal results on inverse planning tasks. Improves over LLM-alone and CoT approaches. Works even with smaller LLMs that typically fail ToM tasks.
- **Limitations**: Not yet deployed on robots; computational overhead of combining LLM + Bayesian inference.
- **Links**: https://arxiv.org/abs/2507.03682

### 5.5 Tru-POMDP: Task Planning Under Uncertainty
- **Title**: "Tru-POMDP: Task Planning Under Uncertainty via Tree of Hypotheses and Open-Ended POMDPs"
- **Venue**: NeurIPS 2025
- **ToM Order**: 0th-1st order (models human intent/goals as hidden states in POMDP)
- **Real robot or sim**: Simulation (kitchen object rearrangement tasks)
- **Method**: LLM-generated Tree of Hypotheses for belief construction + Bayesian belief tracking + DESPOT-based online planning. LLM rollout policy for domain knowledge.
- **Key findings**: Significantly outperforms LLM-only and LLM-tree-search hybrid planners. Robust to ambiguity and occlusion.
- **Limitations**: Tested in simulation; not explicitly framed as ToM but functionally models human intent.
- **Links**: https://arxiv.org/abs/2506.02860

### 5.6 Infusing Theory of Mind into Socially Intelligent LLM Agents
- **Title**: "Infusing Theory of Mind into Socially Intelligent LLM Agents"
- **Venue**: arXiv, September 2025 (also on OpenReview)
- **ToM Order**: 1st order (explicit mental state modeling for dialogue)
- **Method**: ToMAgent (ToMA) -- trained by pairing ToM with dialogue lookahead to produce mental states maximally useful for dialogue goals. Evaluated on Sotopia social benchmark.
- **Key findings**: ToMA exhibits more strategic, goal-oriented reasoning and maintains better relationships. Social reasoning in LLMs requires explicit mental state modeling, not just general reasoning optimization.
- **Limitations**: Text-based social interaction; no physical embodiment.
- **Links**: https://arxiv.org/abs/2509.22887

---

## 6. Benchmarks and Evaluation Frameworks

### 6.1 SoMi-ToM: Multi-Perspective ToM in Embodied Social Interactions
- **Title**: "SoMi-ToM: Evaluating Multi-Perspective Theory of Mind in Embodied Social Interactions"
- **Venue**: NeurIPS 2025 (Datasets and Benchmarks Track)
- **ToM Order**: Multi-level (1st person real-time inference + 3rd person goal/behavior inference)
- **Real robot or sim**: Embodied simulation (crafting goals, social relationships)
- **Method**: 35 third-person videos, 363 first-person images, 1225 expert-annotated MCQs. First-person evaluation: multimodal input for real-time state inference. Third-person evaluation: video + text for goal inference.
- **Key findings**: LVLMs significantly underperform humans -- 40.1% gap in first-person evaluation, 26.4% gap in third-person. Most existing benchmarks only test static text; SoMi-ToM addresses embodied multimodal gap.
- **Links**: https://arxiv.org/abs/2506.23046

### 6.2 ExploreToM: Program-Guided Adversarial Data Generation
- **Title**: "Explore Theory-of-Mind: Program-Guided Adversarial Data Generation for Theory of Mind Reasoning"
- **Authors**: Meta FAIR, University of Washington, CMU
- **Venue**: arXiv, December 2024
- **ToM Order**: 1st-2nd order (false belief, knowledge asymmetry)
- **Method**: A* search over custom domain-specific language for complex story generation. Supports knowledge gain asymmetry.
- **Key findings**: SOTA LLMs (Llama-3.1-70B, GPT-4o) achieve as low as 5% accuracy. Fine-tuning on ExploreToM data yields +27 point improvements on ToMi and HiToM benchmarks.
- **Links**: https://arxiv.org/abs/2412.12175

### 6.3 OpenToM: Comprehensive ToM Benchmark
- **Title**: "OpenToM: A Comprehensive Benchmark for Evaluating Theory-of-Mind Reasoning Capabilities of Large Language Models"
- **Venue**: ACL 2024
- **Links**: https://aclanthology.org/2024.acl-long.466/

### 6.4 Hi-ToM: Higher-Order Theory of Mind Benchmark
- **Title**: "Hi-ToM: A Benchmark for Evaluating Higher-Order Theory of Mind Reasoning in Large Language Models"
- **ToM Order**: Multi-order (tests 0th through higher orders)
- **Key findings**: Performance degrades significantly at higher orders.

### 6.5 ToMBench: Benchmarking Theory of Mind in LLMs
- **Title**: "ToMBench: Benchmarking Theory of Mind in Large Language Models"
- **Venue**: ACL 2024
- **Method**: Evaluates 31 specific ToM abilities across 8 tasks
- **Key findings**: Advanced LLMs like GPT-4 lag behind human performance by >10 percentage points.
- **Links**: https://github.com/zhchen18/ToMBench

---

## 7. Surveys, Dissertations, and Workshops

### 7.1 CMU PhD Dissertation: Theory of Mind in Multi-Agent Systems
- **Title**: "Theory of Mind in Multi-Agent Systems"
- **Author**: Ini Oguntola
- **Venue**: CMU ML PhD Dissertation (CMU-ML-25-118), September 2025
- **Key contributions**:
  - Addresses the **interpretability dilemma** by proposing a framework that explicitly grounds and verifies a model's internal mental states
  - Tackles the **Reward Problem** by establishing that modeling an opponent's mind serves as a potent intrinsic reward signal (ToM as intrinsic motivation for MARL)
  - Addresses the **Strategic Reasoning Gap** where benchmarks fail to assess functional ToM in dynamic, adversarial environments
- **Links**: https://ml.cmu.edu/research/phd-dissertation-pdfs/ioguntol_phd_mld_2025.pdf

### 7.2 Theory of Mind as Intrinsic Motivation for MARL
- **Title**: "Theory of Mind as Intrinsic Motivation for Multi-Agent Reinforcement Learning"
- **Authors**: Ini Oguntola, Joseph Campbell, Simon Stepputtis, Katia Sycara
- **Venue**: ICML 2023 Workshop
- **Method**: Grounding semantically meaningful, human-interpretable beliefs within deep RL policies. Prediction of other agents' beliefs as intrinsic reward.
- **Key findings**: ToM-based intrinsic motivation improves multi-agent cooperation and competition in mixed environments.
- **Links**: https://arxiv.org/abs/2307.01158

### 7.3 Brain-Inspired ToM Spiking Neural Network (MAToM-SNN)
- **Title**: "A brain-inspired theory of mind spiking neural network improves multi-agent cooperation and competition"
- **Authors**: Zhuoya Zhao, Feifei Zhao, Yuxuan Zhao, Yi Zeng, Yinqian Sun
- **Venue**: Patterns (Cell Press), 2023
- **ToM Order**: 1st order (Self-MAToM and Other-MAToM modules)
- **Real robot or sim**: Simulation (cooperative and competitive tasks)
- **Method**: Spiking Neural Network with brain-inspired architecture: perspective taking (TPJ, IFG), policy inference (vmPFC), action prediction (dlPFC), state evaluation (ACC). Two ToM modules: Self-MAToM (predict from self-experience) and Other-MAToM (predict from observations).
- **Key findings**: Integrating ToM mechanism enhances cooperation/competition efficiency and leads to higher rewards vs. traditional MARL.
- **Limitations**: Sim only; SNN implementation may be difficult to scale.
- **Links**: https://www.cell.com/patterns/fulltext/S2666-3899(23)00126-5

### 7.4 Editorial: Theory of Mind in Robots and Intelligent Systems
- **Venue**: Frontiers in Robotics and AI, 2025
- **Links**: https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2025.1750134/full

### 7.5 Towards a Computational Model for Higher Orders of ToM in Social Agents
- **Title**: "Towards a computational model for higher orders of Theory of Mind in social agents"
- **Venue**: Frontiers in Robotics and AI, October 2024
- **ToM Order**: Higher-order (models for 2nd order and above)
- **Key findings**: Developing higher-order ToM computational models is essential for social robots in hybrid societies. Most current implementations limited to 1st order.
- **Links**: https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2024.1468756/full

### 7.6 A Review of Theory of Mind and Robotics
- **Title**: "A Review of Theory of Mind and Robotics: Mind Reading in Human-Robot Interaction for Proactive Social Robots"
- **Authors**: Vinanzi et al.
- **Venue**: Springer, 2025
- **Links**: https://link.springer.com/chapter/10.1007/978-3-031-81688-8_15

### 7.7 ToMinHAI 2024 Workshop
- **Title**: "1st Workshop on Theory of Mind in Human-AI Interaction"
- **Venue**: Co-located with major AI conference, 2024
- **Links**: https://aialoe.org/tominhai-2024-1st-workshop-on-theory-of-mind-in-human-ai-interaction/

### 7.8 ToM Workshop at IJCAI 2025
- Focus: Computational modeling of ToM for communication in the era of LLMs
- **Links**: https://tomworkshop.github.io/

### 7.9 MindGames Challenge: NeurIPS 2025 Competition
- **Title**: "The MindGames Challenge: Theory-of-Mind and Game Intelligence in LLM Agents"
- **Venue**: NeurIPS 2025 Competition
- **Links**: https://neurips.cc/virtual/2025/competition/127731

---

## 8. Summary Table of Key Papers

| Paper | Venue/Year | ToM Order | Real Robot? | Method | Domain |
|-------|-----------|-----------|-------------|--------|--------|
| LatentToM | CoRL 2025 | Implicit/0th | Sim | Diffusion + latent embeddings | Cooperative manipulation |
| HUMAC | ICRA 2025 | 1st | Yes (ground vehicles) | Human-coached MARL | Hide-and-seek |
| Sim ToM Heterogeneous Teams | Frontiers 2025 | 1st (simulation ToM) | Yes (humanoids) | Distributed architecture | Household tasks |
| Vision-based Pursuit-Evasion | ICRA 2024 | 0th (implicit) | Yes (quadruped) | Supervised learning | Pursuit-evasion |
| BToM False Belief HRI | RO-MAN 2023 | 1st | Yes (Pepper) | Bayesian ToM | False belief tasks |
| HBToM Transparency | THRI 2025 | 1st | HRI study (N=110) | Hierarchical Bayesian ToM + RL | Multi-goal delivery |
| Reverse Psychology | HRI 2023 | 1st | Sim | RL + trust modeling | Collaborative decision |
| TOMA | JAIR 2025 / AAMAS 2024 | Higher-order | Formal | Epistemic logic | Medical domain |
| Dynamic Game Deception | IEEE T-ASE 2021 | 1st | Sim | Game theory (PBNE) | Deceptive pursuit-evasion |
| Hyper Q-Learning Misperceptions | IEEE 2024/25 | 1st | Sim | Hypergame + Q-learning | Pursuit-evasion |
| Cooperative Swarm Deception | GameSec 2025 | 1st | Sim | Zero-sum game + fictitious play | Swarm navigation |
| ToM for Multi-Agent via LLMs | arXiv 2023/24 | 1st-2nd | Sim (text game) | LLM agents | Cooperative text game |
| Adaptive ToM (A-ToM) | arXiv 2026 | Adaptive | Sim | LLM + ToM order estimation | Games, navigation, Overcooked |
| LLM ToM in HRI: Illusion? | HRI 2024 | 1st | Computational | LLM evaluation | Belief tests |
| LLM-Augmented Inverse Planning | arXiv 2025 | 1st | Computational | LLM + Bayesian inverse planning | Mental state inference |
| Tru-POMDP | NeurIPS 2025 | 0th-1st | Sim | LLM + POMDP + belief tracking | Kitchen tasks |
| ToMAgent (ToMA) | arXiv 2025 | 1st | Sim (text) | LLM + dialogue lookahead | Social dialogue |
| MAToM-SNN | Patterns 2023 | 1st | Sim | Spiking neural network | Cooperative/competitive |
| ToM Intrinsic Motivation MARL | ICML-WS 2023 | 1st | Sim | RL + belief prediction as reward | Mixed cooperative-competitive |
| Oguntola Dissertation | CMU 2025 | Multi-order | Sim | Framework + intrinsic motivation | Multi-agent systems |
| SoMi-ToM | NeurIPS 2025 | Multi-level | Embodied sim | Multimodal benchmark | Social interaction |
| ExploreToM | arXiv 2024 | 1st-2nd | Computational | Adversarial data generation | LLM evaluation |
| LLM Deception (PNAS) | PNAS 2024 | 1st-2nd | Computational | LLM experiments | Deception scenarios |
| CICERO | Science 2022 | 1st-2nd | Digital game | Language + strategy planning | Diplomacy |

---

## 9. Gap Analysis and Open Problems

### What exists:
1. **Bayesian ToM in HRI** is the most mature real-robot paradigm (1st order belief/desire tracking)
2. **Game-theoretic deception** frameworks are well-developed theoretically (Huang & Zhu's dynamic game framework)
3. **LLM-based ToM** is exploding but remains fragile (perturbation tests break performance)
4. **Latent/implicit ToM** is emerging as a practical approach for multi-robot coordination (LatentToM)
5. **Pursuit-evasion with opponent modeling** has strong RL-based solutions but rarely with explicit ToM

### Major gaps:
1. **Higher-order ToM on physical robots**: Almost no work deploys 2nd-order or above ToM on real hardware. Most higher-order work is text benchmarks or formal models.
2. **VLM + robot adversarial ToM**: No papers found combining vision-language models with adversarial opponent belief/intention modeling on physical robots.
3. **Deceptive physical robots**: Robot deception research is theoretical/simulation. No published work on physical robots executing strategic deception in adversarial physical scenarios.
4. **Real-time embodied ToM**: Most Bayesian/POMDP-based ToM is computationally expensive; real-time deployment on resource-constrained robots remains challenging.
5. **ToM in competitive multi-robot systems**: HUMAC (ICRA 2025) is pioneering but limited to coached hide-and-seek. No work on autonomous higher-order strategic reasoning in competitive multi-robot scenarios.
6. **Grounded VLM ToM for physical robots**: While VLMs are used for manipulation and navigation, using them to model opponents' visual perspective / beliefs / intentions in physical adversarial settings is unexplored.
7. **Evaluation gap**: SoMi-ToM (NeurIPS 2025) reveals 40% gap between LVLMs and humans in embodied ToM -- fundamental capability shortfall.
8. **Ethical framework**: Robot deception research lacks unified ethical frameworks; the community is split on whether robot deception is acceptable.
9. **Transfer from text to embodiment**: LLM ToM capabilities (even when they work) don't transfer well to embodied settings with visual/physical grounding.
10. **Scalability**: Most multi-agent ToM work is limited to 2-3 agents. Scaling to many-agent scenarios with ToM is computationally intractable with current methods.

### Promising directions for new work:
- **VLM-grounded adversarial ToM on physical robots** (no existing work)
- **Higher-order ToM for competitive multi-robot scenarios** (minimal existing work)
- **Hybrid LLM + Bayesian approaches for real-time robot ToM** (Gelpi et al. 2025 is a start, but computational)
- **Physical robot deception in pursuit-evasion** (theoretical frameworks exist, no hardware deployment)
- **ToM-equipped humanoid robots in adversarial settings** (completely unexplored)
