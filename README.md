# Post-DeepSeek-R1
Resources and research after DeepSeek-R1, around test-time computing, resurgence of RL, and new LLM learning/application paradigms.


> This behavior is not only a testament to the model’s growing reasoning abilities but also a captivating example of how reinforcement learning can lead to unexpected and sophisticated outcomes.

-- From [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)

<div align="center">
<img src="https://github.com/jzhou316/Post-DeepSeek-R1/blob/main/images/tweet_mark_chen.png" width="500" alt="">
</div>

-- From [Mark Chen](https://x.com/markchen90/status/1884303237186216272), OpenAI Chief Research Officer

---

### Table of Contents

- [DeepSeek-R1 Reproduction](#deepseek-r1-reproduction-popular-and-fast-ones)
- [R1-like RL Reproduction for More Scenarios](#r1-like-rl-reproduction-for-more-scenarios)
  - [Tools](#tools)
  - [LLM + RL with/for X](#llm--rl-withfor-x)
- [Literature](#literature)
  - [Test-time Scaling](#test-time-scaling)
  - [Process Reward](#process-reward-after-o1)
  - [Multimodal, Image Generation](#multimodal-image-generation)
  - [RL for Different Ways of Generation](#rl-for-different-ways-of-generation)
  - [Improve Long CoT for Reasoning](#improve-long-cot-for-reasoning)
  - [Understanding R1 and RL + LLMs, Tricks to Train RL](#understanding-r1-and-rl--llms-tricks-to-train-rl)
  - [Efficiency](#efficiency)

---

## DeepSeek-R1 Reproduction ("popular" and fast ones)


#### [Simple Reinforcement Learning for Reasoning](https://github.com/hkust-nlp/simpleRL-reason?tab=readme-ov-file#simple-reinforcement-learning-for-reasoning) (HKUST)


- Rule-based reward (no MCTS and reward models)
- Uses PPO rather than GRPO
- Trains small models (7B) on limited data (8K examples)
- Starting from Qwen2.5-Math-7B (base model), performs RL on it directly, achieving surprisingly strong results

<div align="center">
<img src="https://github.com/user-attachments/assets/bacd1680-ccb0-4921-a687-8a595ebf5896" width="700" alt="simplelr-reaoning-intro-figure_00">
</div>

> Training dynamics of our Qwen2.5-SimpleRL-Zero training starting from the Qwen2.5-Math-7B, without SFT or reward models.


#### [DeepScaleR](https://github.com/agentica-project/deepscaler/tree/main?tab=readme-ov-file#deepscaler) (Berkeley)

- Aimed to democratize reinforcement learning (RL) for LLMs and reproduce DeepSeek R1 and OpenAI O1/O3 at scale
- Iteratively scaling Deepseek's GRPO algorithm from 8K→16K→24K context length for thinking
- Trained on top of [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) (_Joe: so the initial model is already capable of deep thinking; better if we can do from base models_)
- Heavily based on modified fork of [veRL](https://github.com/volcengine/verl), an open-source RLHF library
- Good insight and training receipe: error cases are initially longer CoTs, so gradually extending context length for thinking during training (_Joe: a sort of curriculum learning for RL_)

![](https://github.com/agentica-project/deepscaler/blob/main/figures/deepscaler.png)

*Figure 1: DeepScaleR 1.5B model's Pass@1 accuracy on AIME2024 as RL training progresses. At step 1040 and 1520, the context length is extended to 16K and 24K. For more details, see our [blog post](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2).*


#### [Open R1](https://github.com/huggingface/open-r1?tab=readme-ov-file#open-r1) (Hugging Face)

- Fully open reproduction of DeepSeek-R1
- [Blog post](https://huggingface.co/blog/open-r1)


#### [TinyZero](https://github.com/Jiayi-Pan/TinyZero)

- A reproduction of DeepSeek-R1-Zero in countdown and multiplication tasks
- Through RL, the 3B base LM develops self-verification and search abilities all on its own
- Fails to learn reasoning with Qwen2.5-0.5B base
- Works with [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) model
- Experiment run based on [veRL](https://github.com/volcengine/verl)

#### [Mini-R1](https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/mini-deepseek-r1-aha-grpo.ipynb)

- A minimal single notebook that tries to reproduce the DeepSeek-R1 "reasoning" results on a single task (the Countdown Game)
- Uses GRPO and Q-Lora, also with the [TRL](https://huggingface.co/docs/trl/en/index) library
- Starting with the [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) model (suggested using models > 1.5B) (_Joe: Yes, we need the model to start with to have certain capabilities_)
- Good learning material with code


#### [Oat-Zero](https://github.com/sail-sg/oat-zero?tab=readme-ov-file#there-may-not-be-aha-moment-in-r1-zero-like-training--a-pilot-study)

There May Not be Aha Moment in R1-Zero-like Training — A Pilot Study

- Aha moment (such as self-reflection patterns) may already exist in the base model.
- There are Superficial Self-Reflection (SSR) from base models' responses, in which case self-reflections do not necessarily lead to correct final answers.
- Closer look at R1-Zero-like training via RL, and found that the increasing response length phenomenon is not due to the emergence of self-reflection, but a consequence of RL optimizing well-designed rule-based reward functions.

#### [Open Reasoner Zero](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero?tab=readme-ov-file#open-reasoner-zero)

- An Open Source Approach to Scaling Up Reinforcement Learning on the Base Model
- Uses PPO (instead of GRPO; some [discussions](https://x.com/rosstaylor90/status/1892664646890312125))


#### [Colab Reproductions with Unsloth](https://unsloth.ai/blog/r1-reasoning)

- One GPU with GRPO (worth trying when resource constraint)
- Experience the "aha moment" for [free on Colab](https://x.com/danielhanchen/status/1887564724071768529) (seems easy to play with)


### Online Materials, Discussions

- Video tutorials from Sasha Rush on [o1-like test-time scaling](https://github.com/srush/awesome-o1) and [DeepSeek](https://www.youtube.com/watch?v=KtBcIDtS13M)
- [Some takeaways from the R1, DeepSeek-V3  and GRPO papers](https://x.com/Dan_Jeffries1/status/1881679981849215080) (twitter)


### Other RL Trained Models

[(2025 Mar) QwQ-32B: Embracing the Power of Reinforcement Learning](https://qwenlm.github.io/blog/qwq-32b/)



## R1-like RL Reproduction for More Scenarios

### Tools

- RL libraries:
  - [veRL](https://github.com/volcengine/verl) (seems most popular as of Mar 2025). Check this [list](https://github.com/volcengine/verl?tab=readme-ov-file#awesome-work-using-verl) of R1 followup works
  - [TRL](https://huggingface.co/docs/trl/en/index)
  - Inference: [vLLM](https://github.com/vllm-project/vllm) seems a must to speed up inference
- Starting models: [Qwen2.5](https://github.com/QwenLM/Qwen2.5) (base, instruct, R1-distilled, math) seems most popular (as of Mar 2025) (why? some [empirical answers](https://arxiv.org/abs/2503.01307)), both 3B and 7B models are made work; 0.5B is a bit weaker but could also learn
- RL algorithms: [GRPO](https://arxiv.org/abs/2402.03300), [PPO](https://arxiv.org/pdf/1707.06347) (some dispute on whether GRPO is the must, [here](https://github.com/ZihanWang314/ragen?tab=readme-ov-file#-ragen-training-agents-by-reinforcing-reasoning-) and [here](https://x.com/finbarrtimbers/status/1899118175830397322))
  - some tutorials [here](https://anukriti-ranjan.medium.com/preference-tuning-llms-ppo-dpo-grpo-a-simple-guide-135765c87090#:~:text=GRPO%2C%20from%20DeepSeek%20AI%2C%20is,making%20it%20lighter%20and%20faster.) and [here](https://huggingface.co/blog/NormalUhr/grpo)
- GPU resourse: see the other reproductions, and discussion e.g. [here](https://github.com/huggingface/open-r1/issues/100)
  - One GPU with GRPO on [Colab](https://unsloth.ai/blog/r1-reasoning)

### LLM + RL with/for X


#### [RAGEN: Training Agents by Reinforcing Reasoning](https://github.com/ZihanWang314/ragen?tab=readme-ov-file#-ragen-training-agents-by-reinforcing-reasoning-)
RL + LLM applied to **agents**
- Using PPO instead of GRPO

#### [Logic-RL](https://github.com/Unakar/Logic-RL?tab=readme-ov-file#logic-rl)
RL + LLM applied with **synthetic logic puzzles** with controllable complexity and straightforward answer verification

#### [Teaching Language Models to Critique via Reinforcement Learning](https://github.com/HKUNLP/critic-rl?tab=readme-ov-file#-teaching-language-models-to-critique-via-reinforcement-learning-)
RL + LLM applied to **coding**
- Train with GRPO using verifiable rewards from sandbox execution

#### [Code-R1: Reproducing R1 for Code with Reliable Rewards](https://github.com/ganler/code-r1?tab=readme-ov-file#code-r1-reproducing-r1-for-code-with-reliable-rewards)
RL + LLM applied to **coding**

#### [EasyR1: An Efficient, Scalable, Multi-Modality RL Training Framework](https://github.com/hiyouga/EasyR1)
RL + LLM applied to **multimodality** (such as VLMs)

#### [R1-Omni: Explainable Omni-Multimodal Emotion Recognition with Reinforcement Learning](https://arxiv.org/abs/2503.05379)
RL + LLM applied to **multimodality**

- For the specific task of emotion recognition, with visual and audio signals (videos)
- Learning with a 0.5B model

#### [Omni-R1: Do You Really Need Audio to Fine-Tune Your Audio LLM?](https://arxiv.org/abs/2505.09439)
RL + LLM applied to **multimodality**

- Audio LLM, fine-tuned with GRPO

#### [SpatialReasoner: Towards Explicit and Generalizable 3D Spatial Reasoning](https://arxiv.org/abs/2504.20024)
RL + LLM applied to **multimodality**

- Spatial reasoning with 3D augmented input parsed from images

#### [Search-R1: Train your LLMs to reason and call a search engine with reinforcement learning](https://github.com/PeterGriffinJin/Search-R1?tab=readme-ov-file#search-r1-train-your-llms-to-reason-and-call-a-search-engine-with-reinforcement-learning)

RL + LLM applied to **retrieval** (interleaved with generaion/reasoning)
- Tested on NQ dataset, retrieving from Wikipedia

#### [ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning](https://github.com/Agent-RL/ReSearch)
RL + LLM applied to **retrieval** (RAG)
- Trained with HotpotQA data

#### [DeepRetrieval - Hacking Search Engines & Retrievers with LLM + RL](https://github.com/pat-jj/DeepRetrieval)
RL + LLM applied to **retrieval**
- Tested on literature mining, publication search and trial search tasks


---

## Literature

Here is a collection of papers of different topics and flavors. They are not (cannot be) exhaustive, but grouped based on their themes to give some sense of different types of research and problems in the space.

_Joe: I marked the year with month for papers, due to the extreme fast pace in this domain of exploding research_


### Test-time Scaling

[(2024 Aug) Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](https://arxiv.org/abs/2408.03314)

-> Test-time scaling for math

- Includes search strategies such as Best-of-N, beam search, and beam search with lookahead
- Involves process reward model (PRM) and revision models

(2024 Nov) Deliberative Alignment: Reasoning Enables Safer Language Models

-> Test-time scaling for safety

[(2025 Jan) s1: Simple test-time scaling](https://arxiv.org/abs/2501.19393)

-> Test-time scaling for reasoning

- Collected 1K datapoints from diverse datasets and their reasoning traces (from Google Gemini Flash Thinking API), and then a pipeline of quality control and filtering
- Finetune Qwen2.5-32B-Instruct on the 1K datapoints, with training takes just 26 minutes on 16 NVIDIA H100 GPUs
- Control the test-time compute in the sequential generation scenario (as opposed to parallel like search or best of N). Control the reasoning length by inserting tokens "Final Answer:" and "Wait"

[(2025 Feb) S∗: Test Time Scaling for Code Generation](https://arxiv.org/pdf/2502.14382)

-> Test-time scaling for coding

[(2025 Feb) Teaching Language Models to Critique via Reinforcement Learning
](https://arxiv.org/abs/2502.03492)

-> Test-time scaling for coding

> [!note]
> _Joe: If we think about test time computing promoted by OpenAI o1, Deepmind [AlphaCode](https://deepmind.google/discover/blog/competitive-programming-with-alphacode/) in 2022 already used test-time scaling to do a lot of sampling and selection to boost the performance of competitive coding._

[(2025 Feb) Multi-Agent Verification: Scaling Test-Time Compute with Multiple Verifiers](https://arxiv.org/abs/2502.20379)

-> Test-time scaling with multiple agents (LLMs) for verification

[(2025 Mar) Chain-of-Retrieval Augmented Generation](https://arxiv.org/abs/2501.14342)

-> Test-time scaling for RAG

- Design ways that can scale up inference computation for RAG, such as decomping the question into modular questions and iteratively retrieve
- _Joe: this is a recurring theme of current rearch on test-time scaling for X. Design ways to increase inference computation, whether it be long CoT, search, verification, etc._

[(2025 Mar) Remasking Discrete Diffusion Models with Inference-Time Scaling](https://arxiv.org/abs/2503.00307)

-> Test-time scaling for discrete diffusion models for texts


#### Scaling Laws (all kinds of)

<details><summary>Scaling Laws</summary>

[(2024 Feb) Scaling Laws for Downstream Task Performance in Machine Translation](https://arxiv.org/abs/2402.04177)

-> Scaling behavior in a transfer learning setting

[(2025 Feb) Distillation Scaling Laws](https://arxiv.org/abs/2502.08606)

-> Scaling behavior for knowledge distillation

[(2025 Feb) Distributional Scaling Laws for Emergent Capabilities](https://arxiv.org/abs/2502.17356)

-> Emerging capabilities across multiple training runs with different random seeds

- Training experiments with Qwen2.5-0.5B and Qwen2.5-1.5B

</details>

### Process Reward (after o1)

[(2025 Feb) Process Reinforcement through Implicit Rewards](https://arxiv.org/abs/2502.01456)


### Multimodal, Image Generation

[(2025 Jan) Can We Generate Images with CoT? Let's Verify and Reinforce Image Generation Step by Step](https://arxiv.org/abs/2501.13926)

[(2025 Mar) ImageGen-CoT: Enhancing Text-to-Image In-context Learning with Chain-of-Thought Reasoning](https://arxiv.org/abs/2503.19312)

[(2025 Apr) SpatialReasoner: Towards Explicit and Generalizable 3D Spatial Reasoning](https://arxiv.org/abs/2504.20024)

- Spatial reasoning from vision inputs, augmented with parsed 3D structures


[(2025 May) Omni-R1: Do You Really Need Audio to Fine-Tune Your Audio LLM?](https://arxiv.org/abs/2505.09439)



### RL for Different Ways of Generation

[(2025 Feb) Self-rewarding correction for mathematical reasoning
](https://arxiv.org/pdf/2502.19613)

-> Self corrections trained with RL during generaion

[(2025 Mar) Reinforcement Learning for Long-Horizon Interactive LLM Agents](https://arxiv.org/pdf/2502.01600)

-> RL (LOOP, a data- and memory-efficient variant of proximal policy optimization) for long-horizon interactive **agents** ([AppWorld](https://appworld.dev/))

[(2025, Sep) RLP: Reinforcement as a Pretraining Objective](https://arxiv.org/abs/2510.01265)

-> Train with RL to let model think before generating every token
- This is related to the earlier work below, which presents the same idea but without explicitly using RL

[(2024, Mar) Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking](https://arxiv.org/abs/2403.09629)

- Train model to generate rational/thinking before each token generation

### Improve Long CoT for Reasoning

[(2025 Mar) START: Self-taught Reasoner with Tools](https://arxiv.org/abs/2503.04625)

-> Integrate tool usages with reasoning, with controled hint insertion and rejection sampling for training
- Tool usage (writing Python code) inside reasoning
- Enhance tool usage by injecting hint sequences in CoT during training, such as "Wait", "Maybe I can use Python" at various places based on heuristics
- Interleave Python code + executor with reasoning
- Rejection sampling fine-tuning (RFT)
- _Joe: this uses rejection sampling (you can call it RL, from the [Llama2 paper](https://arxiv.org/abs/2307.09288)). And the paper was not well polished (e.g. from small things like in-text citation formats, etc.)_

[(2025 Feb) LIMO: Less is More for Reasoning](https://arxiv.org/abs/2502.03387)

- 817 curated training samples
- Fine-tune Qwen2.5-32B-Instruct with SFT

[(2025 May) SEAL: Steerable Reasoning Calibration of Large Language Models for Free](https://arxiv.org/abs/2504.07986)

- Categorize the reasoning steps into three behaviors: Execution thoughts, Reflecting thoughts, and Transition thoughts
- Analyzed that wrong reasonings often result in much longer generations, with more usages of reflecting and transition
- Extract hidden states corresponding to different behaviral steps, and construct steering vectors to control the type of reasoning steps
- Achieve more effective and efficient reasoning with inference-time steering

### Understanding R1 and RL + LLMs, Tricks to Train RL

[(2025 Jan) Advancing Language Model Reasoning through Reinforcement Learning and Inference Scaling](https://arxiv.org/abs/2501.11651)

-> Tricks to scale up RL training to make it work

- Encourage sample diversity through oversampling
- Auxilliary loss on entropy
- Penalize undesired behaviors

[(2025 Feb) Demystifying Long Chain-of-Thought Reasoning in LLMs](https://arxiv.org/pdf/2502.03373)

-> Analyzing the learning dynamics of emergent reasoning with LLM + RL, across different factors such as SFT initilization, lengh reward design, etc.

[(2025 Mar) Cognitive Behaviors that Enable Self-Improving Reasoners, or, Four Habits of Highly Effective STaRs](https://arxiv.org/abs/2503.01307)

-> Analyzing the behaviors of emergent reasoning from LLM + RL, across base models and training data

- Why Qwen works better then Llama? Qwen already exhibits certain reasoning behaviors before training
- Priming Llama to begin RL training with data of complext reasoning behaviors helps, even when the final anwer is not correct
- _Joe: somehow I don't really get the name of cognitive behaviors (and the whole title); maybe I'm naive_

[(2025 Mar) Understanding R1-Zero-Like Training: A Critical Perspective](https://github.com/sail-sg/understand-r1-zero/blob/main/understand-r1-zero.pdf) 

-> Analyzing base models and RL

[(2025, Mar) The First Few Tokens Are All You Need: An Efficient and Effective Unsupervised Prefix Fine-Tuning Method for Reasoning Models](https://arxiv.org/abs/2503.02875)

-> Analyzing the role of prefixes of reasoning trajectories; could also work for self-improvements

- Found low diversity in the first few token generations (which makes sense as the sequence length is short, and the possibilities of different trajectories grow exponentially)
- Only sample a short prefix and fine-tune the model based on that. Not using labels.


[(2025 Jue) Thought Anchors: Which LLM Reasoning Steps Matter?](https://arxiv.org/abs/2506.19143)
-> Analysis of reasoning sentences

- Break down the reasoning chain into each single sentence, and check their causal relations and importances to other sentences and answer
- Summarized a sentence taxonomy for reasoning sentences (Table 1 in Appendix A)
- And visualize, with a good demo page https://www.thought-anchors.com/

[(2025 Apr) Demystifying Reasoning Dynamics with Mutual Information: Thinking Tokens are Information Peaks in LLM Reasoning](https://arxiv.org/pdf/2506.02867)

- Mutual information is computed between hidden states (continuous vectors) at token step t and ground truth answer
- Mutual information (MI) is not computed by estimating a distribution in the Shannon entropy format, but estimated by the Hilbert–Schmidt Independence Criterion (HSIC) with Gaussian kernels
- MI is computed by first sampling the hidden state vectors, and then compute based on HSIC between two matrices (collections of the continuous vectors)
- Token step t vectors are then mapped to tokens (e.g. by projection to the vocabulary) for concrete analysis

[(2025 Feb) Understanding the Uncertainty of LLM Explanations: A Perspective Based on Reasoning Topology](https://arxiv.org/abs/2502.17026)

-> Not necessarily long CoT, but built a topological graph to explain reasoning patterns.

- The structured representation of reasoning could be applied elsewhere, e.g. to super long reasoning process.

[(2025 Sept) Reasoning Vectors: Transferring Chain-of-Thought Capabilities via Task Arithmetic](https://arxiv.org/abs/2509.01363)
-> Steering/task vectors for reasoning

- Two identity models, one going through SFT, and one GRPO
- Extract task vectors as the difference between parameters to control the reasoning behaviors


[(2025 Oct) First Try Matters: Revisiting the Role of Reflection in Reasoning Models](https://arxiv.org/abs/2510.08308)
-> challenges the conception that reflection in model reasoning actually does "reflection"

- Focused on reflective behaviours of model reasoning
- Found that most reflective behaviors do not actually alter model reasonings, but merely confirm
- Fine-tuning on more reflective behaivors mostly enhance first-answer correctness

[(2025 Sept) RL's Razor: Why Online Reinforcement Learning Forgets Less](https://arxiv.org/abs/2509.04259)

- RL training incurs less forgetting than SFT

[(2025 Apr) Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?](https://arxiv.org/abs/2504.13837)

- Challenges the role of RL to incentive model with new capabilities vs. just capitalizing on existing capabilities

[(2025, Apr) Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?](https://arxiv.org/abs/2504.13837)

- Many works with similar questions and conclusions

[(2025, Apr) Echo Chamber: RL Post-training Amplifies Behaviors Learned in Pretraining](https://arxiv.org/abs/2504.07912)

- Pretraining of smaller LMs + RL to understand RL effects

[(2025, Jun) Spurious Rewards: Rethinking Training Signals in RLVR](https://arxiv.org/abs/2506.10947?)

- Tested spurious rewards such as ground truth, majority voting, format, random, etc. that can still help the model reasoning. Interesting study to understand how RL helps model learn.

[(2025 Nov) Reinforcement Learning Improves Traversal of Hierarchical Knowledge in LLMs](https://arxiv.org/abs/2511.05933)

- Similar research topic as above


#### Data

Thought Anchors: https://www.thought-anchors.com/

Open Thoughts: https://github.com/open-thoughts/open-thoughts

#### Training Receipe

[(2025 Nov) JustRL: Scaling a 1.5B LLM with a Simple RL Recipe](https://relieved-cafe-fe1.notion.site/JustRL-Scaling-a-1-5B-LLM-with-a-Simple-RL-Recipe-24f6198b0b6b80e48e74f519bfdaf0a8)

- Simple RL training receipe for scaling 1.5B LLM training


#### RL Alrogithms

[(2025, Jun) TreeRPO: Tree Relative Policy Optimization](https://arxiv.org/abs/2506.05183)

- Sampling to generate a tree structured trajectory, and collect rewards for every node
- Improves sampling efficiency for training effciency

[(2025 June) Consistent Paths Lead to Truth: Self-Rewarding Reinforcement Learning for LLM Reasoning](https://arxiv.org/pdf/2506.08745)

- Measures how much intermediate reasoning steps lead to the same final answer, as a "consistency" metric summarizing the reasoning trajectory
- Also measures how much sudden changes of the final answer at later reasoning steps are there in the trajectory, as a "volatility" metric
- Observes clear separations of these two metrics between trajectories leading to correct vs. incorrect final answers
- Include these trajectory statistics for reward, plus a "curiosity" reward that encourages diversity; also borrows the grouping idea from GRPO -> no external reward is needed, as during training the reward just depends on the sample trajectories and their final answers

[(2025, July) STeCa: Step-level Trajectory Calibration for LLM Agent Learning](https://aclanthology.org/2025.findings-acl.604/)

### Efficiency

[(2024 Dec) Compressed Chain of Thought: Efficient Reasoning through Dense Representations](https://arxiv.org/abs/2412.13171)

- Reasoning with continuous tokens

[(2025, Jan) Think Smarter not Harder: Adaptive Reasoning with Inference Aware Optimization](https://arxiv.org/abs/2501.17974)

- Budget aware thinking

[(2025 Feb) TokenSkip: Controllable Chain-of-Thought Compression in LLMs](https://arxiv.org/abs/2502.12067)

- Filtering out some "unimportant" CoT tokens based on heuristics, generate compressed CoT tokens, and then fine-tune on the reduced trajectories
- _Joe: similar flavor to context compression, token delection, like LLMLingua_

[(2025 Mar) Chain of Draft: Thinking Faster by Writing Less](https://arxiv.org/abs/2502.18600)

-> _Joe: this is not using RL, but just a simple way of prompting by limiting the reasoning step lengths with instructions in prompts. I think similarly we can train LLM with RL to enforce this, and/or as a reward, to improve efficiency during the reasoning process_

-> _Joe: (a few days later) found out the following paper does that exactly lol_

[(2025 Mar) L1: Controlling How Long A Reasoning Model Thinks With Reinforcement Learning
](https://arxiv.org/abs/2503.04697)

-> _Joe: LLM + RL to encourage shorter reasoning steps. The way is to condition on special symbols in the prompt controling reasoning steps, which poses another reward_

- Training starts from the base model [DeepScaleR-1.5B-Preview](#deepscaler-berkeley) (using the same hyperparameters for GRPO)
- Training data also from DeepScaleR-Preview-Dataset, 40K question-answer pairs drawn from AIME, AMC, Omni-Math and STILL
- Training context length restricted to 4K, and testing restricted to 8K
- Fine-tuned for 700 steps and further 120 steps for two different length reward formulations
- Again using [VeRL](#tools) framework

[(2025 Feb) Demystifying Long Chain-of-Thought Reasoning in LLMs](https://arxiv.org/pdf/2502.03373)

-> _Joe: see Section 4.2 for the length control with reward design. Strategy is similar to the paper above._

[(2025, Feb) [EMNLP 2025] LightThinker: Thinking Step-by-Step Compression](https://arxiv.org/abs/2502.15589)

-> _Joe: Compressing thinking steps into smaller set of special tokens. Train with special attention mask, inference with reduced KV cache based on the mask structures._

- Not using RL. Merging rules are based on heuristics.

[(2025 Mar) The First Few Tokens Are All You Need: An Efficient and Effective Unsupervised Prefix Fine-Tuning Method for Reasoning Models](https://arxiv.org/pdf/2503.02875)


[(2025 Apr) Z1: Efficient Test-time Scaling with Code](https://arxiv.org/abs/2504.00810)

-> Reducing reasoning token length through SFT on QwQ-32B-preview model generated data
- Dataset size of 107K, SFT model Qwen-2.5-Coder-7B-Instruct with bfloat16, FSDP, global batch size to 128 for 2 epochs using 8 NVIDIA
A100-80G GPUs
- Simple reasoning dataset analysis of trigram frequency in Section 2.1 and Appendix A.2
- The biggest difference is removing  `<think>...</think>` delimiters?
- _Joe: Not quite sure about the "Shifted Thinking Window" name_

[(2024 Apr) Reasoning Models Know When They're Right: Probing Hidden States for Self-Verification](https://arxiv.org/abs/2504.05419v1)

-> Probe whether the intermediate reasoning step hidden states can predict the correctness of the final answer
- Can use the probe for early exit for long reasoning

[(2025 Apr) ThinkPrune: Pruning Long Chain-of-Thought of LLMs via Reinforcement Learning](https://arxiv.org/abs/2504.01296)

-> Added reasoning length limit as a reward for RL

[(2025 Apr) Stop Overthinking: A Survey on Efficient Reasoning for Large Language Models](https://arxiv.org/abs/2503.16419)

-> Survey
- Collection of papers: https://github.com/Eclipsess/Awesome-Efficient-Reasoning-LLMs

[(2025 Apr) Learning Adaptive Parallel Reasoning with Language Models](https://arxiv.org/abs/2504.15466)

-> Changing the generation process to combine parallel and sequential search during generation
- Similar to one of my earlier ideas of optimizing generation process that can be trained with RL directly for efficiency
- But focused on Countdown task only, and trained model from scratch for small scale experiments

[(2025 May) Learn to Reason Efficiently with Adaptive Length-based Reward Shaping](https://arxiv.org/abs/2505.15612)

-> Reducing reasoning trajectory with different length reward shapes

[(2025 May) SEAL: Steerable Reasoning Calibration of Large Language Models for Free](https://arxiv.org/abs/2504.07986)

- Categorize the reasoning steps into three behaviors: Execution thoughts, Reflecting thoughts, and Transition thoughts
- Analyzed that wrong reasonings often result in much longer generations, with more usages of reflecting and transition
- Extract hidden states corresponding to different behaviral steps, and construct steering vectors to control the type of reasoning steps
- Achieve more effective and efficient reasoning with inference-time steering, by rougly controling the number of steps for reflection, etc.

[(2025 May) AutoL2S: Auto Long-Short Reasoning for Efficient Large Language Models](https://arxiv.org/abs/2505.22662)


[(2025 June) Just Enough Thinking: Efficient Reasoning with Adaptive Length Penalties Reinforcement Learning](https://arxiv.org/abs/2506.05256)

-> Again adding a length related penalty in the reward for RL training, but adjusted to the difficulty of each questions, measured by the pass rate of K samples
- The length reward formulation is a bit less straightforward
- Doesn't show superior performance compared to previous baselines with simple length reward, such as L1-Max

[(2025 June) Token-Efficient RL for LLM Reasoning](https://arxiv.org/pdf/2504.20834v4)

-> Reduce resource usages when training with GRPO with LoRA
- Restrict the tokens  that contribute to the loss
- Estimate token level advantage, and uses replay for resampling

[(2025 July) RLVMR: Reinforcement Learning with Verifiable Meta-Reasoning Rewards for Robust Long-Horizon Agents](https://arxiv.org/abs/2507.22844)

- RL for long-horizon reasoning with agents

[(2025 Aug) Efficient Inference for Large Reasoning Models: A Survey](https://arxiv.org/abs/2503.23077)

-> Survey

[(2025, Aug) Deep Think with Confidence](https://arxiv.org/abs/2508.15260)
