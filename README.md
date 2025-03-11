# Post-DeepSeek-R1
Resources and research after DeepSeek-R1, around test-time computing, resurgence of RL, and new LLM learning/application paradigms.

---

## DeepSeek-R1 Reproduction ("popular" and fast ones)

> This behavior is not only a testament to the model’s growing reasoning abilities but also a captivating example of how reinforcement learning can lead to unexpected and sophisticated outcomes.

-- From [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)

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


## R1-like RL Reproduction for More Scenarios

### Tools

- RL libraries:
  - [veRL](https://github.com/volcengine/verl) (seems most popular as of Mar 2025). Check this [list](https://github.com/volcengine/verl?tab=readme-ov-file#awesome-work-using-verl) of R1 followup works
  - [TRL](https://huggingface.co/docs/trl/en/index)
  - Inference: [vLLM](https://github.com/vllm-project/vllm) seems a must to speed up inference
- Starting models: [Qwen2.5](https://github.com/QwenLM/Qwen2.5) (base, instruct, R1-distilled, math) seems most popular (as of Mar 2025), both 3B and 7B models are made work; 0.5B is a bit weaker but could also learn
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

### Test-time Scaling

[(2025 Feb) S∗: Test Time Scaling for Code Generation](https://arxiv.org/pdf/2502.14382)

-> Test-time scaling for coding

[(2025 Feb) Teaching Language Models to Critique via Reinforcement Learning
](https://arxiv.org/abs/2502.03492)

-> Test-time scaling for coding


### RL for Different Ways of Generation

[(2025 Feb) Self-rewarding correction for mathematical reasoning
](https://arxiv.org/pdf/2502.19613)

-> Self corrections trained with RL during generaion


### Understanding R1 and RL + LLMs

[(2025 Mar) Cognitive Behaviors that Enable Self-Improving Reasoners, or, Four Habits of Highly Effective STaRs
](https://arxiv.org/abs/2503.01307)

-> Analyzing the behaviors of emergent reasoning from LLM + RL, across base models and training data

- Why Qwen works better then Llama? Qwen already exhibits certain reasoning behaviors before training
- Priming Llama to begin RL training with data of complext reasoning behaviors helps, even when the final anwer is not correct
- _Joe: somehow I don't really like the name of cognitive behaviors_


### Efficiency

[(2025 Mar) Chain of Draft: Thinking Faster by Writing Less](https://arxiv.org/abs/2502.18600)

-> _Joe: this is not using RL, but just a simple way of prompting by limiting the reasoning step lengths with instructions in prompts. I think similarly we can train LLM with RL to enforce this, and/or as a reward, to improve efficiency during the reasoning process_
