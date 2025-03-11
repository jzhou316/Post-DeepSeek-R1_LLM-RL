# Post-DeepSeek-R1
Resources and research after DeepSeek-R1, around test-time computing, resurgence of RL, and new LLM learning/application paradigms.

---

# DeepSeek-R1 Reproduction

> This behavior is not only a testament to the model’s growing reasoning abilities but also a captivating example of how reinforcement learning can lead to unexpected and sophisticated outcomes.

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
- Good insight and training receipe: error cases are initially longer CoTs, so gradually extending context length for thinking during training (_Joe: a sort of curriculum learning for RL_)

![](https://github.com/agentica-project/deepscaler/blob/main/figures/deepscaler.png)

*Figure 1: DeepScaleR 1.5B model's Pass@1 accuracy on AIME2024 as RL training progresses. At step 1040 and 1520, the context length is extended to 16K and 24K. For more details, see our [blog post](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2).*


#### [Open R1](https://github.com/huggingface/open-r1?tab=readme-ov-file#open-r1) (Hugging Face)

- Fully open reproduction of DeepSeek-R1
- [Blog post](https://huggingface.co/blog/open-r1)


#### [Mini-R1](https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/mini-deepseek-r1-aha-grpo.ipynb)

- A minimal single notebook that tries to reproduce the DeepSeek-R1 "reasoning" results on a single task (the Countdown Game)
- Uses GRPO and Q-Lora, also with the TRL library
- Starting with the [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) model (suggested using models > 1.5B) (_Joe: Yes, we need the model to start with to have certain capabilities_)
- Good learning material with code


