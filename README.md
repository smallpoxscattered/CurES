<div align="center">

# CurES - From Gradient Analysis to Efficient Curriculum Learning for Reasoning LLMs
[![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/html/2510.01037v1) [![Github](https://img.shields.io/badge/GVM-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/ZexuSun/CurES)
</div>

## Catalog
- [Introduction](#Introduction)
- [Environment Setup](#environment-setup)
- [Experiments Running](#experiments-running)

## Introduction

Curriculum learning plays a crucial role in enhancing the training efficiency of large language models (LLMs) on reasoning tasks. However, existing methods often fail to adequately account for variations in prompt difficulty or rely on simplistic filtering mechanisms to select prompt datasets within a narrow criterion range, resulting in significant computational waste. In this work, we approach the problem from the perspective of reinforcement learning gradient optimization, offering a systematic and theoretical investigation into how to improve the training efficiency of LLMs. We identify two key factors influencing training efficiency: the selection of training prompts and the allocation of rollout quantities across different prompts. Our theoretical analysis reveals that the sampling distribution of prompts dictates the convergence rate of gradient descent, while the allocation of the rollout quantity influences the consistency and stability of overall gradient updates. Based on these insights, we propose CurES, an efficient training method that accelerates convergence and employs Bayesian posterior estimation to minimize computational overhead. Experiments demonstrate that our CurES outperforms Group Relative Policy Optimization (GRPO) by **+3.30** points and **+4.82** points with 1.5B and 7B models, respectively, and exceeds the best prior sample efficient methods by **+2.12** points on average across eight math reasoning benchmarks. Our CurES also improves convergence speed compare to baselines such as GRPO.

<p align="center">
  <img src="figures/Curriculum.png" width="85%" />
  <figcaption align="left"><b>Figure 1：</b>Illustration of our theoretical and practical contributions. The first part presents our theoretical analysis, which establishes the relationship between the gradient efficiency and models’ question-answering accuracy, denoted as $p_{\theta}(x)$. Building upon these insights, we develop CurES, a practical method that initially estimates $p_{\theta}(x)$ using a small rollout quantity, then reallocates prompt sampling probabilities and rollout quantities based on the estimated accuracy. In contrast to the unified framwork provided by our CurES, existing sample efficient methods fail to optimize from both prompt sampling and rollout quantity aspects. Speed-RL improves the prompt sampling procedure by eliminate the prompts with estimated accuracy of 0 or 1, and GVM propose to assign more rollout quantities to harder prompts.</figcaption>
  <!-- <img src="figures/alg.png" width="85%"> -->
</p>

## Environment Setup
1. Create a new environment.
   ```bash
   conda create -n cures python==3.10
   conda activate cures
   ```

2. Install dependencies
   ```bash
   pip install pip --upgrade
   pip install uv
   git clone https://github.com/ZexuSun/CurES.git
   cd CurES/
   python -m uv pip install -r requirements.txt
   ```

## Experiment Runnning

1. Start the training loop.
   ```bash
   # Initialize Ray
   ray start --head --dashboard-host=0.0.0.0
   ray stop --force
   # Login wandb
   wandb login
   # Use GRPO as advantage estimator.
   # Modify run_cures_grpo.sh (e.g., wandb api key, model root, ckpts root, etc.) before running.
   bash runs/scripts/run_cures_grpo.sh
   # Use Reinforce++ as advantage estimator
   # Modify run_cures_rpp.sh (e.g., wandb api key, model root, ckpts root, etc.) before running.
   bash runs/scripts/run_cures_rpp.sh
   ```

## Acknowledgement
We greatly thanks [verl](https://github.com/volcengine/verl) for providing the awesome codebase!