# GSM8K E2H PreBucketGRPO Params

This file records the parameter set used for the Qwen-1.5B GSM8K run in this repo, for cross-framework alignment.

## Run Identity

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Task: `GSM8K`
- Algorithm: `PreBucketGRPO` (E2H prebucket static curriculum)
- Mode flow: `rollout -> train -> test`
- Run timestamp: `20260402_115020` (Asia/Shanghai)

## Runtime Launch Settings

### vLLM server launch (GPU 0)

```bash
CUDA_VISIBLE_DEVICES=0 trl vllm-serve \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --dtype bfloat16 \
  --max_model_len 4096 \
  --gpu_memory_utilization 0.95 \
  --trust_remote_code true \
  --log_level warning
```

### Train launch (GPUs 1,2)

```bash
CUDA_VISIBLE_DEVICES=1,2 accelerate launch \
  --mixed_precision bf16 \
  --num_machines 1 \
  --num_processes 2 \
  --dynamo_backend no \
  main.py mode=train model=qwen1.5b task=gsm8k algorithm=prebucket_grpo \
  algorithm.args.output_dir=outputs/Qwen-1.5B_gsm8k_prebucket_grpo/20260402_115020 \
  algorithm.args.model_output_dir=checkpoints/Qwen-1.5B_gsm8k_prebucket_grpo/20260402_115020
```

### Test launch (GPUs 1,2)

```bash
CUDA_VISIBLE_DEVICES=1,2 accelerate launch \
  --mixed_precision bf16 \
  --num_machines 1 \
  --num_processes 1 \
  --dynamo_backend no \
  main.py mode=test model=qwen1.5b task=gsm8k algorithm=prebucket_grpo \
  algorithm.args.output_dir=outputs/Qwen-1.5B_gsm8k_prebucket_grpo/20260402_115020 \
  algorithm.args.model_output_dir=checkpoints/Qwen-1.5B_gsm8k_prebucket_grpo/20260402_115020
```

## Environment Settings

```bash
HF_ENDPOINT=https://hf-mirror.com
HF_HUB_ENABLE_HF_TRANSFER=0
HF_HUB_OFFLINE=1
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
FLASH_ATTENTION_2_DISABLED=0
OMP_NUM_THREADS=8
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## Effective Core Config

### algorithm.name

```text
PreBucketGRPO
```

### algorithm.args

```yaml
accelerator_config:
  split_batches: false
beta: 0.001
bf16: true
ddp_find_unused_parameters: false
eval_steps: 800
eval_strategy: steps
gradient_accumulation_steps: 4
gradient_checkpointing: false
learning_rate: 1e-06
log_completions: false
log_on_each_node: false
logging_dir: outputs/Qwen-1.5B_gsm8k_prebucket_grpo/20260402_115020
logging_steps: 10
logging_strategy: steps
loss_type: grpo
lr_scheduler_type: cosine
max_completion_length: 512
max_prompt_length: 1600
max_steps: 1600
model_output_dir: checkpoints/Qwen-1.5B_gsm8k_prebucket_grpo/20260402_115020
num_generations: 4
output_dir: outputs/Qwen-1.5B_gsm8k_prebucket_grpo/20260402_115020
overwrite_output_dir: true
per_device_train_batch_size: 32
report_to:
  - tensorboard
save_strategy: "no"
seed: 42
shuffle_dataset: false
steps_per_generation: 1
tf32: true
use_vllm: true
vllm_mode: server
wandb_log_unique_prompts: false
```

### algorithm.e2h_args (rollout/curriculum)

```yaml
bucket_boundaries: [0.33, 0.67]
bucket_quantiles: [0.3333333333, 0.6666666667]
bucket_strategy: quantile
curriculum_schedule: prebucket_static
initial_rollouts: 20
rollout_batch_size: 352
rollout_generation_batch_size: 1280
rollout_max_completion_length: 192
rollout_temperature: 0.8
rollout_top_p: 0.95
scheduler_args:
  beta: 1.0
  min_prob: true
  mu_exp: 0.5
  sigma: 0.5
stage_boundaries: [0.3333333333, 0.6666666667]
stage_order: [0, 1, 2]
```

### model.args

```yaml
attn_implementation: flash_attention_2
lora_alpha: 64
lora_dropout: 0.1
lora_r: 32
lora_target_modules: [q_proj, v_proj]
lora_task_type: CAUSAL_LM
model_name_or_path: Qwen/Qwen2.5-1.5B-Instruct
torch_dtype: bfloat16
trust_remote_code: true
use_peft: true
```

### task.args

```yaml
assistant_content: |
  Let me solve this step by step.
  <think>
correctness_reward: 0.9
correctness_reward_fn: Gsm8kCorrectnessReward
format_reward: 0.1
format_reward_fn: FormatReward
max_completion_length: 512
max_prompt_length: 1600
max_steps: 1600
path: /root/autodl-tmp/E2H-Reasoning/data/data/gsm8k
system_content: "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.\n"
user_content: "Solve the following math problem\n{question}\n\n Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> 500 </answer>."
```

## Effective Attention Backend

Although `model.args.attn_implementation` is set to `flash_attention_2`, runtime resolved to:

```text
Using attention implementation: sdpa
```

This should be treated as the effective backend for this run.
