#!/bin/bash

set -x
set -euo pipefail
trap 'echo "Error at line $LINENO"; exit 1' ERR

export WANDB_API_KEY="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" # todo+
export HF_ENDPOINT=https://hf-mirror.com

project_name=CurES
algorithm=grpo
model=Qwen2.5-1.5B-Instruct # todo
policy_loss=plusplus
num_rollout_min=4 
num_rollout_max=8 
correct_threshold=0.5 
entropy_param=1.0
enable_filter_groups=False 
filter_groups_metric=seq_final_reward
max_num_gen_batches=10
train_batch_size=64 # todo: 对齐 per_device_train_batch_size: 32
data="openai/gsm8k"
PREFIX=${project_name}-${model}-${algorithm}-${policy_loss}-${data}-DAPO_${enable_filter_groups}-${num_rollout_min}-${num_rollout_max}-${entropy_param}

MODEL_ROOT="/root/autodl-tmp/CurES/model"  # todo
DATA_ROOT="/root/autodl-tmp/CurES/data/gsm8k-instruct" # todo
aime24=$DATA_ROOT/AIME2024-dup16-instruct/train.parquet
aime25=$DATA_ROOT/AIME2025-dup16-instruct/train.parquet
math500=$DATA_ROOT/Math500-instruct/test.parquet
amc23=$DATA_ROOT/AMC23-instruct/train.parquet

test_files="['$DATA_ROOT/test.parquet']" # todo:修改成gsm8k数据集

mkdir -p logs/${project_name}

for i in {1..1}; do # todo: 训练一次
    CKPT_ROOT="/root/autodl-tmp/CurES/checkpoint/${project_name}" # todo
    INTER_DIR=./CurES/Intermediate/${PREFIX}/iter_${i}
    if [ $i -eq 1 ]; then
        model_name_or_path=${MODEL_ROOT}/${model}
    else
        model_name_or_path=${CKPT_ROOT}/${PREFIX}/iter_$((i-1))/global_step_10/actor/huggingface
    fi

    experiment_name=${PREFIX}_iter_${i}
    bash ./CurES/run_initialize.sh $model_name_or_path $i $num_rollout_min $num_rollout_max $correct_threshold $entropy_param $train_batch_size $INTER_DIR $DATA_ROOT
    wait

    train_files="['$DATA_ROOT/train.parquet']" # todo: 修改训练集

    difficulty_data=$INTER_DIR/difficulty_rank_all.json
    counter_data=$INTER_DIR/counter_rank_all.json
    sample_sizes_data=$INTER_DIR/sample_size_rank_all.json

    GPUS=(0) # todo：显卡
    my_world_size=${#GPUS[@]}
    total_epochs=7 # todo: 因为 E2H 中 800 step ，而 7 x (7473 / 64) = 817.35 其中 7 为epoch，7473为数据集总条数，64为batchsize
    # todo 以下是对齐参数
    # max_prompt_length: 1600
    # max_completion_length: 512 -> max_response_length
    # gradient_checkpointing: false
    # --gpu_memory_utilization 0.95 
    # cuda
    CUDA_VISIBLE_DEVICES=0 python3 -m CurES.main_cures \
        algorithm.adv_estimator=$algorithm \
        data.train_files="$train_files" \
        data.val_files="$test_files" \
        data.train_batch_size=$train_batch_size \
        data.max_prompt_length=1600 \
        data.max_response_length=512 \
        data.filter_overlong_prompts=True \
        +data.use_cures=True \
        +data.cures_difficulty_data="$difficulty_data" \
        +data.cures_counter_data="$counter_data" \
        +data.cures_entropy_param=1.0 \
        +data.cures_replacement=True \
        +data.cures_acceptance_threshold=0.5 \
        data.shuffle=False \
        algorithm.filter_groups.enable=${enable_filter_groups} \
        algorithm.filter_groups.metric=${filter_groups_metric} \
        algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
        actor_rollout_ref.model.path=$model_name_or_path \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.enable_gradient_checkpointing=False \
        actor_rollout_ref.actor.ppo_mini_batch_size=64 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.policy_loss=$policy_loss \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=64 \
        actor_rollout_ref.rollout.name=vllm \
        +actor_rollout_ref.rollout.use_em=True \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.95 \
        actor_rollout_ref.rollout.n=4 \
        +actor_rollout_ref.rollout.sample_sizes_data="$sample_sizes_data" \
        actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=64 \
        algorithm.kl_ctrl.kl_coef=0.001 \
        trainer.critic_warmup=0 \
        trainer.logger=['console','wandb'] \
        trainer.project_name=${project_name} \
        trainer.experiment_name=${experiment_name} \
        trainer.n_gpus_per_node=$my_world_size \
        trainer.nnodes=1 \
        trainer.val_before_train=True \
        trainer.save_freq=5 \
        trainer.default_local_dir=${CKPT_ROOT}/${PREFIX}/iter_${i} \
        trainer.test_freq=5 \
        trainer.total_epochs=$total_epochs 2>&1 | tee logs/${project_name}/${experiment_name}_iter${i}.log

    python scripts/legacy_model_merger.py merge \
        --backend=fsdp \
        --hf_model_path=${MODEL_ROOT}/$model \
        --local_dir=${CKPT_ROOT}/${PREFIX}/iter_${i}/global_step_10/actor \
        --target_dir=${CKPT_ROOT}/${PREFIX}/iter_${i}/global_step_10/actor/huggingface
done