#!/bin/bash
set -x
# Runs the "codeparrot-small" parameter model
export PYTHONPATH=/mnt/petrelfs/xiongyingtong/XYT-Megatron-Deepspeed/XYT-DeepSpeed:$PYTHONPATH

# export NCCL_DEBUG=info
export NCCL_SOCKET_IFNAME="bond0"
export NCCL_IB_HCA="mlx5_2,mlx5_3,mlx5_4,mlx5_5"

export CUDA_DEVICE_MAX_CONNECTIONS=1
my_shard='zero15'

#30B model config
num_layers=32
hidden_size=4096
seq_length=${1:-4096}
num_attention_heads=32
export seq_length=${seq_length}


## batch config
# global/micro/worldsize/tp = accumulate gradient
# zero 1.5 会使得每一张卡的显存保持一致。然而zero 1 随着rank 数的上升，每张卡分到的 optimizer state 会更少，导致显存占用更少。
global_batch_size=8
micro_batch_size=1

## ZeRO stage
zero_stage=2
no_pp=true

## para config
tp=1
pp=1

## log interval
log_interval=1

# __doc_head_address_start__
# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}

head_node_ip=$(cat /etc/hosts | grep -w "$head_node" | awk '{print $1}')
echo $head_node

## env config
GPUS_PER_NODE=8
# HOST-10-140-60-[33-70,82-84,86,88-91,94-97,102-106,108-109,113,124-129]
MASTER_ADDR=$head_node_ip
MASTER_PORT=7880
NNODES=$SLURM_NNODES

## file path
CHECKPOINT_PATH=./tmp
VOCAB_FILE=/mnt/petrelfs/xiongyingtong/XYT-Megatron-Deepspeed/dataset/gpt2-vocab.json
MERGE_FILE=/mnt/petrelfs/xiongyingtong/XYT-Megatron-Deepspeed/dataset/gpt2-merges.txt
# DATA_PATH=/mnt/petrelfs/wangguoteng.p/chenqiaoling/chenqiaoling/LLaMA-Megatron-DeepSpeed/data/my-gpt2_text_document



GPT_ARGS=" \
    --num-layers ${num_layers}
    --hidden-size ${hidden_size}
    --num-attention-heads ${num_attention_heads}
    --ds-sequence-parallel-size 8 \
    --seq-length ${seq_length}
    --use-rotary-position-embeddings \
    --use-flash-attn-v2 \
    --ffn-hidden-size 16384 \
    --disable-bias-linear \
    --untie-embeddings-and-output-weights \
    --max-position-embeddings ${seq_length}
    --micro-batch-size ${micro_batch_size}
    --global-batch-size ${global_batch_size}
    --lr 0.0001
    --train-iters 20
    --lr-decay-iters 150000
    --lr-decay-style cosine
    --lr-warmup-iters 2000
    --weight-decay .1
    --adam-beta2 .999
    --bf16
    --log-interval 1
    --save-interval 2000
    --eval-interval 200
    --eval-iters 10
"


DATA_ARGS="
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --data-impl mmap \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval ${log_interval} \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10
"

template_json="ds_config_gpt_TEMPLATE.json"
config_json="ds_config_bert_bsz${global_batch_size}_mbsz${batch_size}_log${log_interval}_zero${zero_stage}.json"
if [[ $zero_stage -gt 0 ]]; then
sed "s/CONFIG_BATCH_SIZE/${global_batch_size}/" ${template_json} \
    | sed "s/CONFIG_MBSIZE/${micro_batch_size}/" \
    | sed "s/LOG_INTERVAL/${log_interval}/" \
    | sed "s/ZERO_STAGE/${zero_stage}/" \
    | sed "s/PRESCALE_GRAD/false/" \
    | sed "s/CONFIG_FP16_ENABLED/false/" \
    | sed "s/CONFIG_BF16_ENABLED/true/" \
      > ${config_json}
else
sed "s/CONFIG_BATCH_SIZE/${global_batch_size}/" ${template_json} \
    | sed "s/CONFIG_MBSIZE/${micro_batch_size}/" \
    | sed "s/LOG_INTERVAL/${log_interval}/" \
    | sed "s/ZERO_STAGE/${zero_stage}/" \
    | sed "s/PRESCALE_GRAD/true/" \
    | sed "s/CONFIG_FP16_ENABLED/false/" \
    | sed "s/CONFIG_BF16_ENABLED/true/" \
      > ${config_json}
fi

# enable accumulate gradient
# deepspeed_options="
#     --deepspeed \
#     --deepspeed_config ${config_json}
#     --zero-stage $zero_stage \
#     --pipeline-model-parallel-size 1 \
#     --deepspeed-activation-checkpointing \
# "

# disable accumulate gradient
deepspeed_options="
    --deepspeed \
    --deepspeed_config ${config_json}
    --zero-stage $zero_stage \
    --pipeline-model-parallel-size 1 \
"

if [[ "${no_pp}" = "true" ]]; then
deepspeed_options="${deepspeed_options} \
    --no-pipeline-parallel"
fi

log_file=flash_attn_8k_debugggg.log


# log_file=test_s{$1}.log
# export self_zero=$1

# deepspeed $DISTRIBUTED_ARGS\
#         ../pretrain_llama.py \
#         --tensor-model-parallel-size $tp \
#         --pipeline-model-parallel-size $pp \
#         $GPT_ARGS \
#         $DATA_ARGS \
#         $OUTPUT_ARGS \
#         $deepspeed_options 2>&1 | tee $log_file
#         --save $CHECKPOINT_PATH \
#         #--load $CHECKPOINT_PATH 
dir=`pwd`
torchrun --nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $SLURM_PROCID --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
    ${dir}/../../pretrain_gpt.py \
    --tensor-model-parallel-size $tp \
    --pipeline-model-parallel-size $pp \
    $GPT_ARGS $DATA_ARGS $OUTPUT_ARGS $deepspeed_options 2>&1|tee $log_file