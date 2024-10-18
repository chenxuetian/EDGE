#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


GPUS_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001

MODEL=monkey_model
OUTPUT=test_output
CONFIG=test.json
NEPOCHS=1


OUTPUT="output_models/${OUTPUT}"
CONFIG="training_configs/${CONFIG}"
LOG="${OUTPUT}/training.log"
mkdir -p "$OUTPUT"

echo "############################################
Training start time: 
    $(date +"%Y-%m-%d %H:%M:%S")
Training with:
    Model: ${MODEL}
    Config: ${CONFIG}
    Output Path: ${OUTPUT}
    Num Epochs: ${NEPOCHS}
    Use GPUs: ${GPUS_PER_NODE}
############################################
" > $LOG


############################################
##### Start Distributed Training HERE
############################################
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT"

torchrun $DISTRIBUTED_ARGS finetune_web.py \
    --model_name_or_path $MODEL \
    --output_dir $OUTPUT \
    --data_path $CONFIG \
    --bf16 True \
    --fix_vit True \
    --num_train_epochs $NEPOCHS \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --optim "lion_8bit" \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --adam_beta1 0.95 \
    --adam_beta2 0.98 \
    --warmup_ratio 0.02 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "none" \
    --model_max_length 2048 \
    >> $LOG 2>&1
