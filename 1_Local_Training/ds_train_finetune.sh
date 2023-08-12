
LR=1e-4

MASTER_PORT=$(shuf -n 1 -i 10000-65535)

deepspeed --num_gpus=1 --master_port $MASTER_PORT main.py \
    --deepspeed deepspeed.json \
    --do_train \
    --train_file /home/project-1/data/AdvertiseGen/train.json \
    --test_file /home/project-1/data/AdvertiseGen/dev.json \
    --prompt_column content \
    --response_column summary \
    --overwrite_cache \
    --model_name_or_path /home/project-1/checkpoints/chatglm-6b \
    --output_dir /home/project-1/baoziyuan/ChatGLM-RPC/1_Local_Training/output/adgen-chatglm-6b-ft-$LR \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 64 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --predict_with_generate \
    --max_steps 5000 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate $LR \
    --fp16 \
    --num_train_epochs 2 \