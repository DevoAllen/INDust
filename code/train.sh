export CUDA_VISIBLE_DEVICES=0,1


# Training in LoRA manner
torchrun --nproc_per_node=2 --master_port=9999   train_lora.py \
    --model_name_or_path Path_to/BELLE-LLaMA-7B-2M \
    --data_path data.json \
    --fp16 True \
    --output_dir ./output/belle-7b-lora-critique \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --model_max_length 768 \
    --use_lora \
    --lora_config ./config/lora_config_llama.json \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 30 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --seed 42 \
    --logging_steps 5 \
    --tf32 False



# We also provide code for directly finetuning.
torchrun --nproc_per_node=2 --master_port=9999   train.py \
    --model_name_or_path Path_to/BELLE-LLaMA-7B-2M \
    --data_path data.json \
    --fp16 True \
    --output_dir ./output/belle-7b-lora-critique \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --model_max_length 768 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 30 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --seed 42 \
    --logging_steps 5 \
    --tf32 False