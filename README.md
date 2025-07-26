1. git clone  https://github.com/estherliu02/IM-SFT.git
2. cd IM-SFT
3. 
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

chmod +x Miniconda3-latest-Linux-x86_64.sh

./Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc

conda create -n sft python=3.10
conda activate sft
pip install torch==2.1.2+cu121 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements2.txt





pip uninstall flash-attn -y
pip uninstall torch torchvision torchaudio -y

# Reinstall PyTorch 2.1.2 with CUDA 12.1
pip install torch==2.1.2+cu121 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install flash-attn==2.2.2 --no-build-isolation


pip install torch
pip install flash-attn --no-build-isolation

pip uninstall flash-attn -y

git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention

git checkout v2.2.2

pip install ninja packaging wheel

export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"

python setup.py install

huggingface-cli login --token MY_HF_TOKEN
wandb login MY_WANDB_TOKEN

rm -f /workspace/.hf_home/hub/tmp*

accelerate launch \
  --mixed_precision bf16 \
  --num_machines 1 \
  --num_processes 2 \
  --use_deepspeed \
  --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
  src/finetune.py \
  --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
  --use_flash_attn \
  --tokenizer_name meta-llama/Llama-3.1-8B-Instruct \
  --use_slow_tokenizer
  --train_file data/silverpairs_prompt_completion.jsonl \
  --enable_liger_kernel \
  --trust_remote_code \
  --max_seq_length 42000 \
  --preprocessing_num_workers 16 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.05 \
  --weight_decay 0.0 \
  --checkpointing_steps 50 \
  --num_train_epochs 6 \
  --ddp_timeout 180000000 \
  --output_dir output/llama3-8b_im_sft \
  --use_lora \
  --lora_rank 8 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lora_target q_proj,v_proj \
  --with_tracking \
  --report_to wandb \
  --logging_steps 1 \
  --use_lm_loss \
> output/llama3-8b-im.log 2>&1