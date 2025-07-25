#!/usr/bin/env python
# coding=utf-8

import argparse
import logging
import math
import os
import random
import datasets
import torch
import json
from functools import partial
from itertools import chain
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import LlamaForCausalLM
from transformers import LlamaConfig
from huggingface_hub import hf_hub_download
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaTokenizerFast,
    SchedulerType,
    DataCollatorForSeq2Seq,
    default_data_collator,
    get_scheduler,
    GPTNeoXTokenizerFast,
    GPT2Tokenizer,
    OPTForCausalLM,
    BitsAndBytesConfig,
)
from transformers.modeling_utils import unwrap_model
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

from utils import (
    compute_kl_divergence_loss,
    compute_kl_divergence_loss_target_token,
    neftune_post_forward_hook,
)
import deepspeed
deepspeed.ops.op_builder.CPUAdamBuilder().load()

# Liger kernel import (conditional)
try:
    import liger_kernel
    LIGER_KERNEL_AVAILABLE = True
except ImportError:
    LIGER_KERNEL_AVAILABLE = False

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="If passed, will use LORA (low-rank parameter-efficient training) to train the model.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=64,
        help="The rank of lora.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=16,
        help="The alpha parameter of lora.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="The dropout rate of lora modules.",
    )
    parser.add_argument(
        "--use_flash_attn",
        action="store_true",
        help="If passed, will use flash attention to train the model.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="The maximum total sequence length (prompt+completion) of each training example.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float,
                        default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts",
                 "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--warmup_ratio", type=float, default=0, help="Ratio of total training steps used for warmup."
    )
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None,
                        help="A seed for reproducible training.")
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=None,
        help="Log the training loss and learning rate every logging_steps steps.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help=(
            "Turn on gradient checkpointing. Saves memory but slows training."
        ),
    )
    parser.add_argument(
        "--use_qlora",
        action="store_true",
        help=(
            "Use qLoRA training - main thing is initialising model in quantised form. Not compatible with deepspeed."
        ),
    )
    parser.add_argument(
        '--clip_grad_norm',
        type=float,
        default=-1,
        help='Clip gradient norm. Not compatible with deepspeed (use deepspeed config instead).',
    )
    parser.add_argument(
        '--use_8bit_optimizer',
        action='store_true',
        help='Use 8bit optimizer from bitsandbytes. Not compatible with deepspeed (use deepspeed config instead).',
    )
    parser.add_argument(
        '--apply_kl_div_loss',
        action='store_true',
        help='Apply KL divergence loss between output logits and reference logits.',
    )
    parser.add_argument(
        '--kl_penalty_ctl',
        type=float,
        default=1.0,
        help='Control parameter for KL divergence loss.',
    )
    # parser.add_argument(
    #     '--kl_penalty_type',
    #     type=str,
    #     default='full',
    #     help='Type of KL divergence loss. We support "target_token" and "full".',
    # )
    parser.add_argument(
        '--use_lm_loss',
        action='store_true',
        help='Use language modeling loss to train instruction tuning rather than masking the loss over instructions.',
    )
    parser.add_argument(
        '--block_size',
        type=int,
        default=None,
        help='Optional input sequence length after tokenization. The training dataset will be truncated in block of this size for training. Default to the model max input length for single sentence inputs (take into account special tokens).',
    )
    parser.add_argument(
        '--use_lm_modelling',
        action='store_true',
        help='Concatenate all instruction tuning examples for SFT training.',
    )
    parser.add_argument(
        '--padding_side',
        type=str,
        default=None,
        help='The side on which to pad the inputs. Select from ["left", "right"].',
    )
    parser.add_argument(
        '--neftune_alpha',
        type=float,
        default=None,
        help='The alpha parameter for NEFTune.',
    ),
    parser.add_argument(
        "--lora_target",
        type=str,
        default="q_proj,v_proj",
        help="Comma-separated list of target modules for LoRA (e.g. 'q_proj,v_proj')",
    ),
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="A name for this run (e.g. for wandb or tensorboard tracking)."
    ),
    parser.add_argument(
        "--rope_scaling_type",
        type=str,
        choices=["linear", "dynamic"],
        help="Type of rope scaling to use if overriding default config."
    ),
    parser.add_argument(
        "--rope_scaling_factor",
        type=float,
        help="Factor to apply if overriding rope scaling config."
    ),
    parser.add_argument(
        "--enable_liger_kernel",
        action="store_true",
        help="Enable Liger kernel optimizations for faster and more memory-efficient training.",
    )


    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None:
        raise ValueError("Need either a dataset name or a training file.")
    else:
        if args.train_file is not None and not args.apply_kl_div_loss:
            extension = args.train_file.split(".")[-1]
            assert extension in [
                "json", "jsonl"], "`train_file` should be a json/jsonl file."
    
    # Validation for Liger kernel
    if args.enable_liger_kernel and not LIGER_KERNEL_AVAILABLE:
        raise ValueError("Liger kernel is not available. Please install it with: pip install liger-kernel")
    
    return args


def encode_with_prompt_completion_format(example, tokenizer, max_seq_length, use_lm_loss=False):
    '''
    Here we assume each example has 'prompt' and 'completion' fields.
    We concatenate prompt and completion and tokenize them together because otherwise prompt will be padded/trancated 
    and it doesn't make sense to follow directly with the completion.
    '''
    # if prompt doesn't end with space and completion doesn't start with space, add space
    if not example['prompt'].endswith((' ', '\n', '\t')) and not example['completion'].startswith((' ', '\n', '\t')):
        example_text = example['prompt'] + ' ' + example['completion']
    else:
        example_text = example['prompt'] + example['completion']
    example_text = example_text + tokenizer.eos_token
    tokenized_example = tokenizer(
        example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()
    tokenized_prompt = tokenizer(
        example['prompt'], return_tensors='pt', max_length=max_seq_length, truncation=True)
    # mask the prompt part for avoiding loss
    if not use_lm_loss:
        labels[:, :tokenized_prompt.input_ids.shape[1]] = -100
    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }


def encode_with_messages_format(example, tokenizer, max_seq_length, use_lm_loss=False):
    '''
    Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    '''
    messages = example['messages']
    if len(messages) == 0:
        raise ValueError('messages field is empty.')

    def _concat_messages(messages):
        message_text = ""
        for message in messages:
            if message["role"] == "system":
                message_text += "<|system|>\n" + \
                    message["content"].strip() + "\n"
            elif message["role"] == "user":
                message_text += "<|user|>\n" + \
                    message["content"].strip() + "\n"
            elif message["role"] == "assistant":
                message_text += "<|assistant|>\n" + \
                    message["content"].strip() + tokenizer.eos_token + "\n"
            else:
                raise ValueError("Invalid role: {}".format(message["role"]))
        return message_text

    example_text = _concat_messages(messages).strip()
    tokenized_example = tokenizer(
        example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    if use_lm_loss:
        # mask the special tokens for avoiding loss
        # here we aovid loss when tokens are <|assistant|>\n, <|system|>\n, or <|user|>\n.
        for special_token in ["wa\n", "<|system|>\n", "<|user|>\n"]:
            special_token_ids = tokenizer(
                special_token, return_tensors='pt', max_length=max_seq_length, truncation=True).input_ids
            length_special_token = special_token_ids.shape[1]
            for idx in range(input_ids.shape[1] - length_special_token + 1):
                if torch.equal(input_ids[:, idx:idx+length_special_token], special_token_ids):
                    labels[:, idx:idx+length_special_token] = -100
    else:
        # mask the non-assistant part for avoiding loss
        for message_idx, message in enumerate(messages):
            if message["role"] != "assistant":
                if message_idx == 0:
                    message_start_idx = 0
                else:
                    message_start_idx = tokenizer(
                        _concat_messages(messages[:message_idx]), return_tensors='pt', max_length=max_seq_length, truncation=True
                    ).input_ids.shape[1]
                if message_idx < len(messages) - 1 and messages[message_idx+1]["role"] == "assistant":
                    # here we also ignore the role of the assistant
                    messages_so_far = _concat_messages(
                        messages[:message_idx+1]) + "<|assistant|>\n"
                else:
                    messages_so_far = _concat_messages(
                        messages[:message_idx+1])
                message_end_idx = tokenizer(
                    messages_so_far,
                    return_tensors='pt',
                    max_length=max_seq_length,
                    truncation=True
                ).input_ids.shape[1]
                labels[:, message_start_idx:message_end_idx] = -100

                if message_end_idx >= max_seq_length:
                    break

    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }


# Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
def group_texts(examples, block_size):
    # Convert examples from tensor to list of lists.
    # examples = {k: [e.tolist() for e in v] for k, v in examples.items()}
    # Concatenate all texts.
    concatenated_examples = {
        k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    # result["labels"] = result["input_ids"].copy()
    return result


def save_with_accelerate(accelerator, model, tokenizer, output_dir, args):
    unwrapped_model = accelerator.unwrap_model(model)
    # When doing multi-gpu training, we need to use accelerator.get_state_dict(model) to get the state_dict.
    # Otherwise, sometimes the model will be saved with only part of the parameters.
    # Also, accelerator needs to use the wrapped model to get the state_dict.
    state_dict = accelerator.get_state_dict(model)
    
    if args.use_lora:
        # When using lora, the unwrapped model is a PeftModel, which doesn't support the is_main_process
        # and has its own save_pretrained function for only saving lora modules.
        # We have to manually specify the is_main_process outside the save_pretrained function.
        if accelerator.is_main_process:
            try:
                # Try the standard PEFT save first
                unwrapped_model.save_pretrained(output_dir, state_dict=state_dict)
                print("✅ Successfully saved with standard PEFT method")
            except Exception as e:
                print(f"⚠️ Standard PEFT save failed: {e}")
                print("🔧 Using manual LoRA save method...")
                
                # Manual save approach - extract LoRA weights directly
                os.makedirs(output_dir, exist_ok=True)
                
                # Save adapter weights manually
                adapter_state_dict = {}
                for name, param in unwrapped_model.named_parameters():
                    if 'lora_' in name:
                        # Remove 'base_model.model.' prefix if present for cleaner naming
                        clean_name = name.replace('base_model.model.', '') if 'base_model.model.' in name else name
                        adapter_state_dict[clean_name] = param.detach().cpu()
                
                if adapter_state_dict:
                    torch.save(adapter_state_dict, os.path.join(output_dir, "adapter_model.bin"))
                    print(f"✅ Saved {len(adapter_state_dict)} LoRA parameters")
                
                # Create adapter_config.json manually
                adapter_config = {
                    "base_model_name_or_path": args.model_name_or_path,
                    "bias": "none",
                    "fan_in_fan_out": False,
                    "inference_mode": False,
                    "init_lora_weights": True,
                    "layers_pattern": None,
                    "layers_to_transform": None,
                    "lora_alpha": args.lora_alpha,
                    "lora_dropout": args.lora_dropout,
                    "modules_to_save": None,
                    "peft_type": "LORA", 
                    "r": args.lora_rank,
                    "revision": None,
                    "target_modules": args.lora_target.split(",") if hasattr(args, 'lora_target') and args.lora_target else ["q_proj", "v_proj"],
                    "task_type": "CAUSAL_LM"
                }
                
                with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
                    json.dump(adapter_config, f, indent=2)
                print("✅ Created adapter_config.json")
                
                # Save a README file
                readme_content = f"""---
library_name: peft
base_model: {args.model_name_or_path}
---

# LoRA Adapter

This adapter was trained using LoRA (Low-Rank Adaptation) technique.

## Model Details
- Base model: {args.model_name_or_path}
- LoRA rank: {args.lora_rank}
- LoRA alpha: {args.lora_alpha}
- Target modules: {args.lora_target if hasattr(args, 'lora_target') else 'q_proj,v_proj'}

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("{args.model_name_or_path}")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "{output_dir}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("{args.model_name_or_path}")
```
"""
                with open(os.path.join(output_dir, "README.md"), "w") as f:
                    f.write(readme_content)
                print("✅ Created README.md")
                
                print("✅ Manual LoRA save completed successfully!")
    else:
        unwrapped_model.save_pretrained(
            output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save, state_dict=state_dict
        )


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)
    # Make one log on every process with the configuration for debugging.
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.FileHandler(os.path.join(
            args.output_dir, 'output.log'), mode='w')]
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Log on each process the small summary:
    logger.info(f"Training/evaluation parameters {args}")
    
    # Log Liger kernel status
    if args.enable_liger_kernel:
        if LIGER_KERNEL_AVAILABLE:
            print("🚀 LIGER KERNEL: Liger kernel is enabled and available")
            logger.info("🚀 Liger kernel is enabled and available")
        else:
            print("❌ LIGER KERNEL: Liger kernel is requested but not available")
            logger.error("❌ Liger kernel is requested but not available")
    else:
        print("⚡ LIGER KERNEL: Liger kernel is disabled")
        logger.info("⚡ Liger kernel is disabled")
    
    with open(os.path.join(args.output_dir, "finetune_args.txt"), "w") as f:
        f.write(str(args))

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    accelerator.wait_for_everyone()

    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
        )
    else:
        if args.apply_kl_div_loss:
            lm_datasets = load_from_disk(args.train_file)
        else:
            data_files = {}
            dataset_args = {}
            if args.train_file is not None:
                data_files["train"] = args.train_file
            raw_datasets = load_dataset(
                "json",
                data_files=data_files,
                **dataset_args,
            )

    
    # Load config dict
    if args.config_name:
        config_dict, _ = LlamaConfig.get_config_dict(args.config_name)
    elif args.model_name_or_path:
        config_dict, _ = LlamaConfig.get_config_dict(args.model_name_or_path)
    else:
        raise ValueError("Must specify --config_name or --model_name_or_path")

    # Optional: show original rope_scaling
    if "rope_scaling" in config_dict:
        print("⚠️ Original rope_scaling in config:", config_dict["rope_scaling"])

    # Handle LLaMA 3.1 rope_scaling compatibility
    if "rope_scaling" in config_dict and config_dict["rope_scaling"] is not None:
        rope_scaling = config_dict["rope_scaling"]
        if isinstance(rope_scaling, dict) and "rope_type" in rope_scaling:
            # This is LLaMA 3.1 format, convert to compatible format for config creation
            compatible_rope_scaling = {
                "type": rope_scaling.get("rope_type", "default"),
                "factor": rope_scaling.get("factor", 1.0)
            }
            config_dict["rope_scaling"] = compatible_rope_scaling
            print("⚠️ Converted LLaMA 3.1 rope_scaling to compatible format for config loading")

    # Patch rope_scaling if specified in arguments
    if args.rope_scaling_type and args.rope_scaling_factor:
        config_dict["rope_scaling"] = {
            "type": args.rope_scaling_type,
            "factor": args.rope_scaling_factor,
        }
        print("✅ Patched rope_scaling:", config_dict["rope_scaling"])

    # ✅ Must use LlamaConfig.from_dict()
    config = LlamaConfig.from_dict(config_dict)

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply Liger kernel optimizations BEFORE model loading
    if args.enable_liger_kernel:
        print("🚀 LIGER KERNEL: Applying Liger kernel optimizations...")
        logger.info("🚀 Applying Liger kernel optimizations...")
        try:
            # Apply Liger kernel to the model based on model type
            if "llama" in args.model_name_or_path.lower():
                from liger_kernel.transformers import apply_liger_kernel_to_llama
                apply_liger_kernel_to_llama()
                print("✅ LIGER KERNEL: Liger kernel for LLaMA successfully applied")
                logger.info("✅ Liger kernel for LLaMA successfully applied")
            elif "mistral" in args.model_name_or_path.lower():
                from liger_kernel.transformers import apply_liger_kernel_to_mistral
                apply_liger_kernel_to_mistral()
                print("✅ LIGER KERNEL: Liger kernel for Mistral successfully applied")
                logger.info("✅ Liger kernel for Mistral successfully applied")
            elif "gemma" in args.model_name_or_path.lower():
                from liger_kernel.transformers import apply_liger_kernel_to_gemma
                apply_liger_kernel_to_gemma()
                print("✅ LIGER KERNEL: Liger kernel for Gemma successfully applied")
                logger.info("✅ Liger kernel for Gemma successfully applied")
            else:
                # Try the general LLaMA kernel as fallback for similar architectures
                from liger_kernel.transformers import apply_liger_kernel_to_llama
                apply_liger_kernel_to_llama()
                print("✅ LIGER KERNEL: Liger kernel (LLaMA fallback) successfully applied")
                logger.info("✅ Liger kernel (LLaMA fallback) successfully applied")
        except ImportError as e:
            print(f"⚠️ LIGER KERNEL: Liger kernel import failed: {e}")
            logger.warning(f"⚠️ Liger kernel import failed: {e}")
            logger.warning("Continuing without Liger kernel optimizations...")
        except Exception as e:
            print(f"⚠️ LIGER KERNEL: Failed to apply Liger kernel: {e}")
            logger.warning(f"⚠️ Failed to apply Liger kernel: {e}")
            logger.warning("Continuing without Liger kernel optimizations...")

    if args.model_name_or_path:
        if args.use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            device_index = accelerator.local_process_index
            device_map = {"": device_index}  # force data-parallel training.
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                load_in_4bit=True,
                quantization_config=bnb_config,
                device_map=device_map,
                torch_dtype=torch.bfloat16,
                use_flash_attention_2=True if args.use_flash_attn else False,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                low_cpu_mem_usage=args.low_cpu_mem_usage,
                use_flash_attention_2=True if args.use_flash_attn else False,
            )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)

        
    if args.use_flash_attn:
        model.to("cuda")  # Explicitly move the model if needed

    if args.use_lm_modelling:
        logger.info("Training with LM modelling loss.")
        logger.info(
            "Here we concatenate all instruction tuning examples for SFT training.")
        logger.info(
            "We do not add any special tokens, as we mimic what the model was trained during pretraining.")
    else:
        # no default pad token for llama!
        # here we add all special tokens again, because the default ones are not in the special_tokens_map
        logger.info(
            "Training with regular instruction tuning using special tokens.")
        logger.info("Adding special tokens to the tokenizer.")
        if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, LlamaTokenizerFast):
            num_added_tokens = tokenizer.add_special_tokens({
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
            })
            assert num_added_tokens in [
                0, 1], "LlamaTokenizer should only add one special token - the pad_token, or no tokens if pad token present."
        elif isinstance(tokenizer, GPTNeoXTokenizerFast):
            num_added_tokens = tokenizer.add_special_tokens({
                "pad_token": "<pad>",
            })
            assert num_added_tokens == 1, "GPTNeoXTokenizer should only add one special token - the pad_token."
        elif isinstance(tokenizer, GPT2Tokenizer) and isinstance(model, OPTForCausalLM):
            num_added_tokens = tokenizer.add_special_tokens(
                {'unk_token': '<unk>'})

    if args.padding_side is not None:
        assert args.padding_side in [
            "left", "right"], "Padding side should be either 'left' or 'right'."
        tokenizer.padding_side = args.padding_side

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if args.use_lora:
        if args.use_qlora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=args.gradient_checkpointing)

        logger.info("Initializing LORA model...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "o_proj", "v_proj",
                            "k_proj", "gate_proj", "up_proj", "down_proj"]
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # Preprocessing the datasets.
    if not args.apply_kl_div_loss:
        if "prompt" in raw_datasets["train"].column_names and "completion" in raw_datasets["train"].column_names:
            encode_function = partial(
                encode_with_prompt_completion_format,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                use_lm_loss=args.use_lm_loss,
            )
        elif "messages" in raw_datasets["train"].column_names:
            encode_function = partial(
                encode_with_messages_format,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                use_lm_loss=args.use_lm_loss,
            )
        else:
            raise ValueError(
                "You need to have either 'prompt'&'completion' or 'messages' in your column names.")
        logger.info(
            "Whether use LM loss to train SFT: {}".format(args.use_lm_loss))

        with accelerator.main_process_first():
            lm_datasets = raw_datasets.map(
                encode_function,
                batched=False,
                num_proc=args.preprocessing_num_workers,
                load_from_cache_file=not args.overwrite_cache,
                remove_columns=[name for name in raw_datasets["train"].column_names if name not in [
                    "input_ids", "labels", "attention_mask"]],
                desc="Tokenizing and reformatting instruction data",
            )
            lm_datasets.set_format(type="pt")
            logger.info("The number of examples in the training dataset: {}".format(
                len(lm_datasets['train'])))
            lm_datasets = lm_datasets.filter(
                lambda example: (example['labels'] != -100).any())
            logger.info("The number of examples in the training dataset after filtering: {}".format(
                len(lm_datasets['train'])))

        if hasattr(config, "max_position_embeddings"):
            max_pos_embeddings = config.max_position_embeddings
        else:
            # Define a default value if the attribute is missing in the config.
            max_pos_embeddings = args.max_seq_length

        if args.use_lm_modelling:
            if args.block_size is None:
                block_size = args.max_seq_length
                if block_size > max_pos_embeddings:
                    logger.warning(
                        f"The tokenizer picked seems to have a very large `model_max_length` ({args.max_seq_length}). "
                        f"Using block_size={min(1024, max_pos_embeddings)} instead. You can change that default value by passing --block_size xxx."
                    )
            else:
                if args.block_size > args.max_seq_length:
                    logger.warning(
                        f"The block_size passed ({args.block_size}) is larger than the maximum length for the model "
                        f"({args.max_seq_length}). Using block_size={args.max_seq_length}."
                    )
                block_size = min(args.block_size, args.max_seq_length)

            logger.info(
                "Concatenate all instruction tuning examples for SFT training.")
            with accelerator.main_process_first():
                # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
                # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
                # to preprocess.
                if "lima" in args.train_file:
                    batch_size = 1030
                elif "alpagasus_3k" in args.train_file:
                    batch_size = 2996
                elif "alpagasus_9k" in args.train_file:
                    batch_size = 9229
                elif "alpagasus_claude_t45_alpaca" in args.train_file:
                    batch_size = 5305
                else:
                    batch_size = 1000
                logger.info(
                    f"The number of examples in the training dataset: {len(lm_datasets['train'])}")
                logger.info(f"Batch size for grouping texts: {batch_size}")

                lm_datasets = lm_datasets.map(
                    partial(group_texts, block_size=block_size),
                    batched=True,
                    batch_size=batch_size,
                    num_proc=args.preprocessing_num_workers,
                    load_from_cache_file=not args.overwrite_cache,
                    remove_columns=[name for name in lm_datasets["train"].column_names if name not in [
                        "input_ids", "labels", "attention_mask"]],
                    desc=f"Grouping texts in chunks of {args.block_size}",
                )

    train_dataset = lm_datasets["train"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(
            f"Sample {index} of the training set: {train_dataset[index]}.")

    if args.use_lm_modelling:
        # DataLoaders creation:
        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=default_data_collator,
            batch_size=args.per_device_train_batch_size
        )
    else:
        # DataLoaders creation:
        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=DataCollatorForSeq2Seq(
                tokenizer=tokenizer, model=model, padding="longest"),
            batch_size=args.per_device_train_batch_size
        )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    if args.use_qlora:
        from bitsandbytes.optim import AdamW
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            optim_bits=8 if args.use_8bit_optimizer else 32,
            is_paged=True
        )
    else:
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Create the learning rate scheduler.
    # Note: the current accelerator.step() calls the .step() of the real scheduler for the `num_processes` times. This is because they assume
    # the user initialize the scheduler with the entire training set. In the case of data parallel training, each process only
    # sees a subset (1/num_processes) of the training set. So each time the process needs to update the lr multiple times so that the total
    # number of updates in the end matches the num_training_steps here.
    # Here we need to set the num_training_steps to either using the entire training set (when epochs is specified) or we need to multiply the
    # num_training_steps by num_processes so that the total number of updates matches the num_training_steps.
    num_training_steps_for_scheduler = args.max_train_steps if overrode_max_train_steps else args.max_train_steps * \
        accelerator.num_processes
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_training_steps=num_training_steps_for_scheduler,
        num_warmup_steps=int(
            num_training_steps_for_scheduler * args.warmup_ratio),
    )

    # Prepare everything with `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(
        args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers(
            project_name="open_instruct",
            config=experiment_config,
            init_kwargs={"wandb": {"name": args.run_name}
                         } if args.run_name and args.report_to == "wandb" else None
        )

    # Train!
    total_batch_size = args.per_device_train_batch_size * \
        accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps),
                        disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[
                -1
            ]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = (
                int(training_difference.replace("step_", ""))
                * args.gradient_accumulation_steps
            )
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    if args.neftune_alpha is not None:
        unwrapped_model = unwrap_model(model)
        embeddings = unwrapped_model.get_input_embeddings()
        hook_handle = embeddings.register_forward_hook(
            partial(neftune_post_forward_hook,
                    neftune_noise_alpha=args.neftune_alpha),
        )

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_loss = 0
        if (
            args.resume_from_checkpoint
            and epoch == starting_epoch
            and resume_step is not None
        ):
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(
                train_dataloader, resume_step
            )
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            # with accelerator.accumulate(model):

            if args.apply_kl_div_loss:
                ref_outputs_probs = batch.pop("ref_model_logits")

            outputs = model(**batch, use_cache=False)
            loss = outputs.loss

            if args.apply_kl_div_loss:
                kl_loss = compute_kl_divergence_loss_target_token(
                    output_logits=outputs.logits,
                    ref_logprobs=ref_outputs_probs,
                    input_ids=batch["input_ids"],
                    labels=batch["labels"],
                )
                loss += args.kl_penalty_ctl * kl_loss

            # We keep track of the loss at each logged step
            total_loss += loss.detach().float()

            accelerator.backward(loss)
            # clip gradient norm. don't do this with deepspeed
            if accelerator.sync_gradients and args.clip_grad_norm > 0:
                accelerator.clip_grad_norm_(
                    model.parameters(), args.clip_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                if args.logging_steps and completed_steps % args.logging_steps == 0:
                    avg_loss = accelerator.gather(total_loss).mean().item(
                    ) / args.gradient_accumulation_steps / args.logging_steps
                    logger.info(
                        f"  Step: {completed_steps}, LR: {lr_scheduler.get_last_lr()[0]}, Loss: {avg_loss}")
                    if args.with_tracking:
                        accelerator.log(
                            {
                                "learning_rate": lr_scheduler.get_last_lr()[0],
                                "train_loss": avg_loss,
                            },
                            step=completed_steps,
                        )
                    total_loss = 0

                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(
                                args.output_dir, output_dir)
                        save_with_accelerate(
                            accelerator, model, tokenizer, output_dir, args)

                if completed_steps >= args.max_train_steps:
                    break

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            save_with_accelerate(accelerator, model,
                                 tokenizer, output_dir, args)

    if args.neftune_alpha is not None:
        hook_handle.remove()

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
        save_with_accelerate(accelerator, model, tokenizer,
                             args.output_dir, args)


if __name__ == "__main__":
    main()
