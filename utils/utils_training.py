import os
import random
import numpy as np
from time import time

import torch
from peft import LoraConfig, get_peft_model

from .utils_ddp import is_main_process, rank0_print


def print_trainable_params(model):
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    rank0_print("trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
        trainable_params, all_param, 100 * trainable_params / all_param))


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    

def fix_model_params(model, model_args, training_args, lora_args):
    
    if not training_args.use_lora:
        if training_args.fix_vit:
            model.transformer.visual.requires_grad_(False)
        model.transformer.visual.attn_pool.requires_grad_(True)
        model.transformer.visual.ln_post.requires_grad_(True)
        model.transformer.visual.proj.requires_grad_(True)
        # for k, v in model.named_parameters():
        #     if "lora" in k :
        #         v.requires_grad_(True)

    if training_args.use_lora:
        if lora_args.q_lora or "chat" in model_args.model_name_or_path.lower():
            modules_to_save = None
        else:
            modules_to_save = []
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
            modules_to_save=modules_to_save  # This argument serves for adding new tokens.
        )
        model = get_peft_model(model, lora_config)
        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()
            
    return model


class Timer:
    enable = False
    init_time = prev_time = -1
    is_main_process = None
    
    @classmethod
    def reset(cls):
        if not cls.enable:
            return
        cls.is_main_process = is_main_process()
        if cls.is_main_process:
            cls.init_time = cls.prev_time = time()
    
    @classmethod
    def timing_interval(cls, name):
        if not cls.enable:
            return
        if cls.is_main_process:
            curr_time = time()
            interval_time = curr_time - cls.prev_time
            cls.prev_time = curr_time
            print(f"{name} TIME: {interval_time:.4f}")
            
    @classmethod
    def timing_total(cls, name):
        if not cls.enable:
            return
        if cls.is_main_process:
            curr_time = time()
            total_time = curr_time - cls.init_time
            print(f"{name} TIME (SO FAR): {total_time:.4f}")
