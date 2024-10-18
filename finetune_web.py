import os
import json
import logging
from typing import Optional, List
from dataclasses import dataclass, field

import torch
import transformers
from transformers import Trainer, deepspeed
from accelerate.utils import DistributedType

from monkey_model.modeling_monkey import MonkeyLMHeadModel
from monkey_model.tokenization_qwen import QWenTokenizer
from monkey_model.configuration_monkey import MonkeyConfig
from edge_dataset.dataset import EDGETensorDataset
from utils.utils_ddp import rank0_print
from utils.utils_training import fix_model_params, print_trainable_params, setup_seed


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="")


@dataclass
class DataArguments:
    data_path: str = field(
        default="", metadata={"help": "Path to the training data."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = False
    fix_vit: bool = True
    ddp_find_unused_parameters: bool = False
    dataloader_num_workers: int = 16
    max_grad_norm: float = 2.0


@dataclass
class LoraArguments:
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["in_proj","out_proj","c_fc"] ##["in_proj","out_proj","c_fc"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False
        

def safe_save_model_for_hf_trainer(trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)


def create_data_module(tokenizer, data_args) -> dict:
    """Make dataset and collator for supervised fine-tuning."""
    
    with open(data_args.data_path) as f:
        dataset_meta = json.load(f)
    dataset_train = EDGETensorDataset(tokenizer, dataset_meta=dataset_meta["train"])
    # dataset_train = EDGETensorDataset(tokenizer, items_filepath="edge_dataset/processed/test_data.jsonl")

    logging.debug("build dataset, done.")
    logging.debug(f'number of samples in training dataset: {len(dataset_train)}')
    # logging.debug(f'number of samples in val dataset: {len(dataset_val)}')
    
    return dict(
        train_dataset=dataset_train, 
        # eval_dataset=dataset_val
    )


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, LoraArguments))
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()

    if getattr(training_args, 'deepspeed', None) and getattr(lora_args, 'q_lora', False):
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    if lora_args.q_lora:
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning("FSDP or ZeRO3 are not incompatible with QLoRA.")

    ### Load tokenizer
    tokenizer = QWenTokenizer.from_pretrained("monkey_model", model_max_length=training_args.model_max_length)
    
    ### Load model
    rank0_print("Initializing config and model ...")
    config = MonkeyConfig.from_pretrained(model_args.model_name_or_path)
    config.use_cache = False
    model = MonkeyLMHeadModel.from_pretrained(model_args.model_name_or_path, config=config, torch_dtype=torch.bfloat16)
    model = fix_model_params(model, model_args, training_args, lora_args)
    print_trainable_params(model)
    
    ### Load data
    data_module = create_data_module(tokenizer=tokenizer, data_args=data_args)

    ### Start trainner
    trainer_args = dict(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args, 
        **data_module
    )
    trainer = Trainer(**trainer_args)
    
    trainer.train()
    trainer.save_state()

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    setup_seed(46)
    train()