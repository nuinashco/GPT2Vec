import os
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from src.utils.other import set_seeds

import torch
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
from src.models import MODELS_MAPPING
from src.callbacks.partial_grad_norm import PartialGradNormCallback

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    set_seeds(cfg.seed)

    model_class = MODELS_MAPPING.get(
        cfg.model.model_family, {}
        ).get(cfg.train.task)
    assert model_class, f"Model family {cfg.model.model_family} and task {cfg.model.task} not supported."

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.model.pretrained_model_name_or_path
    )
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    if cfg.model.quantization:
        quant_cfg = OmegaConf.to_container(cfg.model.quantization, resolve=True)
        quant_cfg["bnb_4bit_compute_dtype"] = getattr(torch, quant_cfg["bnb_4bit_compute_dtype"])
        quant_config = BitsAndBytesConfig(
            **quant_cfg
        )

    model_cfg = OmegaConf.to_container(cfg.model.model, resolve=True)
    model_cfg['torch_dtype'] = getattr(torch, model_cfg['torch_dtype'])
    model = model_class.from_pretrained(
        **model_cfg,
        quantization_config=quant_config if cfg.model.quantization else None,
    )

    if cfg.model.lora:
        lora_cfg = OmegaConf.to_container(cfg.model.lora, resolve=True)
        lora_config = LoraConfig(**lora_cfg)
        model = get_peft_model(model, lora_config)

        model.print_trainable_parameters()

    train_args = TrainingArguments(
        output_dir=f'./checkpoints/{cfg.wandb.name}',
        logging_dir=f'./logs/{cfg.wandb.name}',
        **cfg.train.training_args
    )

    # DATASET
    ds = load_dataset(**cfg.data.dataset)
    sampled_ds = ds.shuffle(
        seed=cfg.data.sample.seed
        ).select(range(cfg.data.sample.num_samples)
        ).select_columns(["text"])
    
    tokenized_dataset = sampled_ds.map(
        lambda examples: tokenizer(examples["text"], **cfg.data.tokenize),
        batched=True, num_proc=10
        ).remove_columns(["text"])
    

    tokenizer.mask_token = cfg.train.mask_token
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.2
    )


    # TODO: CALLBACKS
    # gradnorm_cb = PartialGradNormCallback(
    #     model,
    #     target_modules=["lm_head"],  # LoRA‑style patterns
    #     # target_layers=range(0, 6),              # layers 0‑5 inclusive
    #     log_key="=grad_norm",
    # )

    wandb_run = wandb.init(
        **cfg.wandb,
        config={'hydra': OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)}
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
        # callbacks=[gradnorm_cb]
    )

    trainer.train()

    wandb_run.finish()

if __name__ == "__main__":
    main()