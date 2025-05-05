import os
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from src.utils.other import set_seeds

import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig, TaskType
from src.models import MODELS_MAPPING


@hydra.main(version_base=None, config_path="../config", config_name="config_sl")
def main(cfg: DictConfig):
    set_seeds(cfg.seed)

    model_class = MODELS_MAPPING.get(
        cfg.model.model_family, {}
        ).get(cfg.model.task)
    assert model_class, f"Model family {cfg.model.model_family} and task {cfg.model.task} not supported."

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.pretrained_model_name_or_path
    )

    if cfg.model.quantization:
        bnb_4bit_compute_dtype = cfg.model.quantization.pop('bnb_4bit_compute_dtype')
        quant_config = BitsAndBytesConfig(
            **cfg.model.quantization,
            bnb_4bit_compute_dtype=getattr(torch, bnb_4bit_compute_dtype),
        )

    torch_dtype = cfg.model.model.pop('torch_dtype')
    model = model_class.from_pretrained(
        **cfg.model.model,
        torch_dtype=getattr(torch, torch_dtype),
        quantization_config=quant_config if cfg.model.quantization else None,
    )

    if cfg.model.use_lora:
        lora_config = LoraConfig(
            **cfg.model.lora
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    train_args = TrainingArguments(
        output_dir=f'./checkpoints/{cfg.wandb.name}',
        logging_dir=f'./logs/{cfg.wandb.name}',
        bf16=cfg.model.use_bfloat16,
        **cfg.train.training_args
    )

    # DATASET
    # CALLBACKS
    trainer = Trainer(
        model=model, 
        args=train_args, 
        train_dataset=dataset.train,
        eval_dataset=dataset.val,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )


    wandb_run = wandb.init(
        **cfg.wandb,
        config={'hydra': OmegaConf.to_container(cfg, resolve=True)}
    )

    trainer.train()

    wandb_run.finish()

if __name__ == "__main__":
    main()