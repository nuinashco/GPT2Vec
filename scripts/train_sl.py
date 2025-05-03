import os
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from src.utils.other import set_seeds

import torch
from src.datasets import KEY2DATASET
from src.metrics import KEY2CLF_METRIC
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
from peft import get_peft_model, LoraConfig, TaskType


@hydra.main(version_base=None, config_path="../config", config_name="config_sl")
def main(cfg: DictConfig):
    set_seeds(cfg.seed)

    torch_dtype = torch.bfloat16 if cfg.model.use_bfloat16 else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.pretrained_model_name_or_path
    )
    tokenizer.add_eos_token = True  # We'll add <eos> at the end
    tokenizer.padding_side = "right"

    dataset = KEY2DATASET[cfg.data.dataset_type](
        tokenizer=tokenizer,
        splits=('train', 'validation'),
        **cfg.data.dataset_params
        )
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    model = AutoModelForTokenClassification.from_pretrained(
        cfg.model.pretrained_model_name_or_path,
        id2label=dataset.id2label,
        label2id=dataset.label2id,
        torch_dtype=torch_dtype
    )

    if cfg.model.use_lora:
        lora_config = LoraConfig(
            **cfg.model.lora
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    train_args = TrainingArguments(
        output_dir=f'./model_checkpoints_{cfg.wandb.name}',
        logging_dir=f'./model_logs_{cfg.wandb.name}',
        bf16=cfg.model.use_bfloat16,
        **cfg.train.training_args
    )

    compute_metrics = KEY2CLF_METRIC[cfg.train.metric](
        label2id=dataset.label2id
    )
    trainer = Trainer(
        model=model, 
        args=train_args, 
        train_dataset=dataset.train,
        eval_dataset=dataset.val,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    # wandb_run = wandb.init(
    #     **cfg.wandb,
    #     config=cfg
    # )

    trainer.train()

    # wandb_run.finish()

if __name__ == "__main__":
    main()