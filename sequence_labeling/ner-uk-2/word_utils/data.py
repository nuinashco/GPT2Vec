from typing import List, Dict, Any, Optional, Tuple

import datasets
import numpy as np
import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase


class NerUKDataset:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        dataset_name: str,
        splits: Tuple[str, ...] = ("train", "validation", "test"),
        ignore_id: int = -100,
        retroactive_labels: str = "same_token",
        model_class: str = "custom",
        max_length: int = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.ignore_id = ignore_id
        self.retroactive_labels = retroactive_labels
        self.model_class = model_class
        self.max_length = max_length
        self.raw_ds = load_dataset(dataset_name, trust_remote_code=True)

        # Extract label information from dataset
        self.label_list = self.raw_ds["train"].features["ner_tags"].feature.names
        self.label2id = {label: i for i, label in enumerate(self.label_list)}
        self.id2label = {i: label for i, label in enumerate(self.label_list)}

        # Process each requested split
        self.aligned = {}
        for split in splits:
            if split not in self.raw_ds:
                continue
                
            self.aligned[split] = self.raw_ds[split].map(
                self.tokenize_and_align_labels,
                remove_columns=self.raw_ds["train"].column_names,
                batched=True,
                load_from_cache_file=False,
                desc=f"Tokenizing {split} split",
            )
            self.aligned[split].set_format(type="torch")


    def tokenize_and_align_labels(self, examples):
        tokenized_inputs = self.tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            padding=(self.max_length is not None),
            max_length=self.max_length,
        )

        labels = []
        words = []
        for i, label in enumerate(examples['ner_tags']):
            if self.retroactive_labels in ["same_token"]:
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:
                        label_ids.append(label[word_idx])
                    else:
                        label_ids.append(-100)
                    previous_word_idx = word_idx
                labels.append(label_ids)
                word_ids = [-1 if w is None else w for w in word_ids]
                words.append(word_ids)

            elif self.retroactive_labels == "next_token":
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:
                        label_ids.append(label[word_idx])
                    else:
                        label_ids.append(-100)
                    previous_word_idx = word_idx
                label_ids.append(-100)
                labels.append(label_ids[1:])
                word_ids = word_ids[1:] + [None]
                word_ids = [-1 if w is None else w for w in word_ids]
                words.append(word_ids)

            else:
                raise ValueError(
                    f"retroactive_labels {custom_args.retroactive_labels} is not implemented."
                )

        tokenized_inputs["labels"] = labels
        if self.model_class == "custom":
            tokenized_inputs["token_type_ids"] = words
            
        return tokenized_inputs

    @property
    def train(self) -> datasets.Dataset:
        return self.aligned.get("train")

    @property
    def val(self) -> Optional[datasets.Dataset]:
        return self.aligned.get("validation")

    @property
    def test(self) -> Optional[datasets.Dataset]:
        return self.aligned.get("test")