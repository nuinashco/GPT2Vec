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
        ignore_id: int = -100
    ) -> None:
        self.tokenizer = tokenizer
        self.ignore_id = ignore_id
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
        tokenized_inputs =self.tokenizer(
            examples["tokens"],
            truncation=True, is_split_into_words=True
        )

        all_labels = examples["ner_tags"]
        new_labels = []
        stored_word_ids = []
        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(self.align_labels_with_tokens(labels, word_ids))
            stored_word_ids.append(word_ids)

        tokenized_inputs["labels"] = new_labels
        tokenized_inputs["word_ids"] = stored_word_ids
        return tokenized_inputs


    def align_labels_with_tokens(self, labels, word_ids):
        new_labels = []
        current_word = None
        for word_id in word_ids:
            if word_id != current_word:
                # Start of a new word!
                current_word = word_id
                label = -100 if word_id is None else labels[word_id]
                new_labels.append(label)
            elif word_id is None:
                # Special token
                new_labels.append(-100)
            else:
                # Same word as previous token
                label = labels[word_id]
                # If the label is B-XXX we change it to I-XXX
                if label % 2 == 1:
                    label += 1
                new_labels.append(label)

        return new_labels

    # Does it make any sense?
    # def align_tokens_with_labels(self, token_predictions, word_ids):
    #     max_word_id = int(max([wid for wid in word_ids.numpy() if ~np.isnan(wid)], default=-1))
    #     word_level_predictions = torch.zeros(max_word_id + 1, dtype=torch.long) + self.ignore_id
    #
    #     # Initialize with ignore_id
    #     for token_idx, word_idx in enumerate(word_ids.numpy()):
    #         if np.isnan(word_idx):
    #             continue
    #
    #         word_idx = int(word_idx)
    #         token_pred = token_predictions[token_idx]
    #         if token_predictions[token_idx] == self.ignore_id:
    #             continue
    #
    #         if word_level_predictions[word_idx] == self.ignore_id:
    #             word_level_predictions[word_idx] = token_pred
    #
    #     return word_level_predictions

    @property
    def train(self) -> datasets.Dataset:
        return self.aligned.get("train")

    @property
    def val(self) -> Optional[datasets.Dataset]:
        return self.aligned.get("validation")

    @property
    def test(self) -> Optional[datasets.Dataset]:
        return self.aligned.get("test")