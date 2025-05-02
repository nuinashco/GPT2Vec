from typing import List, Dict, Any, Optional, Union, Tuple, Sequence

import datasets
import numpy as np
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase


IGNORE_ID = -100


class NerUKDataset:
    """
    Wrapper for *NER‑UK 2.0* dataset providing token‑aligned splits ready for HuggingFace Trainer.
    
    This class handles loading the dataset, tokenizing text, and aligning NER labels with 
    tokenized inputs while properly handling subword tokenization.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        dataset_name: str = "Goader/ner-uk-2.0",
        splits: Tuple[str, ...] = ("train", "validation", "test"),
    ) -> None:
        """
        Initialize the NER-UK dataset wrapper.
        
        Args:
            tokenizer: HuggingFace tokenizer to use for text tokenization
            dataset_name: Name of the dataset to load from HuggingFace datasets
            splits: Data splits to load and process (default: all three standard splits)
        """
        self.tokenizer = tokenizer
        self.raw_ds = load_dataset(dataset_name, trust_remote_code=True)

        self.label_list: List[str] = self.raw_ds["train"].features["ner_tags"].feature.names
        self.label2id: Dict[str, int] = {label: i for i, label in enumerate(self.label_list)}
        self.id2label: Dict[int, str] = {i: label for i, label in enumerate(self.label_list)}

        self.aligned: Dict[str, datasets.Dataset] = {}
        for split in splits:
            if split not in self.raw_ds:
                continue
                
            self.aligned[split] = self.raw_ds[split].map(
                self._tokenize_and_align_labels,
                batched=True,
                desc=f"Tokenizing {split} split",
            )

            self.aligned[split].set_format(type="torch")

    @staticmethod
    def _align_labels_with_tokens(
        labels: List[int], 
        word_ids: List[Optional[int]]
    ) -> List[int]:
        """
        Align token-level labels with wordpiece/subword tokens.
        
        For tokens that are part of the same word:
        - First token gets the original label
        - Subsequent tokens: B-XXX labels are converted to I-XXX (odd → even indices)
        - Special tokens (word_id is None) get IGNORE_ID
        
        Args:
            labels: Original word-level NER labels (integers)
            word_ids: Mapping from token positions to original word positions
            
        Returns:
            List of aligned labels for each token
        """
        new_labels: List[int] = []
        current_word = None
        
        for word_id in word_ids:
            if word_id != current_word:
                # New word or special token
                current_word = word_id
                label = IGNORE_ID if word_id is None else labels[word_id]
                new_labels.append(label)
            elif word_id is None:
                new_labels.append(IGNORE_ID)
            else:
                label = labels[word_id]
                if label % 2 == 1:
                    label += 1
                new_labels.append(label)
                
        return new_labels

    def _tokenize_and_align_labels(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tokenize input examples and align labels with resulting tokens.
        
        Args:
            examples: Batch of examples with 'tokens' and 'ner_tags' fields
            
        Returns:
            Dictionary with tokenizer outputs and aligned labels
        """

        tokenized = self.tokenizer(
            examples["tokens"],
            is_split_into_words=True,
            truncation=True,
            return_attention_mask=True,
        )

        aligned_labels: List[List[int]] = []
        stored_word_ids: List[List[Optional[int]]] = []
        
        for i, labels in enumerate(examples["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            aligned_labels.append(self._align_labels_with_tokens(labels, word_ids))
            stored_word_ids.append(word_ids)

        tokenized["labels"] = aligned_labels
        tokenized["word_ids"] = stored_word_ids

        return tokenized

    @property
    def train(self) -> datasets.Dataset:
        """Get the processed training dataset."""
        return self.aligned["train"]

    @property
    def val(self) -> Optional[datasets.Dataset]:
        """Get the processed validation dataset, if available."""
        return self.aligned.get("validation")

    @property
    def test(self) -> Optional[datasets.Dataset]:
        """Get the processed test dataset, if available."""
        return self.aligned.get("test")

    def predictions_to_conll(
        self,
        tokenized_split: datasets.Dataset,
        pred_ids: Union[np.ndarray, List[List[int]]],
    ) -> str:
        """
        Convert model predictions to CoNLL format.
        
        Args:
            tokenized_split: Dataset split with tokenized data and word_ids
            pred_ids: Predicted label IDs for each token
            
        Returns:
            String in CoNLL format with words and their predicted labels
        """
        lines: List[str] = []
        
        for example, preds in zip(tokenized_split, pred_ids):
            word_ids = example["word_ids"]
            tokens = example["tokens"]
            
            current_word_idx = None
            token_buf = []
            tag_buf = []
            
            for idx, word_id in enumerate(word_ids):
                if word_id is None:
                    continue
                    
                if word_id != current_word_idx:
                    if token_buf:
                        lines.append(f"{token_buf[0]} {tag_buf[0]}")
                        
                    token_buf = [tokens[word_id]]
                    tag_buf = [self.id2label[preds[idx]]]
                    current_word_idx = word_id
                else:
                    continue
                    
            if token_buf:
                lines.append(f"{token_buf[0]} {tag_buf[0]}")
                
            lines.append("")
            
        return "\n".join(lines)