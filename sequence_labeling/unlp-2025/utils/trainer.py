import torch
from transformers import Trainer, TrainingArguments
import numpy as np
from transformers import Trainer, TrainingArguments
from typing import Any
from tqdm.autonotebook import tqdm
from transformers.trainer_utils import EvalPrediction


def extract_chars_from_spans(spans):
    """
    Given a list of spans (each a tuple (start, end)),
    return a set of character indices for all spans.
    """
    char_set = set()
    for start, end in spans:
        # Each span covers positions start, start+1, ..., end-1.
        char_set.update(range(start, end))
    return char_set


class SpanIdentificationTrainer(Trainer):
    def __init__(
        self,
        model: Any = None,
        args: TrainingArguments = None,
        data_collator: Any = None,
        train_dataset: Any = None,
        eval_dataset: Any = None,
        tokenizer: Any = None,
        desired_positive_ratio: float = 0.25,
        **kwargs,
    ):
        """
        Initialize the Trainer with our custom compute_metrics.
        """
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=self.compute_metrics,  # assign our custom compute_metrics
            **kwargs,
        )
        self.desired_positive_ratio = desired_positive_ratio

    def _calculate_inner_metric(self, gt_spans_all, pred_spans_all):
        total_true_chars = 0
        total_pred_chars = 0
        total_overlap_chars = 0
        for true_spans, pred_spans in zip(gt_spans_all, pred_spans_all):
            if isinstance(true_spans, str):
                try:
                    true_spans = eval(true_spans)
                except Exception:
                    true_spans = []
                    
            # Convert spans to sets of character indices.
            true_chars = extract_chars_from_spans(true_spans)
            pred_chars = extract_chars_from_spans(pred_spans)
            
            total_true_chars += len(true_chars)
            total_pred_chars += len(pred_chars)
            total_overlap_chars += len(true_chars.intersection(pred_chars))
            
            union_chars = true_chars.union(pred_chars)
            
        # Compute precision, recall, and F1.
        precision = total_overlap_chars / total_pred_chars if total_pred_chars > 0 else 0
        recall = total_overlap_chars / total_true_chars if total_true_chars > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        return metrics
        
        
    def compute_metrics(self, eval_pred: EvalPrediction) -> dict:
        eval_dataset = self.eval_dataset
        logits, labels = eval_pred
        probabilities = torch.softmax(torch.tensor(logits), dim=-1).cpu().numpy()
    
        thresholds = np.linspace(0.1, 0.5, num=41)
        results = []
        best_f1 = -1
        best_metrics = None
    
        for thold in tqdm(thresholds):
            # Apply thresholding instead of argmax
            predictions = (probabilities[:, :, 1] >= thold).astype(int)
    
            true_predictions = [
                [p for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
    
            pred_spans_all = []
            for pred, offsets in zip(true_predictions, eval_dataset['offset_mapping']):
                samplewise_spans = []
                current_span = None
                for token_label, span in zip(pred, offsets):
                    if token_label == 1:  # If the current token is labeled as an entity (1)
                        if current_span is None:
                            current_span = [span[0], span[1]]  # Start a new span
                        else:
                            current_span[1] = span[1]  # Extend the span to include the current token
                    else:  # If token_label == 0 (not an entity)
                        if current_span is not None:
                            samplewise_spans.append(tuple(current_span))  # Save completed span
                            current_span = None  # Reset for the next entity
    
                # If the last token was part of a span, save it
                if current_span is not None:
                    samplewise_spans.append(tuple(current_span))
    
                pred_spans_all.append(samplewise_spans)
    
            # Store results for this threshold
            current_metrics = self._calculate_inner_metric(eval_dataset['trigger_words'], pred_spans_all)
            if current_metrics['f1'] >= best_f1:
                best_f1 = current_metrics['f1']
                best_metrics = current_metrics
                best_metrics['thold'] = thold
                
            
            results.append(current_metrics)
        return best_metrics