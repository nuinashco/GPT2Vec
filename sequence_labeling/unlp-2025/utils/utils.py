import os
import random
import numpy as np
import torch

def set_seeds(seed):
    """Set seeds for reproducibility """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def find_class_balance_threshold(desired_positive_ratio, probabilities, labels):
    """Finds the threshold that achieves the desired positive class balance."""
    best_th = 0.5  # Default starting point
    best_diff = float("inf")
    optimal_th = best_th
    
    for thold in np.linspace(0.01, 0.99, num=100):
        predictions = (probabilities[:, :, 1] >= thold).astype(int)
        true_predictions = [
            [p for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        total_pos = sum([sum(row for row in prediction) for prediction in true_predictions])
        total = sum([len(prediction) for prediction in true_predictions])
        
        positive_ratio = total_pos / total if total > 0 else 0
        
        diff = abs(positive_ratio - desired_positive_ratio)
        if diff < best_diff:
            best_diff = diff
            optimal_th = thold
    
    return optimal_th


def inference_aggregation(probabilities, labels, offset_mappings, thold):
    predictions = (probabilities[:, :, 1] >= thold).astype(int)
    true_predictions = [
        [p for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)
    ]
    pred_spans_all = []
    for pred, offsets in zip(true_predictions, offset_mappings):
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
    return [str(row) for row in pred_spans_all]