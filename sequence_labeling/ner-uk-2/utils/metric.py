import numpy as np
import evaluate

class NerMetrics:
    def __init__(self, label2id):
        self.metric = evaluate.load("seqeval")
        self.label_list = list(label2id.keys())

    def compute_metrics(self, p):
        predictions, labels = p
        predictions = np.argmax(predictions[0], axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = self.metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"]
        }