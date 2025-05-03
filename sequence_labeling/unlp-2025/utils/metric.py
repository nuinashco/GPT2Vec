import pandas as pd
import numpy as np
import pandas.api.types
from sklearn.metrics import f1_score
import ast


class ParticipantVisibleError(Exception):
    """Custom exception for participant-visible errors."""
    pass


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    Compute span-level F1 score based on overlap.

    Parameters:
    - solution (pd.DataFrame): Ground truth DataFrame with row ID and token labels.
    - submission (pd.DataFrame): Submission DataFrame with row ID and token labels.
    - row_id_column_name (str): Column name for the row identifier.

    Returns:
    - float: The token-level weighted F1 score.

    Example:
    >>> solution = pd.DataFrame({
    ...     "id": [1, 2, 3],
    ...     "trigger_words": [[(612, 622), (725, 831)], [(300, 312)], []]
    ... })
    >>> submission = pd.DataFrame({
    ...     "id": [1, 2, 3],
    ...     "trigger_words": [[(612, 622), (700, 720)], [(300, 312)], [(100, 200)]]
    ... })
    >>> score(solution, submission, "id")
    0.16296296296296295
    """
    if not all(col in solution.columns for col in ["id", "trigger_words"]):
        raise ValueError("Solution DataFrame must contain 'id' and 'trigger_words' columns.")
    if not all(col in submission.columns for col in ["id", "trigger_words"]):
        raise ValueError("Submission DataFrame must contain 'id' and 'trigger_words' columns.")
    
    def safe_parse_spans(trigger_words):
        if isinstance(trigger_words, str):
            try:
                return ast.literal_eval(trigger_words)
            except (ValueError, SyntaxError):
                return []
        if isinstance(trigger_words, (list, tuple, np.ndarray)):
            return trigger_words
        return []

    def extract_tokens_from_spans(spans):
        tokens = set()
        for start, end in spans:
            tokens.update(range(start, end))
        return tokens
    
    solution = solution.copy()
    submission = submission.copy()

    solution["trigger_words"] = solution["trigger_words"].apply(safe_parse_spans)
    submission["trigger_words"] = submission["trigger_words"].apply(safe_parse_spans)

    merged = pd.merge(
        solution,
        submission,
        on="id",
        suffixes=("_solution", "_submission")
    )

    total_true_tokens = 0
    total_pred_tokens = 0
    overlapping_tokens = 0

    for _, row in merged.iterrows():
        true_spans = row["trigger_words_solution"]
        pred_spans = row["trigger_words_submission"]

        true_tokens = extract_tokens_from_spans(true_spans)
        pred_tokens = extract_tokens_from_spans(pred_spans)

        total_true_tokens += len(true_tokens)
        total_pred_tokens += len(pred_tokens)
        overlapping_tokens += len(true_tokens & pred_tokens)

    precision = overlapping_tokens / total_pred_tokens if total_pred_tokens > 0 else 0
    recall = overlapping_tokens / total_true_tokens if total_true_tokens > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return f1