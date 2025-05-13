"""
TODO:
- Clean class to calculate the attention geometry scores.
- LoRA handling (with and without LoRA weights merge).

"""

from abc import ABC, abstractmethod
from typing import Iterable, List
import torch, functools


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


class AttentionExtractor:
    def __init__(self, model, q_path: str, k_path: str, attention_type: str | None = None):
        self.model = model
        self.q_path, self.k_path = q_path, k_path
        self.attention_type = attention_type

        self.layer_count = model.config.num_hidden_layers
        self.d  = model.config.hidden_size
        self.dh = self.d // model.config.num_attention_heads

    # def layer_count(self) -> int:
    #     prefix, _ = self.q_path.split("[layer_idx].")
    #     return len(rgetattr(self.model, prefix))


    def matrix(self, layer_idx: int) -> torch.Tensor:
        Wq = self._raw(self.q_path, layer_idx).T.detach()
        Wk = self._raw(self.k_path, layer_idx).T.detach()

        if self.attention_type == "grouped":
            Wk = Wk.view(Wk.shape[0], self.dh, Wk.shape[1] // self.dh)
            rep = (Wq.shape[0] // self.dh) // Wk.shape[-1]
            Wk = Wk.repeat_interleave(rep, 0).view(Wq.shape[0], Wq.shape[0])

        return Wq @ Wk.T

    def _raw(self, path: str, idx: int) -> torch.Tensor:
        layers_path, matrix_path = path.split("[layer_idx].")
        layer_module = rgetattr(self.model, layers_path)[idx]
        return rgetattr(layer_module, matrix_path)


class AttentionScorer(ABC):
    def __init__(self, extractor: AttentionExtractor):
        self.extractor = extractor

    def __call__(self, layers: Iterable[int] | int | None = None) -> List[float] | float:
        if layers is None:
            layers = range(self.extractor.layer_count)
        if isinstance(layers, int):
            layers = [layers]

        with torch.no_grad():
            scores = [self._score(self.extractor.matrix(i)) for i in layers]
        return scores if len(scores) > 1 else scores[0]

    @abstractmethod
    def _score(self, A: torch.Tensor) -> float: ...


class SymmetryScore(AttentionScorer):
    def _score(self, A: torch.Tensor) -> float:
        sym = 0.5 * (A + A.T)
        score = (sym ** 2).sum() / (A ** 2).sum()

        # like in paper
        score = 2 * score - 1

        return score.detach().item()


class DirectionalityScore(AttentionScorer):
    def _score(self, A: torch.Tensor, *, num_std: int = 2) -> float:
        row, col = torch.norm(A, dim=1), torch.norm(A, dim=0)
        rt, ct = row.mean() + num_std*row.std(), col.mean() + num_std*col.std()
        r_exc, c_exc = torch.sum(row[row > rt] - rt), torch.sum(col[col > ct] - ct)
        total = r_exc + c_exc
        score = 0.0 if total == 0 else (c_exc - r_exc) / total

        # like in paper
        score = -1 * score

        return score.detach().item()