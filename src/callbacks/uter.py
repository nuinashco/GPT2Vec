import torch
from typing import Tuple, List, Dict, Any
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl


def uter_per_layer(
        attentions: Tuple[torch.Tensor, ...],
        attn_mask: torch.Tensor
) -> List[float]:
    mask_q = attn_mask.unsqueeze(1).unsqueeze(-1)
    mask_k = attn_mask.unsqueeze(1).unsqueeze(-2)
    valid  = (mask_q & mask_k).to(attentions[0].dtype)

    eps, uters = 1e-9, []
    for A in attentions:
        tri_u = torch.triu(A, diagonal=1)
        tri_valid = torch.triu(valid, diagonal=1)

        num   = ((tri_u * tri_valid) ** 2).sum(dim=(-1, -2))
        denom = ((A * valid) ** 2).sum(dim=(-1, -2))
        uters.append((num / (denom + eps)).mean().item())
    return uters



class UTERCallback(TrainerCallback):
    def __init__(self, device: str | None = None):
        self.device = device


    def on_train_batch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model,
        **kwargs: Dict[str, Any],
    ):
        if not control.should_log:
            return

        inputs = kwargs["inputs"]
        device = self.device or next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        if (isinstance(kwargs["outputs"], dict)
           and "attentions" in kwargs["outputs"]):
            attentions = kwargs["outputs"]["attentions"]
        else:
            with torch.no_grad():
                outs = model(**inputs,
                             output_attentions=True,
                             return_dict=True)
            attentions = outs.attentions

        uters = uter_per_layer(attentions, inputs["attention_mask"])

        logs = {f"uter/L{i}": v for i, v in enumerate(uters)}
        logs["uter/mean"] = sum(uters) / len(uters)
        
        self.log(logs)
