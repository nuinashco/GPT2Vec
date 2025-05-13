from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel

from transformers.modeling_outputs import TokenClassifierOutput


class ModelForWordTask(PreTrainedModel):
    def __init__(self, config, model, merge_subwords=False, **model_args):
        PreTrainedModel.__init__(self, config)
        self.model = model
        self.merge_subwords = merge_subwords

        if (
            hasattr(config, "classifier_dropout")
            and config.classifier_dropout is not None
        ):
            classifier_dropout = config.classifier_dropout
        elif hasattr(config, "hidden_dropout") and config.hidden_dropout is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1

        self.dropout = nn.Dropout(classifier_dropout)
        self.num_labels = config.num_labels
        self.classifier = nn.Linear(config.hidden_size, config.num_labels).to(
            model_args.get("torch_dtype")
        )

        # Initialize weights and apply final processing
        self.post_init()

    def _merge_subwords(self, hidden_states, token_type_ids, attention_mask):
        new_hidden_states = hidden_states.clone()
        for b in range(hidden_states.shape[0]):
            for w in torch.arange(0, token_type_ids[b].max() + 1):
                words_w = (token_type_ids[b] == w) * (attention_mask[b] > 0)
                new_hidden_states[b][words_w] = torch.mean(
                    hidden_states[b][words_w], dim=0
                ).repeat(sum(words_w), 1)
        return new_hidden_states

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        if self.merge_subwords:
            hidden_states = self._merge_subwords(
                hidden_states, token_type_ids, attention_mask
            )

        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs.hidden_states
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
        )