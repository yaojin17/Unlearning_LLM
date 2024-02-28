import torch
from transformers import Trainer, DataCollatorWithPadding
from torch.utils.data import Dataset, SequentialSampler
from typing import Dict, Optional, Sequence
import inspect


class AscentPlusKLDivergenceTrainer(Trainer):
    def __init__(self, pretrain_model=None, **kwargs):
        super().__init__(**kwargs)
        device = self.accelerator.device
        pretrain_model.to(device)
        self.pretrain_model = pretrain_model

    def compute_loss(self, model, inputs, return_outputs=False):
        if "factor" not in inputs.keys():
            return super().compute_loss(model, inputs, return_outputs)
        factors = inputs.pop("factor")
        negative_inputs = {key: val[factors == -1] for key, val in inputs.items()}
        positive_inputs = {key: val[factors != -1] for key, val in inputs.items()}
        if len(negative_inputs["input_ids"]) != 0:
            outputs = model(**negative_inputs)
            negative_loss = outputs.loss * -1
        else:
            negative_loss = 0
        if len(positive_inputs["input_ids"]) != 0:
            positive_loss = (
                compute_kl(
                    self.pretrain_model, model, positive_inputs, self.accelerator.device
                )
                * factors[factors != -1][0]
            )
        else:
            positive_loss = 0
        loss = negative_loss + positive_loss
        return (loss, outputs) if return_outputs else loss

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        return SequentialSampler(self.train_dataset)

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(
                set(["label", "label_ids"] + self.label_names)
            )
            self._signature_columns.append("factor")

# This function is based on code from kevinyaobytedance/llm_unlearn
# available at https://github.com/kevinyaobytedance/llm_unlearn
# Licensed under MIT 
def compute_kl(pretrained_model, current_model, batch, device):
    """
    Compute *forward* KL as the normal utility loss.

    Args:
        pretrained_model: reference model which is the pretrained (original) model.
        current_model: The current unlearning model.
        batch: A batch of normal data.
        device: GPU device.

    Returns:
       The KL loss.
    """
    normal_outputs = current_model(
        batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        labels=batch["labels"].to(device),
    )

    with torch.no_grad():
        pretrained_outputs = pretrained_model(
            batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
        )

    # P: pretrained model; Q: current model.
    prob_p = torch.nn.functional.softmax(pretrained_outputs.logits, -1)
    prob_q = torch.nn.functional.softmax(normal_outputs.logits, -1)

    loss = -(prob_p * torch.log(prob_q + 1e-12)).sum(-1).mean()

    return loss
