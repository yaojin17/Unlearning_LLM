import torch
from transformers import Trainer, DataCollatorWithPadding
from torch.utils.data import Dataset, SequentialSampler
from typing import Dict, Optional, Sequence
import inspect


class AscentPlusDescentTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        if "factor" not in inputs.keys():
            return super().compute_loss(model, inputs, return_outputs)
        factors = inputs.pop("factor")
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["labels"]
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = loss_fct(
            shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1)
        )
        valid_counts = (shift_labels != -100).sum(dim=-1).float()

        loss = loss.view(shift_logits.size(0), -1)

        loss = loss.sum(dim=-1) / valid_counts
        adjusted_loss = (loss * factors).mean()
        return (adjusted_loss, outputs) if return_outputs else adjusted_loss

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        return SequentialSampler(self.train_dataset)
    
    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(set(["label", "label_ids"] + self.label_names))
            self._signature_columns.append('factor')


class AscentPlusDescentDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        batch = super().__call__(features)
        # print([f["factor"] for f in features])
        if "facotr" in features[0].keys():
            batch["factor"] = torch.tensor([f["factor"] for f in features])
        return batch