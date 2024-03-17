import json
from llm_unlearn.utils import adapter_load_dataset
from llm_unlearn.utils import tokenize
import transformers
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import datasets
from datasets import load_dataset
import torch
import argparse
from tqdm import trange
import os


class AdvSupervisedDataset(Dataset):
    """Dataset for adv supervised fine-tuning."""

    def __init__(
        self,
        negative_data_dict,
        positive_data_dict,
        data_args,
    ):
        super(AdvSupervisedDataset, self).__init__()

        print("Formatting inputs...")
        negative_data_dict = negative_data_dict.to_dict()
        positive_data_dict = positive_data_dict.to_dict()
        self.input_ids = []
        self.labels = []
        self.attention_mask = []
        self.factor = []
        for i in trange(len(negative_data_dict["input_ids"])):
            self.input_ids.append(negative_data_dict["input_ids"][i])
            self.labels.append(negative_data_dict["labels"][i])
            self.attention_mask.append(negative_data_dict["attention_mask"][i])
            self.factor.append(-1)
            self.input_ids.extend(
                positive_data_dict["input_ids"][
                    i * data_args.positive_ratio : (i + 1) * data_args.positive_ratio
                ]
            )
            self.labels.extend(
                positive_data_dict["labels"][
                    i * data_args.positive_ratio : (i + 1) * data_args.positive_ratio
                ]
            )
            self.attention_mask.extend(
                positive_data_dict["attention_mask"][
                    i * data_args.positive_ratio : (i + 1) * data_args.positive_ratio
                ]
            )
            self.factor.extend([data_args.positive_factor] * data_args.positive_ratio)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
            factor=self.factor[i],
        )
    def select(self, selection_range):
        # Create a new instance of AdvSupervisedDataset with the same data dictionaries and args,
        # but do not fill the attributes with data yet.
        new_dataset = AdvSupervisedDataset(datasets.Dataset.from_dict({"input_ids": []}), datasets.Dataset.from_dict({"input_ids": []}), None)
        
        # Manually set the selected items for each attribute
        new_dataset.input_ids = [self.input_ids[i] for i in selection_range]
        new_dataset.labels = [self.labels[i] for i in selection_range]
        new_dataset.attention_mask = [self.attention_mask[i] for i in selection_range]
        new_dataset.factor = [self.factor[i] for i in selection_range]
        
        return new_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        help="your tokenizer name or path.",
    )
    parser.add_argument(
        "--positive_ratio",
        type=int,
        default=1,
        help="number of positive examples per negative example.",
    )
    parser.add_argument(
        "--positive_factor",
        type=float,
        default=1.0,
        help="Weight on the positive examples' loss.",
    )
    data_args = parser.parse_args()

    tokenizer_name_or_path = data_args.tokenizer_name_or_path
    model_max_length = 4096
    domain2dir = {
        "arxiv": "arxiv/arxiv_forget_500",
        "github": "github/github_forget_2k"
    }
    domains= ["arxiv", "github"]
    for negative_domain in domains:
        negative_train = load_dataset("llmunlearn/unlearn_dataset", name=negative_domain, split="forget")
        for positive_domain in ["general", negative_domain]:
            # For GA + Gradient descent on general retain data, use the following settings:
            # positive_train = load_dataset("llmunlearn/unlearn_dataset", name="general", split="retain")
            # For GA + Gradient descent on in-distribution retain data, use the following settings:
            # positive_train = load_dataset("llmunlearn/unlearn_dataset", name="github", split="retain")

            positive_train = load_dataset("llmunlearn/unlearn_dataset", name=positive_domain, split="retain")
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name_or_path,
                padding_side="right",
                trust_remote_code=True,
                model_max_length=model_max_length,
            )

            negative_dataset = tokenize(negative_train, tokenizer, model_max_length)
            # num_positive = len(negative_dataset) * data_args.positive_ratio
            num_positive = len(positive_train)
            positive_dataset = tokenize(
                positive_train.select(list(range(num_positive))), tokenizer, model_max_length
            )
            train_dataset = AdvSupervisedDataset(negative_dataset, positive_dataset, data_args)
            save_path = os.path.join("../tokenized_dataset", domain2dir[negative_domain], "ascent_plus_descent")
            if positive_domain == "general":
                save_path += "_general"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_path = os.path.join(save_path, "tokenized_dataset.pt")
            torch.save(train_dataset, save_path)
