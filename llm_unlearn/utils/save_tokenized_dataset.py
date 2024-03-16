from llm_unlearn.utils import tokenize
import torch
from transformers import set_seed, AutoTokenizer
from datasets import load_from_disk, DatasetDict, concatenate_datasets, load_dataset
import os
import random

import argparse
model_max_length = 4096
dir = "../tokenized_dataset"

dataset_path_dict = {
    "arxiv_forget_500": {"name":"arxiv", "split":"forget"},
    "general_1k": {"name":"general", "split":"evaluation"},
    "arxiv_approximate_6k": {"name":"arxiv", "split":"approximate"},
    "github_forget_2k": {"name":"github", "split":"forget"},
    "github_approximate": {"name":"github", "split":"approximate"},
}


def adapter_load_dataset(dataset_path):
    if dataset_path.endswith("jsonl"):
        raw_dataset = load_dataset("json", data_files=dataset_path, split="train")
    else:
        raw_dataset = load_from_disk(dataset_path)
    column_names = list(raw_dataset.features)
    if "text" not in column_names:
        raw_dataset = raw_dataset.rename_column("content", "text")
    remove_feature_list = list(raw_dataset.column_names)
    remove_feature_list.remove("text")
    raw_dataset = raw_dataset.remove_columns(remove_feature_list)
    return raw_dataset


def save_tokenized_dataset(
    tokenizer_name_or_path,
    dataset_name,
    tokenize_method,
    completely_random=False,
    soft_label=False,
    top_k=int(1e10),
    top_p=1.0,
    rm_groundtruth=False,
):
    set_seed(42)
    
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        padding_side="right",
        trust_remote_code=True,
        model_max_length=model_max_length,
    )
    if tokenizer.pad_token is None:
        new_pad_token = "<pad>"
        tokenizer.add_special_tokens({"pad_token": new_pad_token})
    if dataset_name in dataset_path_dict.keys():
        dataset_path = dataset_path_dict[dataset_name]
        # import pdb
        # pdb.set_trace()
        raw_dataset = load_dataset("llmunlearn/unlearn_dataset", name=dataset_path["name"], split=dataset_path["split"])
    else:
        raise ValueError(f"dataset_name is wrong")

    save_path = os.path.join(dir, dataset_path["name"], dataset_name, tokenize_method)
    if tokenize_method == "normal":
        dataset = tokenize(raw_dataset, tokenizer, model_max_length)
    elif tokenize_method == "random_label":
        if completely_random:
            dataset = tokenize(
                raw_dataset,
                tokenizer,
                model_max_length,
                random_label=True,
                completely_random=True,
            )
            save_path = os.path.join(save_path, "completely_random")
        else:
            dataset = tokenize(
                raw_dataset,
                tokenizer,
                model_max_length,
                random_label=True,
                top_k=top_k,
                top_p=top_p,
                rm_groundtruth=rm_groundtruth,
            )
            save_path = os.path.join(save_path, f"top_k{top_k}_top_p{top_p}")
    else:
        raise ValueError(f"tokenize_method is wrong")
    if rm_groundtruth:
        save_path = save_path + "_rmgt"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = os.path.join(save_path, "tokenized_dataset.pt")
    torch.save(dataset, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_name_or_path", '-t',type=str, default=None,
                        help="tokenizer_name_or_path.")
    args = parser.parse_args()
    dataset_name_list = [
        "arxiv_forget_500",  
        "arxiv_approximate_6k",
        "general_1k",
        "github_forget_2k",
        "github_approximate",
    ]
    tokenizer_name_or_path = args.tokenizer_name_or_path
    tokenize_method_list = [
        "normal",
        "random_label"
    ]
    top_k_list = [
        1, 
        # 5, 10, 50, 500
        ]
    top_p_list = [0.2, 0.4, 0.8, 0.9, 0.95]

    # Perform regular tokenization to all datasets
    tokenize_method = "normal"
    for dataset_name in dataset_name_list:
        save_tokenized_dataset(tokenizer_name_or_path, dataset_name, tokenize_method)

    tokenize_method = "random_label"
    for dataset_name in ["arxiv_forget_500", "github_forget_2k"]:
        # Perform random label tokenization to forget sets
        save_tokenized_dataset(
            tokenizer_name_or_path, dataset_name, tokenize_method, completely_random=True
        )
        # Perform adversarial sample tokenization to forget sets, 
        # the top_k and top_p value could be adjusted.
        # In our paper we just use top_k = 1
        for top_k in top_k_list:
            save_tokenized_dataset(
                tokenizer_name_or_path,
                dataset_name,
                tokenize_method,
                top_k=top_k,
                rm_groundtruth=True,
            )

        # for top_p in top_p_list:
        #     save_tokenized_dataset(
        #         tokenizer_name_or_path,
        #         dataset_name,
        #         tokenize_method,
        #         top_p=top_p,
        #         rm_groundtruth=True,
        #     )

        # save_tokenized_dataset(
        #     tokenizer_name_or_path,
        #     dataset_name,
        #     tokenize_method,
        #     top_k=10000000,
        #     rm_groundtruth=True,
        # )
