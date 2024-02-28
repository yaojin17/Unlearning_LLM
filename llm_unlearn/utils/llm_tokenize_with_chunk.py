import copy
import numpy as np
import torch
from llm_unlearn.utils import compute_logits_and_samples_for_batch
from transformers import BatchEncoding
from datasets import Dataset, DatasetDict
from tqdm import trange

def tokenize(
    dataset,
    tokenizer,
    max_length,
    random_label=False,
    completely_random=False,
    top_k=50,
    top_p=1.0,
    rm_groundtruth=False,
):
    column_names = ["text",]
    text_column_name = "text"

    def chunk_and_pad(examples):
        result = {"input_ids": [], "attention_mask": []}

        for i in range(len(examples["input_ids"])):
            input_ids = examples["input_ids"][i]
            attention_mask = examples["attention_mask"][i]

            num_chunks = len(input_ids) // max_length + int(
                len(input_ids) % max_length != 0
            )
            # num_chunks = len(input_ids) // max_length
            for j in range(num_chunks):
                start_index = j * max_length
                end_index = start_index + max_length

                chunk_input_ids = input_ids[start_index:end_index]
                chunk_attention_mask = attention_mask[start_index:end_index]
                if len(chunk_input_ids) < 5:
                    continue
                padding_length = max_length - len(chunk_input_ids)
                chunk_input_ids.extend([tokenizer.pad_token_id] * padding_length)
                chunk_attention_mask.extend([0] * padding_length)

                result["input_ids"].append(chunk_input_ids)
                result["attention_mask"].append(chunk_attention_mask)

        return result

    def tokenize_function(examples):
        output = tokenizer(examples[text_column_name])
        output = chunk_and_pad(output)
        output = BatchEncoding(output, tensor_type="pt")
        output["labels"] = copy.deepcopy(output["input_ids"])
        if random_label:
            if completely_random:
                special_tokens = [
                    tokenizer.pad_token_id,
                    tokenizer.eos_token_id,
                    tokenizer.bos_token_id,
                    tokenizer.unk_token_id,
                ]
                for j, sequence in enumerate(output["input_ids"]):
                    for i, token_id in enumerate(sequence):
                        if token_id not in special_tokens:
                            output["labels"][j][i] = np.random.choice(
                                tokenizer.vocab_size
                            )
                        else:
                            output["labels"][j][i] = token_id
            else:
                for i in range(len(output["input_ids"])):
                    input = {
                        key: value[i].unsqueeze(0) for key, value in output.items()
                    }
                    input = BatchEncoding(input, tensor_type="pt")
                    _, sampled_token_ids = compute_logits_and_samples_for_batch(
                        input,
                        tokenizer,
                        top_k=top_k,
                        top_p=top_p,
                        rm_groundtruth=rm_groundtruth,
                    )
                    sampled_token_ids = sampled_token_ids.squeeze(0)
                    indices = torch.nonzero(
                        output["input_ids"][i] == tokenizer.pad_token_id, as_tuple=True
                    )

                    sampled_token_ids[indices] = tokenizer.pad_token_id
                    output["labels"][i] = sampled_token_ids

        pad_token_mask = output["labels"] == tokenizer.pad_token_id
        output["labels"] = torch.where(
            pad_token_mask,
            torch.tensor(-100, device=output["labels"].device),
            output["labels"],
        )
        return output

    return dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        # batch_size=1,
        desc="Running tokenizer",
        load_from_cache_file=False,
        # num_proc=100
    )
